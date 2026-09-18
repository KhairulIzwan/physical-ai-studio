# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import datetime
import gc
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from core.logging.utils import job_logging_ctx
from models.utils import load_policy, setup_policy
from services.snapshot_service import SnapshotService
from settings import get_settings

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass


from loguru import logger
from physicalai.data import LeRobotDataModule
from physicalai.export import ExportablePolicyMixin
from physicalai.train import Trainer

from schemas import Job, Model, Snapshot
from schemas.base_job import JobStatus
from schemas.job import TrainJobPayload
from services import DatasetService, ModelService
from services.event_processor import EventType
from services.job_service import JobService
from services.training_service import (
    TrainingLogCallback,
    TrainingService,
    TrainingTrackingCallback,
    TrainingTrackingDispatcher,
)
from utils.device import get_lightning_strategy, get_torch_device
from workers.base import BaseProcessWorker

SCHEDULE_INTERVAL_SEC = 5


class TrainingWorker(BaseProcessWorker):
    ROLE = "TrainingWorker"

    def __init__(self, stop_event: EventClass, interrupt_event: EventClass, event_queue: mp.Queue):
        super().__init__(stop_event=stop_event)
        self.queue = event_queue
        self.interrupt_event = interrupt_event

    async def run_loop(self) -> None:
        job_service = JobService()
        logger.info("Training Worker is running")
        while not self.should_stop():
            settings = get_settings()

            job = await job_service.get_pending_train_job()
            if job is not None:
                with job_logging_ctx(job_id=str(job.id)):
                    payload = TrainJobPayload.model_validate(job.payload)
                    id = uuid4()

                    base_model = None
                    if payload.base_model_id is not None:
                        base_model = await ModelService.get_model_by_id(payload.base_model_id)

                    dataset = await DatasetService.get_dataset_by_id(payload.dataset_id)
                    model_dir = Path(str(settings.models_dir / str(id)))
                    snapshot_dir = settings.snapshot_dir / SnapshotService.generate_snapshot_folder_name()
                    snapshot = await SnapshotService.create_snapshot_for_dataset(dataset, destination=snapshot_dir)

                    model = Model(
                        id=id,
                        project_id=payload.project_id,
                        dataset_id=payload.dataset_id,
                        path=str(model_dir),
                        name=payload.model_name,
                        snapshot_id=snapshot.id,
                        policy=payload.policy,
                        properties={},
                        train_job_id=job.id,
                        parent_model_id=payload.base_model_id,
                        version=base_model.version + 1 if base_model else 1,
                        created_at=None,
                    )

                    self.interrupt_event.clear()
                    await asyncio.create_task(self._train_model(job, model, snapshot, payload, base_model))
            self.stop_aware_sleep(0.5)

    async def setup(self) -> None:
        await super().setup()
        with logger.contextualize(worker=self.__class__.__name__):
            await TrainingService.abort_orphan_jobs()

    async def teardown(self) -> None:
        await super().teardown()
        with logger.contextualize(worker=self.__class__.__name__):
            await TrainingService.abort_orphan_jobs()

    async def _train_model(
        self, job: Job, model: Model, snapshot: Snapshot, payload: TrainJobPayload, base_model: Model | None = None
    ):
        settings = get_settings()
        await JobService.update_job(
            job=job,
            update={
                "status": JobStatus.RUNNING,
                "message": "Training started",
                "start_time": datetime.datetime.now(tz=datetime.UTC),
            },
        )
        dispatcher = TrainingTrackingDispatcher(
            job_id=job.id,
            event_queue=self.queue,
            interrupt_event=self.interrupt_event,
        )
        try:
            path = Path(model.path)
            cache_path = settings.cache_dir / str(job.id)

            # Resolve training device -- explicit from payload or auto-detected
            device_type = payload.device.type if payload.device else None
            device_index = payload.device.index if payload.device else None

            accelerator = get_torch_device(device_type)

            l_dm = LeRobotDataModule(
                repo_id="snapshot",  # doesnt matter for loading the data.
                root=snapshot.path,
                train_batch_size=payload.batch_size,
                num_workers=payload.num_workers,
                val_split=payload.val_split,
            )

            if base_model is not None:
                policy = load_policy(base_model, compile_model=payload.compile_model)
            else:
                policy = setup_policy(
                    model,
                    compile_model=payload.compile_model,
                    freeze_vision_encoder=payload.freeze_vision_encoder,
                )

            precision = str(payload.precision)
            strategy = get_lightning_strategy(device_type)
            devices = [device_index] if device_index is not None else 1

            checkpoint_callback = ModelCheckpoint(
                dirpath=cache_path,
                filename="model",  # filename without suffix
                save_top_k=1,
                monitor="val/loss",
                mode="min",
                # load_policy() only ever restores weights for inference/fine-tuning,
                # never optimizer state, so skip it: Adam's per-param moment buffers
                # roughly triple the checkpoint's memory/disk footprint for large
                # models and have caused OOM kills during checkpoint writes.
                save_weights_only=True,
            )
            csv_logger = CSVLogger(cache_path.parent, name=cache_path.stem)

            def _create_trainer() -> Trainer:
                return Trainer(
                    logger=csv_logger,
                    callbacks=[
                        checkpoint_callback,
                        TrainingTrackingCallback(
                            shutdown_event=self._stop_event,
                            interrupt_event=self.interrupt_event,
                            dispatcher=dispatcher,
                        ),
                        TrainingLogCallback(),
                    ],
                    accelerator=accelerator,
                    strategy=strategy,
                    devices=devices,
                    max_steps=payload.max_steps,
                    auto_scale_batch_size=payload.auto_scale_batch_size,
                    precision=precision,
                    check_val_every_n_epoch=1,
                )

            trainer = _create_trainer()

            dispatcher.start()
            trainer.fit(model=policy, datamodule=l_dm)

            del l_dm
            self._release_training_memory()

            final_checkpoint = cache_path / "model.ckpt"
            if not final_checkpoint.exists():
                self._save_final_checkpoint_if_missing(trainer, final_checkpoint)

            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(cache_path, path)

            export_policy = policy
            if payload.compile_model and model.policy in ["act", "smolvla"]:
                try:
                    logger.info("Reloading non-compiled policy for export")
                    export_policy = load_policy(model, compile_model=False)
                except Exception as e:
                    logger.warning("Failed to reload non-compiled policy for export; falling back to trained policy")
                    logger.exception(e)

            await self._export_policy(policy=export_policy, path=path, job=job)

            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.COMPLETED, message="Training finished"
            )
            model = await ModelService.create_model(model)
            self.queue.put((EventType.MODEL_UPDATE, model))
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            job = await JobService.update_job_status(
                job_id=job.id, status=JobStatus.FAILED, message=f"Training failed: {e}"
            )
        self.interrupt_event.set()
        if dispatcher.is_alive():
            dispatcher.join(timeout=10)
        self.queue.put((EventType.JOB_UPDATE, job))

    @staticmethod
    def _release_training_memory() -> None:
        """Release cached tensors/allocator memory after fit() and before checkpoint/export.

        On memory-constrained (iGPU, shared-RAM) hosts, dataloader workers and cached
        tensors are still fully resident right after fit() returns, and both the
        checkpoint save and the export step can need a large amount of additional
        headroom -- this must run unconditionally on the common path, not only on the
        fallback-save path, otherwise export can OOM even when training and
        checkpointing both succeeded.
        """
        gc.collect()
        if torch.xpu.is_available():
            torch.xpu.empty_cache()

    @staticmethod
    def _save_final_checkpoint_if_missing(trainer: Trainer, final_checkpoint: Path) -> None:
        """Fall back to an explicit checkpoint save when none exists yet.

        ModelCheckpoint (monitor="val/loss") writes to this same path whenever a
        validation epoch completes, so on the common path it has already produced a
        usable checkpoint and an explicit save here would be pure duplicate work. This
        fallback only runs for the edge case where training ended before any
        validation ran (e.g. max_steps too small for one epoch), guaranteeing a
        checkpoint always exists.
        """
        trainer.save_checkpoint(final_checkpoint, weights_only=True)

    # Backends that trace the model graph (ONNX export, and OpenVINO which goes via
    # ONNX) need multiple copies of the model's parameters/activations resident at
    # once. For very large policies (multi-billion parameter VLA/VLM models) this can
    # exceed available memory even though the checkpoint save itself succeeded, which
    # is exactly what caused a post-100%-progress OOM kill on this iGPU/shared-RAM
    # host right after a training run had otherwise completed cleanly.
    _TRACED_EXPORT_BACKENDS = {"onnx", "openvino"}
    _BYTES_PER_PARAM_TRACE_ESTIMATE = 4 * 3  # fp32 param + activation/graph overhead
    _EXPORT_MEMORY_SAFETY_MARGIN = 1.2

    def _has_enough_memory_for_traced_export(self, policy: object) -> bool:
        try:
            import psutil

            param_count = sum(p.numel() for p in policy.parameters())
            estimated_bytes = param_count * self._BYTES_PER_PARAM_TRACE_ESTIMATE * self._EXPORT_MEMORY_SAFETY_MARGIN
            available_bytes = psutil.virtual_memory().available
        except Exception as e:  # best-effort guard, never block export on failure to check
            logger.warning("Could not estimate memory needs for traced export; proceeding anyway")
            logger.exception(e)
            return True

        if estimated_bytes > available_bytes:
            logger.warning(
                "Skipping traced export: policy has ~{}M params, estimated export memory need "
                "{:.1f}GB exceeds {:.1f}GB currently available",
                param_count // 1_000_000,
                estimated_bytes / 1e9,
                available_bytes / 1e9,
            )
            return False
        return True

    async def _export_policy(self, policy: object, path: Path, job: Job) -> None:
        if not isinstance(policy, ExportablePolicyMixin):
            logger.info("Skipping export: policy does not support export backends")
            return

        logger.info("Starting model export for trained policy")
        for backend in policy.get_supported_export_backends():
            backend_name = backend.value if hasattr(backend, "value") else str(backend)
            if backend_name.lower() in self._TRACED_EXPORT_BACKENDS and not self._has_enough_memory_for_traced_export(
                policy
            ):
                continue
            try:
                logger.info("Exporting model to {} format", backend_name)
                await JobService.update_job_status(
                    job_id=job.id,
                    status=JobStatus.RUNNING,
                    message=f"Exporting to {backend_name} format",
                )
                export_dir = path / "exports" / backend
                policy.export(export_dir, backend=backend)
                logger.info("Model export to {} completed", backend_name)
                self._release_training_memory()
            except Exception as e:
                logger.error("Failed exporting model to {} format", backend_name)
                logger.exception(e)

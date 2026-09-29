import time

import numpy as np

from runtime.control_loop_100hz import ActuatorLoop100Hz

from .fakes import FakeRobot


def test_send_action_does_not_block_or_write_synchronously() -> None:
    follower = FakeRobot(positions=[[0.0, 0.0]])
    loop = ActuatorLoop100Hz(follower, target_hz=100.0)

    loop.send_action(np.array([1.0, 2.0]), goal_time=0.0333)

    # send_action only records the waypoint; the bus write happens on the
    # background thread, not synchronously in the caller.
    assert follower.sent_actions == []


def test_background_thread_writes_to_wrapped_robot_after_connect() -> None:
    follower = FakeRobot(positions=[[0.0, 0.0]])
    loop = ActuatorLoop100Hz(follower, target_hz=100.0)

    loop.connect()
    try:
        loop.send_action(np.array([1.0, 2.0]), goal_time=0.0333)
        deadline = time.monotonic() + 1.0
        while not follower.sent_actions and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        loop.disconnect()

    assert follower.sent_actions
    np.testing.assert_array_equal(follower.sent_actions[-1], [1.0, 2.0])


def test_disconnect_stops_the_background_thread() -> None:
    follower = FakeRobot(positions=[[0.0, 0.0]])
    loop = ActuatorLoop100Hz(follower, target_hz=100.0)

    loop.connect()
    loop.disconnect()

    assert loop._thread is None
    assert not follower._connected


def test_get_observation_passes_through_to_wrapped_robot() -> None:
    follower = FakeRobot(positions=[[5.0, 6.0]])
    loop = ActuatorLoop100Hz(follower, target_hz=100.0)

    observation = loop.get_observation()

    np.testing.assert_array_equal(observation.joint_positions, [5.0, 6.0])


def test_joint_names_and_device_ids_pass_through() -> None:
    follower = FakeRobot(positions=[[0.0]], joint_names=["shoulder"])
    loop = ActuatorLoop100Hz(follower, target_hz=100.0)

    assert loop.joint_names == ["shoulder"]
    assert loop.device_ids == ()


def test_metrics_summary_reports_achieved_frequency() -> None:
    follower = FakeRobot(positions=[[0.0, 0.0]])
    loop = ActuatorLoop100Hz(follower, target_hz=100.0)

    loop.connect()
    try:
        loop.send_action(np.array([1.0, 2.0]), goal_time=0.0333)
        time.sleep(0.3)
    finally:
        loop.disconnect()

    metrics = loop.get_metrics_summary()

    assert metrics["total_ticks"] > 0
    assert 50.0 < metrics["mean_frequency_hz"] < 200.0

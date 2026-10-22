"""Shared fixtures: project paths, the default config, and a mock TRON2 upper-layer server.

No robot or camera is needed. The mock speaks the same WebSocket JSON protocol as the
robot, so drivers/robot/tron2_driver.py runs unchanged against it.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.camera_utils import load_config  # noqa: E402

# Pose of ee_link at the prepare posture, as shipped in config/default.yaml.
PREPARE_EE_POSE = [0.238598, 0.0, 0.229199, 0.521292, 0.521292, 0.47776, -0.47776]
# Pinhole intrinsics roughly matching a 1280x720 D405 colour stream.
K = np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="session")
def cfg() -> dict[str, Any]:
    return load_config(PROJECT_ROOT / "config" / "default.yaml")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockTron2:
    """Minimal stand-in for the TRON2 upper-layer WebSocket service.

    The arm follows the commanded targets exactly, so moves arrive. Behaviour can be
    switched per test through ``mode``:

      pose     "ok" | "silent" (no reply) | "nan" | "short" | "zeroquat" | "null"
      posture  reply result for request_wf_stand_kneel, or "silent"
      drop_on_stand  close the connection right after answering a WF stand
    """

    def __init__(self) -> None:
        self.port = _free_port()
        self.mode: dict[str, Any] = {"pose": "ok", "posture": "success", "drop_on_stand": False}
        self.q = [0.0] * 6
        self.pos = [0.0, 0.0, 0.0]
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.received: list[tuple[float, dict]] = []
        self.connections = 0
        self._clients: set = set()
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(5.0):
            raise RuntimeError("mock TRON2 server did not start")

    # server

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        import websockets

        self._stopped = asyncio.Event()
        async with websockets.serve(self._handler, "127.0.0.1", self.port):
            self._ready.set()
            await self._stopped.wait()

    async def _handler(self, ws) -> None:
        self.connections += 1
        self._clients.add(ws)
        try:
            await ws.send(json.dumps({"accid": "MOCK", "title": "notify_robot_info",
                                      "data": {"status": "idle"}}))
            async for raw in ws:
                msg = json.loads(raw)
                self.received.append((time.monotonic(), msg))
                if await self._reply(ws, msg):
                    return
        finally:
            self._clients.discard(ws)

    async def _reply(self, ws, msg: dict) -> bool:
        title, data = msg["title"], msg.get("data") or {}
        if title == "request_yg_servoj":
            self.q = list(data["q"])
        elif title == "request_yg_servop":
            self.pos, self.quat = list(data["pos"][:3]), list(data["pos"][3:])
        elif title == "request_get_yg_arm_state":
            await self._send(ws, "response_get_yg_arm_state", {"result": "success", "q": self.q})
        elif title == "request_get_yg_arm_pose":
            mode = self.mode["pose"]
            if mode == "silent":
                return False
            pos, quat = list(self.pos), list(self.quat)
            if mode == "nan":
                pos[0] = float("nan")
            elif mode == "short":
                pos = pos[:2]
            elif mode == "zeroquat":
                quat = [0.0, 0.0, 0.0, 0.0]
            elif mode == "null":
                pos = None
            await self._send(ws, "response_get_yg_arm_pose",
                             {"result": "success", "position": pos, "quat": quat})
        elif title == "request_wf_stand_kneel":
            if self.mode["posture"] == "silent":
                return False
            await self._send(ws, "response_wf_stand_kneel", {"result": self.mode["posture"]},
                             guid=msg.get("guid"))
            if data.get("action") == "stand" and self.mode["drop_on_stand"]:
                await asyncio.sleep(0.05)
                await ws.close()
                return True
        elif title == "request_adjust_head_pitch":
            await self._send(ws, "response_adjust_head_pitch", {"result": "success"},
                             guid=msg.get("guid"))
        return False

    @staticmethod
    async def _send(ws, title: str, data: dict, guid: Any = None) -> None:
        root = {"accid": "MOCK", "title": title, "data": data}
        if guid is not None:
            root["guid"] = guid
        await ws.send(json.dumps(root))

    # test helpers

    def push_raw(self, text: str) -> None:
        for ws in list(self._clients):
            asyncio.run_coroutine_threadsafe(ws.send(text), self._loop).result(2.0)

    def drop_clients(self) -> None:
        # Fire and forget: close() also waits for the client's TCP teardown, which
        # websocket-client may delay; the driver sees the close frame immediately.
        for ws in list(self._clients):
            asyncio.run_coroutine_threadsafe(ws.close(), self._loop)

    def messages(self, title: str, since: float = 0.0) -> list[dict]:
        return [m for t, m in list(self.received) if m["title"] == title and t >= since]

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._stopped.set)
        self._thread.join(5.0)


@pytest.fixture
def mock_tron2():
    server = MockTron2()
    yield server
    server.stop()


@pytest.fixture
def make_driver(mock_tron2):
    """Build Tron2Driver instances pointed at the mock; all are ended after the test."""
    from drivers.robot.tron2_driver import Tron2Driver

    made = []

    def _make(port: int | None = None, **tron2_overrides):
        tron2 = {
            "robot_ip": "127.0.0.1", "port": port or mock_tron2.port, "accid": None,
            "stream_hz": 100.0, "feedback_hz": 30.0, "connect_timeout_s": 3.0,
            "feedback_timeout_s": 0.6, "posture_timeout_s": 1.0, "arm_reacquire_timeout_s": 10.0,
            "speed_percent": 100, "joint_speed_rad_s": 3.0, "prepare_ee_pose": PREPARE_EE_POSE,
        }
        tron2.update(tron2_overrides)
        driver = Tron2Driver({"tron2": tron2, "gripper": {"tron2": {"settle_s": 0.0}}})
        made.append(driver)
        return driver

    yield _make
    for driver in made:
        try:
            driver.end(park=False)
        except Exception:
            pass

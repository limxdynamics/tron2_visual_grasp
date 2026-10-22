"""Web debug console: starts, serves status, and delivers button presses to the mission."""

from __future__ import annotations

import json
import socket
import time
import urllib.request

import numpy as np


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_status_command_and_hsv_setting(cfg):
    from utils.detection import load_detector
    from utils.web_debug import WebDebug

    port = _free_port()
    web = WebDebug(port=port)
    web.start()
    try:
        model, _ = load_detector(cfg)
        web.vision.bind(model, {})
        web.set_frame(np.zeros((120, 160, 3), np.uint8))
        time.sleep(0.3)
        base = f"http://127.0.0.1:{port}"
        assert urllib.request.urlopen(f"{base}/status", timeout=5).status == 200
        req = urllib.request.Request(f"{base}/cmd/stop", method="POST")
        assert urllib.request.urlopen(req, timeout=5).status == 204
        assert web.pop_command() == "stop"
        body = json.dumps({"kind": "hsv", "bound": "lower", "index": 0, "value": 25}).encode()
        req = urllib.request.Request(f"{base}/vision/settings", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        assert urllib.request.urlopen(req, timeout=5).status == 200
        assert model.get_hsv_bounds()[0][0] == 25
    finally:
        web.stop()

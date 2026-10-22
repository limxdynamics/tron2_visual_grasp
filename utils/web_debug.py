"""Minimal MJPEG + status web console for headless runs.

Serves a single page with one annotated stream per camera, a live status table
and the same commands the terminal accepts, so the scripts stay usable without
an X display. Missions that switch cameras mid-run declare one view per camera
so each panel keeps its own picture instead of overwriting a shared one.
"""

from __future__ import annotations

import html
import logging
import threading
import time
from collections import deque
from typing import Any, Optional

import cv2
import numpy as np

from utils.vision_tuning import VisionTuning

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{{background:#14161a;color:#d8dce3;font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;margin:0;padding:16px}}
 h1{{font-size:16px;margin:0 0 12px;color:#8ab4f8}}
 .row{{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}}
 img{{background:#000;border:1px solid #2a2f38;border-radius:6px;max-width:100%}}
 .panel{{flex:1;min-width:min(320px,100%)}}
 .cams{{display:flex;gap:12px;flex-wrap:wrap;max-width:100%}}
 figure{{margin:0;max-width:100%}}
 figcaption{{color:#8b93a1;padding-bottom:4px}}
 figcaption a{{color:#8b93a1;text-decoration:none}}
 figcaption a:hover{{color:#8ab4f8}}
 table{{border-collapse:collapse;width:100%}}
 td{{padding:3px 8px;border-bottom:1px solid #23272f;vertical-align:top}}
 td:first-child{{color:#8b93a1;white-space:nowrap;width:1%}}
 button{{background:#232833;color:#d8dce3;border:1px solid #39404d;border-radius:5px;
   padding:7px 14px;margin:4px 6px 4px 0;cursor:pointer;font:inherit}}
 button:hover{{background:#2d3441;border-color:#4a5464}}
 #log{{background:#0e1013;border:1px solid #23272f;border-radius:6px;padding:8px;
   height:220px;overflow:auto;white-space:pre-wrap;margin-top:10px}}
 .warn{{color:#e6a23c}} .err{{color:#f56c6c}} .ok{{color:#67c23a}}
</style></head><body>
<h1>{title}</h1>
<div class="row">
  <div class="cams">{views}</div>
  <div class="panel">
    <table id="st"></table>
    <div id="btns"></div>
    <iframe src="/vision" title="HSV and exposure" style="width:100%;height:980px;border:0"></iframe>
    <div id="log"></div>
  </div>
</div>
<script>
const btns = {buttons};
document.getElementById('btns').innerHTML =
  btns.map(b => `<button onclick="send('${{b[0]}}')">${{b[1]}}</button>`).join('');
function send(c){{ fetch('/cmd/' + c, {{method:'POST'}}); }}
// One request per frame: a stalled MJPEG connection would freeze until a manual reload.
function startView(name){{
  const img = document.getElementById('v_' + name);
  const next = () => {{ img.src = '/snapshot/' + name + '?t=' + Date.now(); }};
  img.onload = () => setTimeout(next, 40);
  img.onerror = () => setTimeout(next, 500);
  next();
}}
for (const v of {view_names}) startView(v);
let seen = 0;
async function poll(){{
  try{{
    const r = await fetch('/status?since=' + seen);
    const j = await r.json();
    document.getElementById('st').innerHTML =
      Object.entries(j.status).map(([k,v]) => `<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join('');
    const log = document.getElementById('log');
    for (const e of j.events){{
      const d = document.createElement('div');
      d.className = e.level; d.textContent = e.text; log.appendChild(d);
    }}
    if (j.events.length){{ log.scrollTop = log.scrollHeight; }}
    seen = j.next;
  }}catch(e){{}}
  setTimeout(poll, 400);
}}
poll();
</script></body></html>"""

_VIEW_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{label}</title>
<style>
 body{{background:#14161a;color:#d8dce3;font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;
   margin:0;padding:8px;display:flex;flex-direction:column;height:100vh;box-sizing:border-box}}
 h1{{font-size:14px;margin:0 0 6px;color:#8ab4f8}}
 img{{background:#000;border:1px solid #2a2f38;border-radius:6px;
   flex:1;min-height:0;width:100%;object-fit:contain}}
</style></head><body>
<h1>{label}</h1>
<img id="v">
<script>
const img = document.getElementById('v');
const next = () => {{ img.src = '/snapshot/{name}?t=' + Date.now(); }};
img.onload = () => setTimeout(next, 40);
img.onerror = () => setTimeout(next, 500);
next();
</script></body></html>"""


class WebDebug:
    """Serve the annotated frame, a status table and command buttons."""

    def __init__(self, port: int = 8888, title: str = "tron2_visual_grasp",
                 buttons: Optional[list[tuple[str, str]]] = None,
                 views: Optional[list[tuple[str, str]]] = None,
                 max_width: int = 640, quality: int = 70,
                 frames: bool = True) -> None:
        self.port = int(port)
        self.title = title
        # Encoding a frame per loop costs real milliseconds; the status rows are
        # cheap, so the two are switched separately.
        self.frames = bool(frames)
        self.buttons = buttons or []
        self.views = [(str(n), str(l)) for n, l in (views or [("main", "Camera")])]
        self.vision = VisionTuning()
        self._default_view = self.views[0][0]
        self._max_width = int(max_width)
        self._quality = int(quality)
        self._jpegs: dict[str, bytes] = {}
        self._seq: dict[str, int] = {}
        self._placeholders: dict[str, bytes] = {}
        self._status: dict[str, Any] = {}
        self._events: deque = deque(maxlen=300)
        self._event_seq = 0
        self._commands: deque = deque(maxlen=32)
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._app: Any = None
        self._server: Any = None

    # Producer side

    def set_frame(self, bgr, view: Optional[str] = None) -> None:
        name = view or self._default_view
        if not self.frames:
            return
        ok, buf = cv2.imencode(".jpg", self._downscale(bgr),
                              [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not ok:
            return
        with self._frame_ready:
            self._jpegs[name] = buf.tobytes()
            self._seq[name] = self._seq.get(name, 0) + 1
            self._frame_ready.notify_all()

    def _downscale(self, bgr):
        """Encode at display size: full-res JPEG stalls the caller's control loop."""
        height, width = bgr.shape[:2]
        if width <= self._max_width:
            return bgr
        scale = self._max_width / float(width)
        return cv2.resize(bgr, (self._max_width, max(1, int(round(height * scale)))),
                          interpolation=cv2.INTER_AREA)

    def _placeholder(self, view: str) -> bytes:
        cached = self._placeholders.get(view)
        if cached is not None:
            return cached
        label = dict(self.views).get(view, view)
        canvas = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(canvas, f"waiting for {label}", (24, 188),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 148, 161), 2)
        ok, buf = cv2.imencode(".jpg", canvas)
        self._placeholders[view] = buf.tobytes() if ok else b""
        return self._placeholders[view]

    def set_status(self, status: dict[str, Any]) -> None:
        with self._lock:
            self._status = dict(status)

    def add_event(self, text: str, level: str = "info") -> None:
        with self._lock:
            self._events.append({"text": f"[{time.strftime('%H:%M:%S')}] {text}",
                                 "level": level})
            self._event_seq += 1

    def pop_command(self) -> Optional[str]:
        with self._lock:
            return self._commands.popleft() if self._commands else None

    # Server

    def start(self) -> str:
        from flask import Flask, Response, jsonify
        from werkzeug.serving import make_server

        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app = Flask(__name__)
        self.vision.register(app)

        @app.route("/")
        def index():
            import json as _json
            views = "".join(
                f'<figure><figcaption>'
                f'<a href="/view/{name}" target="_blank">{label} &#8599;</a></figcaption>'
                f'<img id="v_{name}" width="640"></figure>'
                for name, label in self.views
            )
            return _PAGE.format(title=self.title, views=views,
                                view_names=_json.dumps([n for n, _ in self.views]),
                                buttons=_json.dumps(self.buttons))

        @app.route("/view/<view>")
        def single_view(view: str):
            # The name is echoed into the page, so only configured views are served.
            labels = dict(self.views)
            if view not in labels:
                return ("", 404)
            return _VIEW_PAGE.format(label=html.escape(labels[view]), name=view)

        @app.route("/snapshot/<view>")
        def snapshot(view: str):
            with self._lock:
                jpeg = self._jpegs.get(view)
            if jpeg is None:
                jpeg = self._placeholder(view)
            return Response(jpeg, mimetype="image/jpeg",
                            headers={"Cache-Control": "no-store"})

        @app.route("/stream")
        @app.route("/stream/<view>")
        def stream(view: Optional[str] = None):
            name = view or self._default_view
            def frames():
                last_seq = -1
                while True:
                    with self._frame_ready:
                        if self._seq.get(name, 0) == last_seq:
                            # Wake on the next frame; the timeout only keeps the
                            # connection alive while a camera is still closed.
                            self._frame_ready.wait(1.0)
                        last_seq = self._seq.get(name, 0)
                        jpeg = self._jpegs.get(name)
                    if jpeg is None:
                        jpeg = self._placeholder(name)
                    yield (b"--f\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() +
                           b"\r\n\r\n" + jpeg + b"\r\n")
            return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=f",
                            headers={"Cache-Control": "no-store"})

        @app.route("/status")
        def status():
            from flask import request
            since = request.args.get("since", type=int) or 0
            with self._lock:
                events = list(self._events)
                base = self._event_seq - len(events)
                return jsonify({
                    "status": self._status,
                    "events": events[max(0, since - base):],
                    "next": self._event_seq,
                })

        @app.route("/cmd/<name>", methods=["POST"])
        def cmd(name: str):
            with self._lock:
                self._commands.append(name)
            return ("", 204)

        self._app = app
        self._server = make_server("0.0.0.0", self.port, app, threaded=True)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://<host>:{self.port}/"

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None

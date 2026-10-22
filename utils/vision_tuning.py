"""Live HSV and exposure controls served inside the web debug console."""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import cv2
import numpy as np


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HSV / Exposure</title>
<style>
*{box-sizing:border-box}body{margin:0;padding:12px;color:#d8dce3;background:#14161a;
font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;letter-spacing:0}
h2{font-size:14px;margin:0 0 8px}fieldset{border:0;padding:0;margin:0 0 16px;min-width:0}
legend{padding:0;margin-bottom:6px;color:#8ab4f8}
.control{display:grid;grid-template-columns:64px minmax(60px,1fr) 58px;gap:8px;align-items:center;margin:5px 0}
input[type=range]{width:100%;min-width:0;accent-color:#70bdb1}output{text-align:right;font-size:12px}
.automatic{display:block;margin-bottom:8px}#error{color:#f56c6c;overflow-wrap:anywhere}
.muted{color:#8b93a1}input:focus-visible{outline:2px solid #8ab4f8}
#preview{display:block;width:100%;margin-bottom:6px;background:#0c0d10}
</style></head><body>
<h2>HSV / Exposure</h2>
<div id="previewbox" hidden><img id="preview" alt="HSV preview">
<div class="muted" id="previewnote">Top: detections. Bottom: HSV mask (yellow = after morphology). Dragging previews without saving.</div></div>
<div id="controls"></div><div id="error" role="status"></div>
<script>
const root = document.getElementById('controls');
const error = document.getElementById('error');
const preview = document.getElementById('preview');
const previewBox = document.getElementById('previewbox');
const previewNote = document.getElementById('previewnote');
let signature = '', pending = 0, requests = Promise.resolve();
let previewBusy = false, previewDirty = true;
function hsvQuery() {
  const value = id => { const input = document.getElementById(id); return input ? input.value : null; };
  const lower = [0, 1, 2].map(i => value('lower' + i)), upper = [0, 1, 2].map(i => value('upper' + i));
  if (lower.includes(null) || upper.includes(null)) return '';
  return 'lower=' + lower.join(',') + '&upper=' + upper.join(',') + '&';
}
function refreshPreview() {
  if (previewBusy || !previewDirty || previewBox.hidden) return;
  previewBusy = true; previewDirty = false;
  preview.src = '/vision/preview.jpg?' + hsvQuery() + 't=' + Date.now();
}
preview.onload = () => {
  previewBusy = false; preview.hidden = false;
  previewNote.textContent = 'Top: detections. Bottom: HSV mask (yellow = after morphology). Dragging previews without saving.';
  refreshPreview();
};
preview.onerror = () => {
  previewBusy = false; preview.hidden = true;
  previewNote.textContent = 'No frame yet: preview appears once the detector has run.';
};
function slider(parent, id, label, min, max, step, value, payload) {
  const row = document.createElement('label'); row.className = 'control';
  const name = document.createElement('span'); name.textContent = label;
  const input = document.createElement('input'); input.type = 'range'; input.id = id;
  Object.assign(input, {min, max, step, value});
  const output = document.createElement('output'); output.id = id + '_value';
  output.htmlFor = id; output.textContent = value;
  input.oninput = () => {
    output.textContent = input.value;
    if (payload.kind === 'hsv') { previewDirty = true; refreshPreview(); }
  };
  input.onchange = () => save({...payload, value:Number(input.value)});
  row.append(name, input, output); parent.appendChild(row);
}
function section(label) {
  const field = document.createElement('fieldset');
  const legend = document.createElement('legend'); legend.textContent = label;
  field.appendChild(legend); root.appendChild(field); return field;
}
function update(id, value, min, max) {
  const input = document.getElementById(id);
  if (!input || input === document.activeElement) return;
  input.value = value;
  if (min !== undefined) { input.min = min; input.max = max; }
  document.getElementById(id + '_value').textContent = value;
}
function render(state) {
  const next = JSON.stringify([!!state.hsv, state.cameras.map(cam => [cam.name, !!cam.controls, cam.error])]);
  if (signature !== next) {
    signature = next; root.replaceChildren();
    previewBox.hidden = !state.hsv;
    if (state.hsv) {
      const field = section('HSV');
      for (const bound of ['lower', 'upper']) {
        ['H', 'S', 'V'].forEach((channel, index) => {
          slider(field, bound + index, channel + (bound === 'lower' ? ' min' : ' max'),
            0, index === 0 ? 179 : 255, 1, state.hsv[bound][index], {kind:'hsv', bound, index});
        });
      }
    }
    for (const cam of state.cameras) {
      const field = section(cam.name + ' / Exposure (us)');
      if (!cam.controls) {
        const note = document.createElement('div'); note.className = 'muted';
        note.textContent = cam.error || 'Exposure unavailable'; field.appendChild(note); continue;
      }
      const label = document.createElement('label'); label.className = 'automatic';
      const toggle = document.createElement('input'); toggle.type = 'checkbox';
      toggle.id = 'auto_' + cam.name;
      toggle.onchange = () => save({kind:'auto', camera:cam.name, value:toggle.checked});
      label.append(toggle, ' Auto exposure'); field.appendChild(label);
      const values = cam.controls;
      slider(field, 'exp_' + cam.name, 'us', values.min, values.max, values.step,
        values.value, {kind:'exposure', camera:cam.name});
    }
    if (!state.hsv && !state.cameras.length) root.textContent = 'Waiting for camera / detector';
  }
  if (state.hsv) {
    for (let index = 0; index < 3; index++) {
      update('lower' + index, state.hsv.lower[index], 0, state.hsv.upper[index]);
      update('upper' + index, state.hsv.upper[index], state.hsv.lower[index], index === 0 ? 179 : 255);
    }
  }
  for (const cam of state.cameras) {
    if (!cam.controls) continue;
    document.getElementById('auto_' + cam.name).checked = cam.controls.auto;
    update('exp_' + cam.name, cam.controls.value);
  }
}
function save(payload) {
  pending++;
  requests = requests.then(async () => {
    try {
      const response = await fetch('/vision/settings', {method:'POST',
        headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      const state = await response.json();
      if (!response.ok) throw new Error(state.error || response.statusText);
      error.textContent = '';
      if (pending === 1) render(state);
    } catch (failure) { error.textContent = failure.message; }
    finally { pending--; }
  });
  return requests;
}
async function poll() {
  try {
    if (!pending) {
      const response = await fetch('/vision/state', {cache:'no-store'});
      if (!response.ok) throw new Error(response.statusText);
      const state = await response.json();
      if (!pending) render(state);
    }
  } catch (failure) { error.textContent = failure.message; }
  previewDirty = true; refreshPreview();
  setTimeout(poll, 1000);
}
poll();
</script></body></html>"""


class VisionTuning:
    def __init__(self) -> None:
        self._model: Any = None
        self._cameras: dict[str, Any] = {}
        self._lock = threading.RLock()

    def bind(self, model: Any, cameras: dict[str, Any]) -> None:
        with self._lock:
            self._model = model
            self._cameras = dict(cameras)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            hsv = None
            getter = getattr(self._model, 'get_hsv_bounds', None)
            if getter is not None:
                lower, upper = getter()
                hsv = {"lower": lower, "upper": upper}
            cameras = []
            for name, camera in self._cameras.items():
                entry = {"name": name, "controls": None}
                try:
                    entry["controls"] = camera.get_exposure_controls()
                except (RuntimeError, NotImplementedError) as exc:
                    entry["error"] = str(exc)
                cameras.append(entry)
            return {"hsv": hsv, "cameras": cameras}

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            kind = payload.get("kind")
            if kind == "hsv":
                getter = getattr(self._model, 'get_hsv_bounds', None)
                if getter is None:
                    raise ValueError("HSV controls are unavailable for this detector")
                bound, index = payload.get("bound"), payload.get("index")
                if bound not in ("lower", "upper") or type(index) is not int or index not in range(3):
                    raise ValueError("Invalid HSV channel")
                lower, upper = getter()
                (lower if bound == "lower" else upper)[index] = payload["value"]
                self._model.set_hsv_bounds(lower, upper)
            elif kind in ("auto", "exposure"):
                camera = self._cameras.get(payload.get("camera"))
                if camera is None:
                    raise ValueError("Unknown camera")
                if kind == "auto":
                    if type(payload.get("value")) is not bool:
                        raise ValueError("Auto exposure requires a boolean")
                    camera.set_auto_exposure(payload["value"])
                else:
                    value = float(payload["value"])
                    controls = camera.get_exposure_controls()
                    if (controls is None or not math.isfinite(value)
                            or not controls["min"] <= value <= controls["max"]):
                        raise ValueError("Exposure is outside the camera range")
                    steps = (value - controls["min"]) / controls["step"]
                    if not math.isclose(steps, round(steps), abs_tol=1e-6):
                        raise ValueError("Exposure does not match the camera step")
                    camera.set_exposure_us(value)
            else:
                raise ValueError("Unknown vision setting")
            return self.snapshot()

    def preview_jpeg(self, lower: Any = None, upper: Any = None,
                     max_width: int = 480) -> bytes | None:
        """Render the detector's latest frame with the given (or current) HSV bounds."""
        with self._lock:
            model = self._model
        getter = getattr(model, 'last_frame', None)
        latest = getter() if getter is not None else None
        if latest is None:
            return None
        frame, stamp, conf = latest
        if lower is None or upper is None:
            lower, upper = model.get_hsv_bounds()
        lower, upper = model.validate_hsv_bounds(lower, upper)
        result = model.predict(frame, hsv=(lower, upper), conf=conf)[0]
        colour, cleaned = model.hsv_masks(frame, lower, upper)

        top = frame.copy()
        boxes = result.obb.xyxy.astype(int)
        for (x1, y1, x2, y2), score in zip(boxes, result.obb.conf):
            cv2.rectangle(top, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(top, f"{score:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        bottom = (frame * 0.2).astype(np.uint8)
        bottom[colour > 0] = frame[colour > 0]
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bottom, contours, -1, (0, 255, 255), 1)
        coverage = 100.0 * np.count_nonzero(colour) / colour.size
        label = (f"det {len(boxes)}  mask {coverage:.1f}%  "
                 f"frame {time.time() - stamp:.0f}s ago")
        cv2.putText(top, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(top, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        image = np.vstack([top, bottom])
        height, width = image.shape[:2]
        if width > max_width:
            image = cv2.resize(image, (max_width, round(height * max_width / width)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buf.tobytes() if ok else None

    def register(self, app: Any) -> None:
        from flask import Response, jsonify, request

        @app.get('/vision')
        def vision_page():
            return _PAGE

        @app.get('/vision/state')
        def vision_state():
            return jsonify(self.snapshot())

        @app.get('/vision/preview.jpg')
        def vision_preview():
            def bounds(name: str) -> list[int] | None:
                text = request.args.get(name)
                return None if text is None else [int(part) for part in text.split(',')]
            try:
                jpeg = self.preview_jpeg(bounds('lower'), bounds('upper'))
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
            if jpeg is None:
                return jsonify(error="No frame yet"), 404
            return Response(jpeg, mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})

        @app.post('/vision/settings')
        def vision_settings():
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return jsonify(error="Expected a JSON object"), 400
            try:
                return jsonify(self.apply(payload))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                return jsonify(error=str(exc)), 400
            except (RuntimeError, NotImplementedError) as exc:
                return jsonify(error=str(exc)), 409
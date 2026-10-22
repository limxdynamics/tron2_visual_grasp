"""Fetch the TRON2 3D model used by the demo's 3D view.

The model is not shipped in this repository. On first use, the WFYG_TRON2A
variant (MuJoCo XML + STL meshes, about 50 MB) is fetched with a sparse git
checkout of limxdynamics/tron2-robot-description at a pinned commit, together
with its LICENSE and NOTICE (Apache-2.0), into a per-user cache directory.

Set TVG_ROBOT_MODEL to an existing robot.xml to use a local copy instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

MODEL_REPO = "https://github.com/limxdynamics/tron2-robot-description.git"
MODEL_COMMIT = "f547f5bc949f2a4c98e076e61cf6d3ca73d179a0"
VARIANT = "tron2a/WFYG_TRON2A"
_SPARSE_PATHS = [f"{VARIANT}/xml/", f"{VARIANT}/meshes/", "LICENSE", "NOTICE"]


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tron2_visual_grasp" / f"tron2-robot-description-{MODEL_COMMIT[:8]}"


def model_xml(cache_dir: Optional[Path] = None) -> Path:
    override = os.environ.get("TVG_ROBOT_MODEL")
    if override:
        return Path(override).expanduser()
    return (cache_dir or default_cache_dir()) / VARIANT / "xml" / "robot.xml"


def ensure_model(cache_dir: Optional[Path] = None, quiet: bool = False) -> Optional[Path]:
    """Return the path of robot.xml, downloading the model first if needed; None on failure."""
    xml = model_xml(cache_dir)
    if xml.exists():
        return xml
    if os.environ.get("TVG_ROBOT_MODEL"):
        print(f"[Sim3D] TVG_ROBOT_MODEL={xml} does not exist")
        return None
    if shutil.which("git") is None:
        print("[Sim3D] git not found; cannot fetch the TRON2 3D model")
        return None
    root = cache_dir or default_cache_dir()
    tmp = root.with_name(root.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if not quiet:
        print(f"[Sim3D] Downloading the TRON2 3D model (~50 MB, Apache-2.0) from {MODEL_REPO} "
              f"@ {MODEL_COMMIT[:8]} into {root} ...")

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp, check=True, capture_output=True, text=True, timeout=600)

    try:
        git("init", "-q")
        git("remote", "add", "origin", MODEL_REPO)
        git("sparse-checkout", "set", "--no-cone", *_SPARSE_PATHS)
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", MODEL_COMMIT)
        git("checkout", "-q", "FETCH_HEAD")
    except (subprocess.SubprocessError, OSError) as exc:
        detail = getattr(exc, "stderr", "") or exc
        print(f"[Sim3D] Model download failed: {str(detail).strip()[:300]}")
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    shutil.rmtree(tmp / ".git", ignore_errors=True)
    shutil.rmtree(root, ignore_errors=True)
    tmp.rename(root)
    return xml if xml.exists() else None

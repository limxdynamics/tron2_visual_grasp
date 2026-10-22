#!/usr/bin/env bash
# Launcher for the TRON2 visual-grasp scripts: sets up the Python environment,
# then dispatches to a script under scripts/.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The mission runs in the conda env 'tron2_visual_grasp' (deps in environment.yml),
# never the conda base env. Set TVG_PYTHON to use another interpreter instead.
ENV_NAME="tron2_visual_grasp"

usage() {
    cat <<'EOF'
Usage: ./run.sh <command> [options...]

Commands:
  grasp      Full mission: approach, squat, then D405 grasp
  dry-run    Perception only, never drives the robot
  demo       Hardware-free demo: the full mission on a simulated TRON2 and camera
  calib      Eye-in-hand calibration capture
  verify     Eye-in-hand calibration check
  detect     Ball detection preview, never drives the robot
  home       Move the arm to a resting pose, then exit
  pose       Print the arm joints and TCP pose, read-only
  shell      Run any other file with the project environment

All extra options are passed straight through to the underlying script.

Examples:
  ./run.sh demo              # then open http://localhost:8888/
  ./run.sh dry-run
  ./run.sh grasp --grasp-timeout 30
  ./run.sh calib
  ./run.sh home --pose ready
  ./run.sh pose --watch
EOF
}

if [[ $# -eq 0 || "${1}" == "-h" || "${1}" == "--help" ]]; then
    usage
    exit 0
fi

command="$1"
shift

declare -a args=()
case "${command}" in
    grasp)    script="scripts/squat_grasp.py" ;;
    dry-run)  script="scripts/squat_grasp.py"; args+=(--dry-run) ;;
    demo)     script="scripts/squat_grasp.py"; args+=(--sim) ;;
    calib)    script="scripts/collect_handeye_eih.py" ;;
    verify)   script="scripts/verify_handeye.py" ;;
    detect)   script="scripts/object_detection.py" ;;
    home)     script="scripts/home_arm.py" ;;
    pose)     script="scripts/arm_state.py" ;;
    shell)
        if [[ $# -eq 0 ]]; then
            echo "run.sh: 'shell' needs a script path" >&2
            exit 2
        fi
        script="$1"
        shift
        ;;
    *)
        echo "run.sh: unknown command '${command}'" >&2
        echo >&2
        usage >&2
        exit 2
        ;;
esac

if [[ ! -f "${PROJECT_ROOT}/${script}" ]]; then
    echo "run.sh: ${script} not found" >&2
    exit 1
fi

# Resolve the interpreter. Locate conda through `conda info --base` so Miniconda,
# Anaconda, Miniforge or a custom install prefix all work. Never fall back to a
# system Python: it would run without the project dependencies.
CONDA_BASE=""
ENV_ACTIVATE=0
if [[ -n "${TVG_PYTHON:-}" ]]; then
    PYTHON="${TVG_PYTHON}"
    if [[ ! -x "${PYTHON}" ]]; then
        echo "run.sh: TVG_PYTHON=${PYTHON} is not an executable interpreter" >&2
        exit 1
    fi
else
    CONDA_CMD="${CONDA_EXE:-conda}"
    if command -v "${CONDA_CMD}" >/dev/null 2>&1; then
        CONDA_BASE="$("${CONDA_CMD}" info --base 2>/dev/null || true)"
    fi
    if [[ -z "${CONDA_BASE}" ]]; then
        echo "run.sh: conda not found. Install conda, or set TVG_PYTHON=/path/to/python" >&2
        exit 1
    fi
    PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
    if [[ ! -x "${PYTHON}" ]]; then
        echo "run.sh: conda env '${ENV_NAME}' not found under ${CONDA_BASE}/envs." >&2
        echo "run.sh: create it first:  conda env create -f environment.yml" >&2
        echo "run.sh: or set TVG_PYTHON=/path/to/python to use another interpreter" >&2
        exit 1
    fi
    ENV_ACTIVATE=1
fi

# Running the env's interpreter directly skips conda's activate.d hooks; activate properly.
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if (( ENV_ACTIVATE )) && [[ -f "${CONDA_SH}" ]]; then
    set +eu
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${ENV_NAME}"
    set -eu
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
if [[ "${command}" != "dry-run" && "${command}" != "demo" && "${command}" != "pose" && "${command}" != "detect" ]] && [[ ! " $* " == *" --dry-run "* ]]; then
    echo "[run.sh] ${script} may drive the robot. Keep the emergency stop in hand."
fi
exec "${PYTHON}" "${script}" "${args[@]}" "$@"

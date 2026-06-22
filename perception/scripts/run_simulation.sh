#!/bin/bash
#SBATCH --job-name=3dv_sim
#SBATCH --output=/work/courses/3dv/team15/project-3dv/logs/sim_%j.log
#SBATCH --error=/work/courses/3dv/team15/project-3dv/logs/sim_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Usage:
#   sbatch scripts/run_simulation.sh [--scene N] [--planner rrtstar|curobo|none]
# All extra arguments are forwarded to run_simulation.py.

set -euo pipefail

REPO=/work/courses/3dv/team15/project-3dv
PERCEPTION=/work/courses/3dv/team15/curobo-sq/perception
SUPERDEC=/work/courses/3dv/team15/superdec
CUROBO_SRC=/work/courses/3dv/team15/curobo-sq/curobo/src
CHECKPOINT=/work/courses/3dv/team15/checkpoints/superdec_tabletop/superdec_tabletop_finetune_v2/epoch_300.pt

# ── Environment ────────────────────────────────────────────────────────────────
echo "=== Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a') ==="
echo "=== Job ID: $SLURM_JOB_ID  ==="

# Activate the project venv / conda env if present
if [ -f "$REPO/.venv/bin/activate" ]; then
    source "$REPO/.venv/bin/activate"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Using existing conda env: $CONDA_DEFAULT_ENV"
fi

# Make perception module and superdec importable
export PYTHONPATH="${PERCEPTION}:${SUPERDEC}:${CUROBO_SRC}:${PYTHONPATH:-}"

# ── Create output directory ────────────────────────────────────────────────────
mkdir -p "${REPO}/logs" "${REPO}/outputs"

# ── Default arguments (overridden by command-line) ────────────────────────────
SCENE=${SCENE:-1}
PLANNER=${PLANNER:-rrtstar}
N_QUERIES=${N_QUERIES:-5}

# ── Parse simple flags passed via sbatch --export or command line ─────────────
EXTRA_ARGS=()
for arg in "$@"; do
    EXTRA_ARGS+=("$arg")
done

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "Running: python3 ${PERCEPTION}/scripts/run_simulation.py"
echo "         --scene ${SCENE} --planner ${PLANNER} --n-queries ${N_QUERIES}"
echo "         --checkpoint ${CHECKPOINT}"
echo "         ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
echo ""

python3 "${PERCEPTION}/scripts/run_simulation.py" \
    --scene "${SCENE}" \
    --planner "${PLANNER}" \
    --n-queries "${N_QUERIES}" \
    --checkpoint "${CHECKPOINT}" \
    --out-dir "${REPO}/outputs" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Done (exit $?) ==="

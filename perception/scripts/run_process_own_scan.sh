#!/bin/bash
#SBATCH --job-name=own-scan
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team15/project-3dv/logs/own_scan_%j.out

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/courses/3dv/team15/superdec/.venv/bin/activate

export PYTHONPATH="\
/work/courses/3dv/team15/curobo-sq/perception:\
/work/courses/3dv/team15/curobo-sq/superdec:\
${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR=/work/courses/3dv/team15/.torch_extensions

cd /work/courses/3dv/team15/project-3dv

python3 /work/courses/3dv/team15/curobo-sq/perception/scripts/process_own_scan.py \
  --r3d /work/courses/3dv/team15/data/own_scan/2026-03-30--20-51-15.r3d \
  --checkpoint /work/courses/3dv/team15/checkpoints/superdec_tabletop/superdec_tabletop_finetune_v3/epoch_300.pt \
  --output outputs/own_scan/ \
  --frame-stride 15 \
  --conf-threshold 1 \
  --voxel-size 0.005 \
  --verbose \
  "$@"

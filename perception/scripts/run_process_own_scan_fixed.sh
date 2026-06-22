#!/bin/bash
#SBATCH --job-name=own-scan-fixed
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/work/courses/3dv/team15/project-3dv/logs/own_scan_fixed_%j.out

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/courses/3dv/team15/superdec/.venv/bin/activate

export PYTHONPATH="\
/work/courses/3dv/team15/curobo-sq/perception:\
/work/courses/3dv/team15/curobo-sq/superdec:\
${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR=/work/courses/3dv/team15/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team15/project-3dv

python3 /work/courses/3dv/team15/curobo-sq/perception/scripts/process_own_scan.py \
  --input-cloud outputs/own_scan_fixed/fused_cloud_rotated.npz \
  --checkpoint /work/courses/3dv/team15/checkpoints/superdec_tabletop/superdec_tabletop_finetune_v3/epoch_300.pt \
  --exist-threshold 0.3 \
  --verbose \
  --output outputs/own_scan_rotated/ \
  2>&1 | tee outputs/own_scan_rotated/run.log

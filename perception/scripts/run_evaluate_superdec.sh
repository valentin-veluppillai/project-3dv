#!/bin/bash
#SBATCH --job-name=sq-evaluate
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/work/courses/3dv/team15/project-3dv/logs/evaluate_%j.out

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/courses/3dv/team15/superdec/.venv/bin/activate

export PYTHONPATH="\
/work/courses/3dv/team15/curobo-sq/perception:\
/work/courses/3dv/team15/curobo-sq/superdec:\
${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR=/work/courses/3dv/team15/.torch_extensions

cd /work/courses/3dv/team15/project-3dv

python3 /work/courses/3dv/team15/curobo-sq/perception/scripts/evaluate_superdec.py \
  --checkpoint /work/courses/3dv/team15/checkpoints/superdec_tabletop/superdec_tabletop_finetune_v3/epoch_300.pt \
  --data-rgbd data/rgbd-scenes-v2/pc \
  --data-shapenet /work/courses/3dv/team15/superdec/data/ShapeNet \
  --scenes 1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
  --categories bowl,bottle,mug,knife,laptop \
  --n-shapenet 3 \
  --exist-threshold 0.3 \
  --output outputs/evaluate_superdec/ \
  --verbose \
  "$@"

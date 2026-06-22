#!/bin/bash
#SBATCH --job-name=sq-perception-test
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/work/courses/3dv/team15/logs/test_split_%j.out

. /etc/profile.d/modules.sh
module add cuda/12.9

source /work/courses/3dv/team15/superdec/.venv/bin/activate
export PYTHONPATH=/work/courses/3dv/team15/superdec:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/work/courses/3dv/team15/.torch_extensions

cd /work/courses/3dv/team15/project-3dv

python3 /work/courses/3dv/team15/curobo-sq/perception/scripts/test_on_superdec_split.py \
  --n-samples 5 \
  --output-dir outputs/test_split_gpu/ \
  "$@"

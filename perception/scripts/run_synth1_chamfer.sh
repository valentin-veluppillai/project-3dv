#!/bin/bash
#SBATCH --job-name=sq-on-frame-c
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/work/courses/3dv/team15/project-3dv/logs/on_frame_chamfer_%j.out

. /etc/profile.d/modules.sh
module add cuda/12.9
source /work/courses/3dv/team15/superdec/.venv/bin/activate

export PYTHONPATH="/work/courses/3dv/team15/curobo-sq/perception:/work/courses/3dv/team15/superdec_concave:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team15/project-3dv

python3 /work/courses/3dv/team15/curobo-sq/perception/scripts/run_on_frame.py \
  --to_scene_npz data/synth/scene_1.npz \
  --superdec_dir /work/courses/3dv/team15/superdec_concave \
  --ckpt_dir     /work/courses/3dv/team15/superdec_concave/checkpoints/expocc_tt_chamfer \
  --out_dir      data/fit_out_synth_1_chamfer \
  --exclude_ids  0 \
  --semantic_min 41

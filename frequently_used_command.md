cd /home/ps2338/simtoolreal


### demo recording for slurm ####
sbatch \
  --job-name=checkpoint-video \
  --partition=fang --account=fang \
  --gres=gpu:nvidia_rtx_a6000:1 \
  --cpus-per-task=8 --mem=32G --time=00:45:00 \
  --output="logs/checkpoint-video-%j.log" \
  record_demos.sh



sbatch \
  --job-name=video-378081 \
  --partition=fang --account=fang \
  --gres=gpu:nvidia_rtx_a6000:1 \
  --cpus-per-task=8 --mem=32G --time=00:45:00 \
  --output="logs/video-378081-%j.log" \
  record_demos.sh
xvfb-run -a python -m sample_factory.launcher.run \
--run=benchmark.slurm.train_mujoco --backend=slurm \
--slurm_workdir=slurm_output \
--slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 \
 --experiment_suffix=slurm --slurm_sbatch_template=/home/zhehui/reward_decrease/dir_cleanrl/sbatch_timeout.sh \
 --slurm_print_only=False
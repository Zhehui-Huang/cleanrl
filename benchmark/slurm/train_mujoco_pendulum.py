from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from utils.utils import timeStamped

MUJOCO_BASELINE = (
    'python -m cleanrl.ppo_continuous_action '
    '--total-timesteps=100000000 --num-envs=64 --anneal-lr=True --clip-vloss=False --ent-coef=0 '
    '--learning-rate=0.00295 --max-grad-norm=3.5 --num-minibatches=4 --num-steps=64 --update-epochs=2 --vf-coef=1.3 '
    '--track --capture-video --wandb-project-name mujoco-cleanrl '
    '--wandb-entity=multi-drones --wandb-project-name=zh-reward-decrease'
)


_params = ParamGrid([
    ("seed", [0]),
    ("env-id", ["InvertedPendulum-v4", "InvertedDoublePendulum-v4"]),
])

_experiment = Experiment(
    'cleanrl_pendulum',
    MUJOCO_BASELINE,
    _params.generate_params(randomize=False),
)

run_name = timeStamped("cleanrl_baseline", fmt="{fname}_%Y%m%d_%H%M")

RUN_DESCRIPTION = RunDescription(run_name, experiments=[_experiment])

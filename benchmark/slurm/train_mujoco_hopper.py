from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from utils.utils import timeStamped

MUJOCO_BASELINE = (
    'python -m cleanrl.ppo_continuous_action '
    '--total-timesteps=500000000 --num-envs=64 --anneal-lr=True --clip-vloss=False --ent-coef=0 '
    '--learning-rate=0.00295 --max-grad-norm=3.5 --num-minibatches=4 --num-steps=64 --update-epochs=2 --vf-coef=1.3 '
    '--track --capture-video --wandb-project-name mujoco-cleanrl '
    '--wandb-entity=multi-drones --wandb-project-name=zh-reward-decrease'
)


_params = ParamGrid([
    ("seed", [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
    ("env-id", ["Hopper-v2"]),
])

_experiment = Experiment(
    'cleanrl_hopper',
    MUJOCO_BASELINE,
    _params.generate_params(randomize=False),
)

run_name = timeStamped("cleanrl_baseline", fmt="{fname}_%Y%m%d_%H%M")

RUN_DESCRIPTION = RunDescription(run_name, experiments=[_experiment])

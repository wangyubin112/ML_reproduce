# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin

import argparse
import os
import time

import gym
import numpy as np
from stable_baselines.common import set_global_seeds

from utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

from config import MIN_THROTTLE, MAX_THROTTLE, FRAME_SKIP,\
    MAX_CTE_ERROR, SIM_PARAMS, N_COMMAND_HISTORY, SIZE_Z, MAX_STEERING_DIFF,\
    IMAGE_WIDTH, IMAGE_HEIGHT, ID_ENV # wyb

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs')
parser.add_argument('--algo', help='RL Algorithm', default='sac',
                    type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=5000,
                    type=int)
parser.add_argument('--exp-id', help='Experiment ID (-1: no exp folder, 0: latest)', default=0,
                    type=int)
parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                    type=int)
parser.add_argument('--no-render', action='store_true', default=False,
                    help='Do not render the environment (useful for tests)')
parser.add_argument('--deterministic', action='store_true', default=False,
                    help='Use deterministic actions')
parser.add_argument('--norm-reward', action='store_true', default=False,
                    help='Normalize reward if applicable (trained with VecNormalize)')
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
# parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='vae/model/carla/vae-32_best.pkl') # wyb
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='vae/model/carla/vae-{}_best.pkl'.format(SIZE_Z)) # wyb
parser.add_argument('-best', '--best-model', action='store_true', default=True,
                    help='Use best saved model of that experiment (if it exists)')
parser.add_argument('--level', help='Level index', type=int, default=0)
args = parser.parse_args()

algo = args.algo
folder = args.folder

if args.exp_id == 0:
    args.exp_id = get_latest_run_id(os.path.join(folder, algo), ID_ENV)
    print('Loading latest experiment, id={}'.format(args.exp_id))

# Sanity checks
if args.exp_id > 0:
    log_path = os.path.join(folder, algo, '{}_{}'.format(ID_ENV, args.exp_id))
else:
    log_path = os.path.join(folder, algo)

best_path = ''
if args.best_model:
    best_path = '_best'

# wyb
save_path = os.path.join(log_path, "{}_{}".format(ID_ENV+'_'+str(SIZE_Z), get_latest_run_id(log_path, ID_ENV) + 1)) # wyb
model_path = os.path.join(save_path, "{}{}.zip".format(ID_ENV, best_path)) # wyb

# model_path = os.path.join(log_path, "{}{}.pkl".format(ID_ENV, best_path))

assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, ID_ENV, model_path)

set_global_seeds(args.seed)

stats_path = os.path.join(save_path, ID_ENV) # wyb
# stats_path = os.path.join(log_path, ID_ENV)
hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward)
hyperparams['vae_path'] = args.vae_path

log_dir = args.reward_log if args.reward_log != '' else None

env = create_test_env(args.level, stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                      hyperparams=hyperparams)

model = ALGOS[algo].load(model_path)

obs = env.reset()

# Force deterministic for SAC and DDPG
deterministic = args.deterministic or algo in ['ddpg', 'sac']
if args.verbose >= 1:
    print("Deterministic actions: {}".format(deterministic))

running_reward = 0.0
ep_len = 0
for _ in range(args.n_timesteps):
    action, _ = model.predict(obs, deterministic=deterministic)
    # Clip Action to avoid out of bound errors
    if isinstance(env.action_space, gym.spaces.Box):
        action = np.clip(action, env.action_space.low, env.action_space.high)
    obs, reward, done, infos = env.step(action)
    if not args.no_render:
        env.render('human')
    running_reward += reward[0]
    ep_len += 1

    if done and args.verbose >= 1:
        # NOTE: for env using VecNormalize, the mean reward
        # is a normalized reward when `--norm_reward` flag is passed
        print("Episode Reward: {:.2f}".format(running_reward))
        print("Episode Length", ep_len)
        running_reward = 0.0
        ep_len = 0

env.reset()
env.envs[0].env.exit_scene()
# Close connection does work properly for now
# env.envs[0].env.close_connection()
time.sleep(0.5)

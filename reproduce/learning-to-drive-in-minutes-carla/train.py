# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin
import argparse
import os
import time
from collections import OrderedDict
from pprint import pprint

import numpy as np
import yaml
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.ppo2.ppo2 import constfn
from stable_baselines.gail import ExpertDataset

from config import MIN_THROTTLE, MAX_THROTTLE, FRAME_SKIP,\
    MAX_CTE_ERROR, SIM_PARAMS, N_COMMAND_HISTORY, Z_SIZE, MAX_STEERING_DIFF,\
    IMAGE_WIDTH, IMAGE_HEIGHT, ID_ENV, SIZE_Z # wyb
from utils.utils import make_env, ALGOS, linear_schedule, get_latest_run_id, load_vae, create_callback
from teleop.teleop_client import TeleopEnv
from teleop.recorder import Recorder

parser = argparse.ArgumentParser()
parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                    default='', type=str)
parser.add_argument('--algo', help='RL Algorithm', default='sac',
                    type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=50000,
                    type=int)
parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                    type=int)
parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='vae/model/carla/vae-{}_best.pkl'.format(SIZE_Z)) # wyb 'models/donkey/vae/vae-level-0-dim-32.pkl'
parser.add_argument('--save-vae', action='store_true', default=False,
                    help='Save VAE')
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--level', help='Level index', type=int, default=0)
parser.add_argument('--random-features', action='store_true', default=False,
                    help='Use random features')
parser.add_argument('--teleop', action='store_true', default=False,
                    help='Use teleoperation for training')
parser.add_argument('-pretrain', '--pretrain-path', type=str,
                    help='Path to an expert dataset for pretraining')
parser.add_argument('--n-epochs', type=int, default=50,
                    help='Number of epochs when doing pretraining')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Minibatch size when doing pretraining')
parser.add_argument('--traj-limitation', type=int, default=-1,
                    help='The number of trajectory to use (if -1, load all)')
parser.add_argument('--name-env', type=str, default='Carla',  # wyb
                    help='env name')            
args = parser.parse_args()

set_global_seeds(args.seed)

if args.trained_agent != "":
    assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
        "The trained_agent must be a valid path to a .pkl file"

tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, ID_ENV)

print("=" * 10, ID_ENV, args.algo, "=" * 10)

vae = None
if args.vae_path != '':
    print("Loading VAE ...")
    vae = load_vae(args.vae_path, input_dimension=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
elif args.random_features:
    print("Randomly initialized VAE")
    vae = load_vae(z_size=Z_SIZE)
    # Save network
    args.save_vae = True
else:
    print("Learning from pixels...")

# Load hyperparameters from yaml file
with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
    hyperparams = yaml.load(f)[args.name_env] # wyb

# Sort hyperparams that will be saved
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
# save vae path
saved_hyperparams['vae_path'] = args.vae_path
if vae is not None:
    saved_hyperparams['z_size'] = vae.z_size

# Save simulation params
for key in SIM_PARAMS:
    saved_hyperparams[key] = eval(key)
saved_hyperparams['seed'] = args.seed
pprint(saved_hyperparams)

# Compute and create log path
log_path = os.path.join(args.log_folder, args.algo)
save_path = os.path.join(log_path, "{}_{}".format(ID_ENV+'_'+str(SIZE_Z), get_latest_run_id(log_path, ID_ENV) + 1))
params_path = os.path.join(save_path, ID_ENV)
os.makedirs(params_path, exist_ok=True)

# Create learning rate schedules for ppo2 and sac
if args.algo in ["ppo2", "sac"]:
    for key in ['learning_rate', 'cliprange']:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split('_')
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], float):
            hyperparams[key] = constfn(hyperparams[key])
        else:
            raise ValueError('Invalid valid for {}: {}'.format(key, hyperparams[key]))

# Should we overwrite the number of timesteps?
if args.n_timesteps > 0:
    n_timesteps = args.n_timesteps
else:
    n_timesteps = int(hyperparams['n_timesteps'])
del hyperparams['n_timesteps']

normalize = False
normalize_kwargs = {}
if 'normalize' in hyperparams.keys():
    normalize = hyperparams['normalize']
    if isinstance(normalize, str):
        normalize_kwargs = eval(normalize)
        normalize = True
    del hyperparams['normalize']

if 'policy_kwargs' in hyperparams.keys():
    hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

if not args.teleop:
    env = DummyVecEnv([make_env(args.level, args.seed, vae=vae, teleop=args.teleop, obs_res=(IMAGE_WIDTH, IMAGE_HEIGHT))])
    # env_fn = make_env(args.level, args.seed, vae=vae, teleop=args.teleop, obs_res=(IMAGE_WIDTH, IMAGE_HEIGHT))
    # env = env_fn()
else:
    env = make_env(args.level, args.seed, vae=vae, teleop=args.teleop,
                   n_stack=hyperparams.get('frame_stack', 1), obs_res=(IMAGE_WIDTH, IMAGE_HEIGHT))()

if normalize:
    if hyperparams.get('normalize', False) and args.algo in ['ddpg']:
        print("WARNING: normalization not supported yet for DDPG")
    else:
        print("Normalizing input and return")
        env = VecNormalize(env, **normalize_kwargs)

# Optional Frame-stacking
n_stack = 1
if hyperparams.get('frame_stack', False):
    n_stack = hyperparams['frame_stack']
    if not args.teleop:
        env = VecFrameStack(env, n_stack)
    print("Stacking {} frames".format(n_stack))
    del hyperparams['frame_stack']

# Parse noise string for DDPG and SAC
if args.algo in ['ddpg', 'sac'] and hyperparams.get('noise_type') is not None:
    noise_type = hyperparams['noise_type'].strip()
    noise_std = hyperparams['noise_std']
    n_actions = env.action_space.shape[0]
    if 'adaptive-param' in noise_type:
        hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                            desired_action_stddev=noise_std)
    elif 'normal' in noise_type:
        hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                        sigma=noise_std * np.ones(n_actions))
    elif 'ornstein-uhlenbeck' in noise_type:
        hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                   sigma=noise_std * np.ones(n_actions))
    else:
        raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
    print("Applying {} noise with std {}".format(noise_type, noise_std))
    del hyperparams['noise_type']
    del hyperparams['noise_std']

if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
    # Continue training
    print("Loading pretrained agent")
    # Policy should not be changed
    del hyperparams['policy']

    model = ALGOS[args.algo].load(args.trained_agent, env=env,
                                  tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

    exp_folder = args.trained_agent.split('.pkl')[0]
    if normalize:
        print("Loading saved running average")
        env.load_running_average(exp_folder)
else:
    # Train an agent from scratch
    model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

if args.pretrain_path is not None:
    print("Petraining model for {} epochs".format(args.n_epochs))
    if os.path.isdir(args.pretrain_path):
        args.pretrain_path = os.path.join(args.pretrain_path, 'expert_dataset.npz')
    assert args.pretrain_path.endswith('.npz') and os.path.isfile(args.pretrain_path), "Invalid pretain path: {}".format(args.pretrain_path)
    expert_dataset = np.load(args.pretrain_path)
    # Convert dataset if needed
    if vae is not None:
        print("Converting to vae latent space...")
        expert_dataset = Recorder.convert_obs_to_latent_vec(expert_dataset, vae, N_COMMAND_HISTORY)
    # Create the dataloader and petrain (Behavior-Cloning)
    dataset = ExpertDataset(traj_data=expert_dataset,
                            traj_limitation=args.traj_limitation, batch_size=args.batch_size)

    # Fill the replay buffer
    if args.algo == "sac":
        print("Filling replay buffer")
        for i in range(len(expert_dataset['obs']) - 1):
            done = expert_dataset['episode_starts'][i + 1]
            obs, next_obs = expert_dataset['obs'][i], expert_dataset['obs'][i + 1]
            action, reward = expert_dataset['actions'][i], expert_dataset['rewards'][i]
            model.replay_buffer.add(obs, action, reward, next_obs, float(done))
        # Initialize the value fn
        model.n_updates = 0
        for _ in range(10):
            model.optimize(max(model.batch_size, model.learning_starts), None, model.learning_rate(1))
    else:
        # TODO: pretrain also the std to match the one from the dataset
        model.pretrain(dataset, n_epochs=args.n_epochs)
    del dataset

# Teleoperation mode:
# we don't wrap the environment with a monitor or in a vecenv
if args.teleop:
    assert args.algo == "sac", "Teleoperation mode is not yet implemented for {}".format(args.algo)
    env = TeleopEnv(env, is_training=True)
    model.set_env(env)
    env.model = model

kwargs = {}
if args.log_interval > -1:
    kwargs = {'log_interval': args.log_interval}

if args.algo == 'sac':
    kwargs.update({'callback': create_callback(args.algo,
                                               os.path.join(save_path, ID_ENV + "_best"),
                                               verbose=1)})

model.learn(n_timesteps, **kwargs)

if args.teleop:
    env.wait()
    env.exit()
    time.sleep(0.5)
else:
    # Close the connection properly
    env.reset()
    if isinstance(env, VecFrameStack):
        env = env.venv
    # # HACK to bypass Monitor wrapper
    # env.envs[0].env.exit_scene()

    env.close() # wyb

print("Saving model to {}".format(save_path))
# Save trained model
model.save(os.path.join(save_path, ID_ENV))
# Save hyperparams
with open(os.path.join(params_path, 'config.yml'), 'w') as f:
    yaml.dump(saved_hyperparams, f)

if args.save_vae and vae is not None:
    print("Saving VAE")
    vae.save(os.path.join(params_path, 'vae'))

if normalize:
    # Unwrap
    if isinstance(env, VecFrameStack):
        env = env.venv
    # Important: save the running average, for testing the agent we need that normalization
    env.save_running_average(params_path)

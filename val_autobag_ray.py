import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from decap_env import *
import os
import argparse
import sys

os.environ['RAY_worker_register_timeout_seconds'] = '12000'
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str)
args = parser.parse_args()
ray.init(include_dashboard=True, dashboard_port=9999, num_cpus=41)

#configures training of the agent with associated hyperparameters
#See Ray documentation for details on each parameter
config_train = {
            #"sample_batch_size": 200,
            "train_batch_size": 600,
            #"sgd_minibatch_size": 1200,
            #"num_sgd_iter": 3,
            "lr":1e-4,#tune.loguniform(1e-3, 1e-5),
            #"vf_loss_coeff": 0.5,
            #"simple_optimizer": True,
            #"disable_env_checking": True,
            "recreate_failed_workers":True,
            "horizon": 30,#tune.grid_search([30, 50]),
            "num_gpus": 0,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": 40,
            "env_config":{},#{"generalize":True, "run_valid":False},
            }

#Runs training and saves the result in ~/ray_results/train_ngspice_45nm
#If checkpoint fails for any reason, training can be restored 
if not args.checkpoint_dir:
    trials = tune.run_experiments({
        "decap_rl_09162022": {
        "checkpoint_freq":1,
        "run": "PPO",
        "env": DecapPlaceParallel,
        "stop": {"episode_reward_mean": -0.02, "timesteps_total":60000},
        "config": config_train},
    })
else:
    print("RESTORING NOW!!!!!!")
    tune.run_experiments({
        "restore_ppo": {
        "run": "PPO",
        "config": config_train,
        "env": DecapPlaceParallel,
        #"restore": trials[0]._checkpoint.value},
        "restore": args.checkpoint_dir,
        "checkpoint_freq":1},
    })
ray.shutdown()

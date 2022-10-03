import argparse
import itertools

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.make_env import make_env
from utils.network import PolicyNetwork


def evaluate(
    dir_name,
    policy,
    config,
    num_evaluate_point,
    dim_evaluate_point,
    evaluate_num,
    device,
    state_dim,
):
    '''Evaluate policy

    Parameters
    ----------
    dir_name : str
        Path of the directory corresponding to the policy to be evaluated
    policy : PolicyNetwork
        policy network
    config : Dict
        configs
    num_evaluate_point : int
        Number to partition the uncertainty parameter set
    dim_evaluate_point : int
        Dimension of uncertainty parameter
    evaluate_num : int
        Number of times a measure is evaluated for a given environment
    device : torch.Device
        device
    state_dim : int
        dimension of state
    '''

    if dim_evaluate_point == 1:
        evaluate_points = np.linspace(
            config["environment"]["change_param_min"],
            config["environment"]["change_param_max"],
            num_evaluate_point,
        )
    elif dim_evaluate_point == 2:
        eval_list0 = np.linspace(
            config["environment"]["change_param_min"][0],
            config["environment"]["change_param_max"][0],
            num_evaluate_point,
        )
        eval_list1 = np.linspace(
            config["environment"]["change_param_min"][1],
            config["environment"]["change_param_max"][1],
            num_evaluate_point,
        )
        evaluate_points = list(itertools.product(eval_list0, eval_list1))
    elif dim_evaluate_point == 3:
        eval_list0 = np.linspace(
            config["environment"]["change_param_min"][0],
            config["environment"]["change_param_max"][0],
            num_evaluate_point,
        )
        eval_list1 = np.linspace(
            config["environment"]["change_param_min"][1],
            config["environment"]["change_param_max"][1],
            num_evaluate_point,
        )
        eval_list2 = np.linspace(
            config["environment"]["change_param_min"][2],
            config["environment"]["change_param_max"][2],
            num_evaluate_point,
        )
        evaluate_points = list(itertools.product(eval_list0, eval_list1, eval_list2))
    else:
        raise NotImplementedError()

    output_file_path = f"{dir_name}/evaluate_each.txt"
    with open(output_file_path, "w") as f:
        for evaluate_point in evaluate_points:
            env = make_env(
                config["environment"]["env_name"],
                config["environment"]["change_param_names"],
                evaluate_point,
                config["system"]["seed"],
                config["algorithm"]["name"],
                config["environment"]["xml_file"],
                config["xml_name"],
                "./",
            )
            reward_list = simulation(env, policy, evaluate_num, device, state_dim)
            reward_list = [str(reward) for reward in reward_list]
            reward_str = ",".join(reward_list)
            f.write(f"{evaluate_point},{reward_str}\n")


def simulation(env, policy, evaluate_num, device, state_dim):
    '''Interaction to test environment

    Parameters
    ----------
    env : gym.Env
        gym environment
    policy : PolicyNetwork
        policy network
    evaluate_num : int
        Number of times a measure is evaluated for a given environment
    device : torch.Device
        device
    state_dim : int
        dimension of state
    '''

    rewards = []
    for _ in range(evaluate_num):
        state = env.reset()
        total_reward = 0
        for _ in range(env._max_episode_steps):
            state_tensor = torch.tensor(state, dtype=torch.float, device=device).view(
                -1, state_dim
            )
            action = policy(state_tensor).squeeze(0).detach().cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    return rewards


def main(
    dir_name,
    iteration,
    num_evaluate_point,
    dim_evaluate_point,
    evaluate_num,
    device,
    xml_name,
):
    '''Main

    Parameters
    ----------
    dir_name : str
        Path of the directory corresponding to the policy to be evaluated
    iteration : int
        Iteration of policy to be evaluated
    num_evaluate_point : int
        Number to partition the uncertainty parameter set
    dim_evaluate_point : int
        Dimension of uncertainty parameter
    evaluate_num : int
        Number of times a measure is evaluated for a given environment
    device : torch.Device
        device
    xml_name : str
        Xml file path of mujoco

    '''
    hyperparameters_path = f"{dir_name}/hyperparameter.yaml"
    with open(hyperparameters_path, "r") as f:
        config = OmegaConf.load(f)
    if "xml_name" not in config.keys():
        config["xml_name"] = xml_name
    env = make_env(
        config["environment"]["env_name"],
        [],
        [],
        config["system"]["seed"],
        config["algorithm"]["name"],
        config["environment"]["xml_file"],
        config["xml_name"],
        "./",
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    policy = PolicyNetwork(
        state_dim,
        action_dim,
        config["network"]["policy_hidden_num"],
        config["network"]["policy_hidden_size"],
        max_action,
        device,
    )
    policy_path = f"{dir_name}/policies/policy-{iteration}.pkl"
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device("cpu")))
    policy.to(device)

    evaluate(
        dir_name,
        policy,
        config,
        num_evaluate_point,
        dim_evaluate_point,
        evaluate_num,
        device,
        state_dim,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--num_evaluate_point", type=int, default=10)
    parser.add_argument("--dim_evaluate_point", type=int, default=None)
    parser.add_argument("--evaluate_num", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--xml_name", type=str, default=None)
    args = parser.parse_args()
    main(
        args.dir,
        args.iteration,
        args.num_evaluate_point,
        args.dim_evaluate_point,
        args.evaluate_num,
        args.device,
        args.xml_name,
    )

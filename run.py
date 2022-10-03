import os
import subprocess

SEEDS = range(5)

ALGORITHMS_1 = ["soft_m2td3", "m2td3"]

ENVIRONMENTS_1_1 = ["Antv2-1_3", "HalfCheetahv2-1_4", "InvertedPendulumv2-1_31"]
ENVIRONMENTS_1_2 = [
    "HumanoidStandupv2-1_16",
    "Hopperv2-1_3",
    "Walker2dv2-1_4",
]

ENVIRONMENTS_2_1 = ["Antv2-2_3_3", "Walker2dv2-2_4_5", "Hopperv2-2_3_3"]
ENVIRONMENTS_2_2 = [
    "InvertedPendulumv2-2_31_11",
    "HalfCheetahv2-2_4_7",
    "HumanoidStandupv2-2_16_8",
]

ENVIRONMENTS_3_1 = ["Antv2-3_3_3_3", "HalfCheetahv2-3_4_7_4", "Hopperv2-3_3_3_4"]
ENVIRONMENTS_3_2 = [
    "HumanoidStandupv2-3_16_5_8",
    "Walker2dv2-3_4_5_6",
]

ENVIRONMENTS_SMALL = ["HalfCheetahv2-1_3", "Hopperv2-1_2"]

MAX_STEPS_DICT = {1: 2000000, 2: 4000000, 3: 5000000}
EVALUATE_INTERVAL_DICT = {1: 100000, 2: 200000, 3: 250000}


def wait(procs):
    for proc_i, proc in enumerate(procs):
        _ = proc.communicate()
    return []


def extract_dim(environment):
    dim = int(environment[environment.index("-") + 1])
    return dim


def make_train_proc(algorithm, environment, seed, max_steps, evaluate_interval):
    train_cmd = [
        "python",
        "main.py",
        f"algorithm={algorithm}",
        f"environment={environment}",
        f"system.seed={seed}",
        f"experiment_name={seed}_{algorithm}_{environment}",
        f"algorithm.max_steps={max_steps}",
        f"evaluation.evaluate_interval={evaluate_interval}",
    ]

    with open(f"logs/{seed}_{algorithm}_{environment}.log", "w") as f:
        proc = subprocess.Popen(
            train_cmd, stdout=f, stderr=subprocess.PIPE, env=os.environ
        )
    return proc


def make_eval_proc(algorithm, environment, seed, max_steps, evaluate_interval, dim):
    if dim == 1 or dim == 2:
        eval_cmd = [
            "python",
            "evaluate.py",
            "--dir",
            f"experiments/{seed}_{algorithm}_{environment}",
            "--interval",
            f"{evaluate_interval}",
            "--max_iteration",
            f"{max_steps}",
            "--dim_evaluate_point",
            f"{dim}",
        ]
    elif dim == 3:
        eval_cmd = [
            "python",
            "evaluate_last.py",
            "--dir",
            f"experiments/{seed}_{algorithm}_{environment}",
            "--iteration",
            f"{max_steps}",
            "--dim_evaluate_point",
            f"{dim}",
        ]
    else:
        raise RuntimeError()
    with open(f"logs/{seed}_{algorithm}_{environment}.log", "a") as f:
        proc = subprocess.Popen(
            eval_cmd, stdout=f, stderr=subprocess.PIPE, env=os.environ
        )
    return proc


def run_training(environment, algorithms):
    procs = []
    dim = extract_dim(environment)
    max_steps, evaluate_interval = MAX_STEPS_DICT[dim], EVALUATE_INTERVAL_DICT[dim]
    for algorithm in algorithms:
        for seed in SEEDS:
            proc = make_train_proc(
                algorithm, environment, seed, max_steps, evaluate_interval
            )
            procs.append(proc)
    return procs


def run_training_env(algorithm, environments):
    procs = []
    dim = extract_dim(environments[0])
    max_steps, evaluate_interval = MAX_STEPS_DICT[dim], EVALUATE_INTERVAL_DICT[dim]
    for environment in environments:
        for seed in SEEDS:
            proc = make_train_proc(
                algorithm, environment, seed, max_steps, evaluate_interval
            )
            procs.append(proc)
    return procs


def run_evaluate(environment, algorithms):
    procs = []
    dim = extract_dim(environment)
    max_steps, evaluate_interval = MAX_STEPS_DICT[dim], EVALUATE_INTERVAL_DICT[dim]
    for algorithm in algorithms:
        for seed in SEEDS:
            proc = make_eval_proc(
                algorithm, environment, seed, max_steps, evaluate_interval, dim
            )
            procs.append(proc)
    return procs


def run_evaluate_env(algorithm, environments):
    procs = []
    dim = extract_dim(environments[0])
    max_steps, evaluate_interval = MAX_STEPS_DICT[dim], EVALUATE_INTERVAL_DICT[dim]
    for environment in environments:
        for seed in SEEDS:
            proc = make_eval_proc(
                algorithm, environment, seed, max_steps, evaluate_interval, dim
            )
            procs.append(proc)
    return procs


def run_algorithms(environment, algorithms):
    procs = run_training(environment, algorithms)
    procs = wait(procs)
    procs = run_evaluate(environment, algorithms)
    procs = wait(procs)


def run_environments(algorithm, environments):
    procs = run_training_env(algorithm, environments)
    procs = wait(procs)
    procs = run_evaluate_env(algorithm, environments)
    procs = wait(procs)


def main():
    for environment in (
        ENVIRONMENTS_1_1
        + ENVIRONMENTS_1_2
        + ENVIRONMENTS_2_1
        + ENVIRONMENTS_2_2
        + ENVIRONMENTS_3_1
        + ENVIRONMENTS_3_2
        + ENVIRONMENTS_SMALL
    ):
        run_algorithms(environment, ALGORITHMS_1)


if __name__ == "__main__":
    main()

import os
import xml.etree.ElementTree as ET
from typing import List

import gym


def make_env(
    env_name,
    change_param_names,
    change_param_values,
    seed,
    alg_name,
    base_xml_file,
    xml_name,
    gym_path = "../../",
):
    '''Create an environment with specified physical parameters

    Parameters
    ----------
    env_name : str
        environement name
    change_param_names : List[str]
        Names of physical parameter to be changed
    change_param_values : List[float]
        Values of physical parameter to be changed
    seed : int
        random seed
    alg_name : str
        name of algorithm
    base_xml_file : str
        Xml file path of mujoco
    xml_name : str
        Prefix xml file
    gym_path : str
        Path of gym library

    '''

    if env_name == "HalfCheetah-v2":
        env = make_halfcheetah(
            env_name,
            change_param_names,
            change_param_values,
            seed,
            alg_name,
            base_xml_file,
            xml_name,
            gym_path,
        )
    elif env_name == "InvertedPendulum-v2":
        env = make_invertedpendulum(
            env_name,
            change_param_names,
            change_param_values,
            seed,
            alg_name,
            base_xml_file,
            xml_name,
            gym_path,
        )
    elif env_name == "Walker2d-v2":
        env = make_walker(
            env_name,
            change_param_names,
            change_param_values,
            seed,
            alg_name,
            base_xml_file,
            xml_name,
            gym_path,
        )
    elif env_name == "Hopper-v2":
        env = make_hopper(
            env_name,
            change_param_names,
            change_param_values,
            seed,
            alg_name,
            base_xml_file,
            xml_name,
            gym_path,
        )
    elif env_name == "Ant-v2":
        env = make_ant(
            env_name,
            change_param_names,
            change_param_values,
            seed,
            alg_name,
            base_xml_file,
            xml_name,
            gym_path,
        )
    elif env_name == "HumanoidStandup-v2":
        env = make_humanoidstandup(
            env_name,
            change_param_names,
            change_param_values,
            seed,
            alg_name,
            base_xml_file,
            xml_name,
            gym_path,
        )
    else:
        raise NotImplementedError()
    return env


def make_halfcheetah(
    env_name,
    change_param_names,
    change_param_values,
    seed,
    alg_name,
    base_xml_file,
    xml_name,
    gym_path,
):
    '''Create an environment with specified physical parameters

    Parameters
    ----------
    env_name : str
        environement name
    change_param_names : List[str]
        Names of physical parameter to be changed
    change_param_values : List[float]
        Values of physical parameter to be changed
    seed : int
        random seed
    alg_name : str
        name of algorithm
    base_xml_file : str
        Xml file path of mujoco
    xml_name : str
        Prefix xml file
    gym_path : str
        Path of gym library

    '''

    change_flags = {
        change_param_name: False for change_param_name in change_param_names
    }

    tree = ET.parse(f"{gym_path}gym/envs/mujoco/assets/{base_xml_file}")
    root = tree.getroot()
    for change_param_name, change_param_value in zip(
        change_param_names, change_param_values
    ):
        for name in root.iter("default"):
            for n in name.iter("geom"):
                l = n.attrib["friction"].split()
                if change_param_name == "worldfriction":
                    l[0] = str(change_param_value)  # world friction
                    n.attrib["friction"] = " ".join(l)
                    change_flags[change_param_name] = True
    fname = f"{gym_path}gym/envs/mujoco/assets/{env_name}_{len(change_param_names)}_new_{alg_name}_{seed}_{xml_name}.xml"
    tree.write(fname)
    env = gym.make(env_name, xml_file=os.path.basename(fname))
    for change_param_name, change_param_value in zip(
        change_param_names, change_param_values
    ):
        if change_param_name == "torsomass":
            env.env.model.body_mass[1] = change_param_value
            change_flags[change_param_name] = True
        if change_param_name == "backthighmass":
            env.env.model.body_mass[2] = change_param_value
            change_flags[change_param_name] = True
    for key, value in change_flags.items():
        if not value:
            print("not change", key)
            raise RuntimeError()
    return env


def make_invertedpendulum(
    env_name,
    change_param_names,
    change_param_values,
    seed,
    alg_name,
    base_xml_file,
    xml_name,
    gym_path,
):
    '''Create an environment with specified physical parameters

    Parameters
    ----------
    env_name : str
        environement name
    change_param_names : List[str]
        Names of physical parameter to be changed
    change_param_values : List[float]
        Values of physical parameter to be changed
    seed : int
        random seed
    alg_name : str
        name of algorithm
    base_xml_file : str
        Xml file path of mujoco
    xml_name : str
        Prefix xml file
    gym_path : str
        Path of gym library

    '''

    change_flags = {
        change_param_name: False for change_param_name in change_param_names
    }

    env = gym.make(env_name, xml_file=os.path.basename(base_xml_file))
    for change_param_name, change_param_value in zip(
        change_param_names, change_param_values
    ):
        if change_param_name == "polemass":
            env.env.model.body_mass[2] = change_param_value
            change_flags[change_param_name] = True
        if change_param_name == "cartmass":
            env.env.model.body_mass[1] = change_param_value
            change_flags[change_param_name] = True
    for key, value in change_flags.items():
        if not value:
            print("not change", key)
            raise RuntimeError()
    return env


def make_walker(
    env_name,
    change_param_names,
    change_param_values,
    seed,
    alg_name,
    base_xml_file,
    xml_name,
    gym_path,
):
    '''Create an environment with specified physical parameters

    Parameters
    ----------
    env_name : str
        environement name
    change_param_names : List[str]
        Names of physical parameter to be changed
    change_param_values : List[float]
        Values of physical parameter to be changed
    seed : int
        random seed
    alg_name : str
        name of algorithm
    base_xml_file : str
        Xml file path of mujoco
    xml_name : str
        Prefix xml file
    gym_path : str
        Path of gym library

    '''

    change_flags = {
        change_param_name: False for change_param_name in change_param_names
    }

    tree = ET.parse(f"{gym_path}gym/envs/mujoco/assets/{base_xml_file}")
    root = tree.getroot()
    for change_param_name, change_param_value in zip(
        change_param_names, change_param_values
    ):
        for name in root.iter("default"):
            for n in name.iter("geom"):
                l = n.attrib["friction"].split()
                if change_param_name == "worldfriction":
                    l[0] = str(change_param_value)  # world friction
                    n.attrib["friction"] = " ".join(l)
                    change_flags[change_param_name] = True
    fname = f"{gym_path}gym/envs/mujoco/assets/{env_name}_{len(change_param_names)}_new_{alg_name}_{seed}_{xml_name}.xml"
    tree.write(fname)
    env = gym.make(env_name, xml_file=os.path.basename(fname))
    for change_param_name, change_param_value in zip(
        change_param_names, change_param_values
    ):
        if change_param_name == "torsomass":
            env.env.model.body_mass[1] = change_param_value
            change_flags[change_param_name] = True
        if change_param_name == "thighmass":
            env.env.model.body_mass[2] = change_param_value
            change_flags[change_param_name] = True
    for key, value in change_flags.items():
        if not value:
            print("not change", key)
            raise RuntimeError()
    return env


def make_hopper(
    env_name,
    change_param_names,
    change_param_values,
    seed,
    alg_name,
    base_xml_file,
    xml_name,
    gym_path,
):
    '''Create an environment with specified physical parameters

    Parameters
    ----------
    env_name : str
        environement name
    change_param_names : List[str]
        Names of physical parameter to be changed
    change_param_values : List[float]
        Values of physical parameter to be changed
    seed : int
        random seed
    alg_name : str
        name of algorithm
    base_xml_file : str
        Xml file path of mujoco
    xml_name : str
        Prefix xml file
    gym_path : str
        Path of gym library

    '''

    change_flags = {
        change_param_name: False for change_param_name in change_param_names
    }

    tree = ET.parse(f"{gym_path}gym/envs/mujoco/assets/{base_xml_file}")
    root = tree.getroot()
    for change_param_name, change_param_value in zip(
        change_param_names, change_param_values
    ):
        for name in root.iter("default"):
            for n in name.iter("geom"):
                l = n.attrib["friction"].split()
                if change_param_name == "worldfriction":
                    l[0] = str(change_param_value)  # world friction
                    n.attrib["friction"] = " ".join(l)
                    change_flags[change_param_name] = True
    fname = f"{gym_path}gym/envs/mujoco/assets/{env_name}_{len(change_param_names)}_new_{alg_name}_{seed}_{xml_name}.xml"
    tree.write(fname)
    env = gym.make(env_name, xml_file=os.path.basename(fname))
    for change_param_name, change_param_value in zip(
        change_param_names, change_param_values
    ):
        if change_param_name == "torsomass":
            env.env.model.body_mass[1] = change_param_value
            change_flags[change_param_name] = True
        if change_param_name == "thighmass":
            env.env.model.body_mass[2] = change_param_value
            change_flags[change_param_name] = True
    for key, value in change_flags.items():
        if not value:
            print("not change", key)
            raise RuntimeError()
    return env


def make_ant(
    env_name,
    change_param_names,
    change_param_values,
    seed,
    alg_name,
    base_xml_file,
    xml_name,
    gym_path,
):
    '''Create an environment with specified physical parameters

    Parameters
    ----------
    env_name : str
        environement name
    change_param_names : List[str]
        Names of physical parameter to be changed
    change_param_values : List[float]
        Values of physical parameter to be changed
    seed : int
        random seed
    alg_name : str
        name of algorithm
    base_xml_file : str
        Xml file path of mujoco
    xml_name : str
        Prefix xml file
    gym_path : str
        Path of gym library

    '''

    change_flags = {
        change_param_name: False for change_param_name in change_param_names
    }

    env = gym.make(env_name, xml_file=os.path.basename(base_xml_file))
    for change_param_name, change_param_value in zip(
        change_param_names, change_param_values
    ):
        if change_param_name == "torsomass":
            env.env.model.body_mass[1] = change_param_value
            change_flags[change_param_name] = True
        if change_param_name == "frontleftlegmass":
            env.env.model.body_mass[2] = change_param_value
            change_flags[change_param_name] = True
        if change_param_name == "frontrightlegmass":
            env.env.model.body_mass[4] = change_param_value
            change_flags[change_param_name] = True
    for key, value in change_flags.items():
        if not value:
            print("not change", key)
            raise RuntimeError()
    return env


def make_humanoidstandup(
    env_name,
    change_param_names,
    change_param_values,
    seed,
    alg_name,
    base_xml_file,
    xml_name,
    gym_path,
):
    '''Create an environment with specified physical parameters

    Parameters
    ----------
    env_name : str
        environement name
    change_param_names : List[str]
        Names of physical parameter to be changed
    change_param_values : List[float]
        Values of physical parameter to be changed
    seed : int
        random seed
    alg_name : str
        name of algorithm
    base_xml_file : str
        Xml file path of mujoco
    xml_name : str
        Prefix xml file
    gym_path : str
        Path of gym library

    '''

    change_flags = {
        change_param_name: False for change_param_name in change_param_names
    }

    env = gym.make(env_name, xml_file=os.path.basename(base_xml_file))
    for change_param_name, change_param_value in zip(
        change_param_names, change_param_values
    ):
        if change_param_name == "torsomass":
            env.env.model.body_mass[1] = change_param_value
            change_flags[change_param_name] = True
        if change_param_name == "rightfootmass":
            env.env.model.body_mass[6] = change_param_value
            change_flags[change_param_name] = True
        if change_param_name == "leftthighmass":
            env.env.model.body_mass[7] = change_param_value
            change_flags[change_param_name] = True
    for key, value in change_flags.items():
        if not value:
            print("not change", key)
            raise RuntimeError()
    return env

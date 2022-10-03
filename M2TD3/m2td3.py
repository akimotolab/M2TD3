import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.network import CriticNetwork, HatOmegaNetwork, PolicyNetwork
from utils.replay_buffer import ReplayBuffer
from utils.transitions import Transition


class M2TD3:
    ''' M2TD3 agent

    Parameters
    ----------
    config : Dict
        configs
    state_dim : int
         Number of state dimensions
    action_dim : int
        Number of action dimensions
    omega_dim : int
        Number of omega dimensions
    max_action : float
        Maximum value of action
    rand_state : np.random.RandomState
        Control random numbers
    device : torch.device
        device

    '''

    def __init__(
        self, config, state_dim, action_dim, omega_dim, max_action, rand_state, device
    ):
        self.config = config
        self.device = device
        self.rand_state = rand_state

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.omega_dim = omega_dim
        self.min_omega = np.array(config["environment"]["change_param_min"])
        self.max_omega = np.array(config["environment"]["change_param_max"])
        self.min_omega_tensor = torch.tensor(
            config["environment"]["change_param_min"], dtype=torch.float, device=device
        )
        self.max_omega_tensor = torch.tensor(
            config["environment"]["change_param_max"], dtype=torch.float, device=device
        )
        self.max_action = max_action

        self.policy_std = config["algorithm"]["policy_std_rate"] * max_action
        self.policy_noise = config["algorithm"]["policy_noise_rate"] * max_action
        self.noise_clip_policy = (
            config["algorithm"]["noise_clip_policy_rate"] * max_action
        )

        self.omega_std = list(
            config["algorithm"]["omega_std_rate"]
            * (self.max_omega - self.min_omega)
            / 2
        )
        self.min_omega_std = list(
            config["algorithm"]["min_omega_std_rate"]
            * (self.max_omega - self.min_omega)
            / 2
        )
        self.omega_std_step = (
            np.array(self.omega_std) - np.array(self.min_omega_std)
        ) / (self.config["algorithm"]["max_steps"])

        self.omega_noise = (
            config["algorithm"]["policy_noise_rate"]
            * (self.max_omega - self.min_omega)
            / 2
        )
        self.noise_clip_omega = torch.tensor(
            config["algorithm"]["noise_clip_omega_rate"]
            * (self.max_omega - self.min_omega)
            / 2,
            device=self.device,
            dtype=torch.float,
        )

        self.hatomega_input = torch.tensor([[1]], dtype=torch.float, device=self.device)
        self.hatomega_input_batch = torch.tensor(
            [[1] * self.config["algorithm"]["batch_size"]],
            dtype=torch.float,
            device=self.device,
        ).view(self.config["algorithm"]["batch_size"], 1)
        self.hatomega_prob = [
            1 / config["network"]["hatomega_num"]
            for _ in range(config["network"]["hatomega_num"])
        ]
        self.element_list = [i for i in range(config["network"]["hatomega_num"])]
        self.update_omega = [0 for _ in range(len(self.max_omega))]

        self._init_network(
            state_dim, action_dim, config["environment"]["dim"], max_action, config
        )
        self._init_optimizer(config)

        self.replay_buffer = ReplayBuffer(
            rand_state, capacity=config["algorithm"]["replay_size"]
        )

    def _init_network(self, state_dim, action_dim, omega_dim, max_action, config):
        '''Initialize network

        Parameters
        ----------
        state_dim : int
            Number of state dimensions
        action_dim : int
            Number of action dimensions
        omega_dim : int
            Number of omega dimensions
        max_action : float
            Maximum value of action
        config : Dict
            configs

        '''

        self.policy_network = PolicyNetwork(
            state_dim,
            action_dim,
            config["network"]["policy_hidden_num"],
            config["network"]["policy_hidden_size"],
            max_action,
            self.device,
        ).to(self.device)

        self.critic_network = CriticNetwork(
            state_dim,
            action_dim,
            omega_dim,
            config["network"]["critic_hidden_num"],
            config["network"]["critic_hidden_size"],
            config["network"]["p_bias"],
        ).to(self.device)

        self.hatomega_list = [None for _ in range(config["network"]["hatomega_num"])]
        self.optimizer_hatomega_list = [
            None for _ in range(config["network"]["hatomega_num"])
        ]
        for i in range(config["network"]["hatomega_num"]):
            self._init_hatomega(i)

        self.policy_target = copy.deepcopy(self.policy_network)
        self.critic_target = copy.deepcopy(self.critic_network)

    def _init_optimizer(self, config):
        '''Initialize optimizer

        Parameters
        ----------
        config : Dict
            configs

        '''

        self.optimizer_policy = optim.Adam(
            self.policy_network.parameters(), lr=config["algorithm"]["p_lr"]
        )
        self.optimizer_critic_p = optim.Adam(
            self.critic_network.parameters(), lr=config["algorithm"]["q_lr"]
        )

    def add_memory(self, *args):
        '''Add transition to replay buffer

        Parameters
        ----------
        args :
            "state", "action", "next_state", "reward", "done", "omega"

        '''

        transition = Transition(*args)
        self.replay_buffer.append(transition)

    def get_action(self, state, greedy=False):
        '''Get action

        Parameters
        ----------
        state : np.Array
            state
        greedy : bool
            flag whether or not to perform greedy behavior

        '''

        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).view(
            -1, self.state_dim
        )
        action = self.policy_network(state_tensor)
        if not greedy:
            noise = torch.tensor(
                self.rand_state.normal(0, self.policy_std),
                dtype=torch.float,
                device=self.device,
            )
            action = action + noise
        return action.squeeze(0).detach().cpu().numpy()

    def get_omega(self, greedy=False):
        '''Get omega

        Parameters
        ----------
        greedy : bool
            flag whether or not to perform greedy behavior

        '''

        dis_restart_flag = False
        prob_restart_flag = False
        if self.config["algorithm"]["restart_distance"]:
            change_hatomega_index_list = self._calc_diff()
            for index in change_hatomega_index_list:
                self._init_hatomega(index)
                self._init_hatomega_prob(index)
                dis_restart_flag = True
        if self.config["algorithm"]["restart_probability"]:
            change_hatomega_index_list = self._minimum_prob()
            for index in change_hatomega_index_list:
                self._init_hatomega(index)
                self._init_hatomega_prob(index)
                prob_restart_flag = True

        hatomega_index = self._select_hatomega()
        omega = self.hatomega_list[hatomega_index](self.hatomega_input)

        if not greedy:
            noise = torch.tensor(
                self.rand_state.normal(0, self.omega_std),
                dtype=torch.float,
                device=self.device,
            )
            omega += noise
        return (
            omega.squeeze(0).detach().cpu().numpy(),
            dis_restart_flag,
            prob_restart_flag,
        )

    def _buffer2batch(self):
        '''Make mini-batch from buffer datas

        '''

        transitions = self.replay_buffer.sample(self.config["algorithm"]["batch_size"])
        if transitions is None:
            return None, None, None, None, None, None
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(np.array(batch.state, dtype=float), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(np.array(batch.action, dtype=float), device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(
            np.array(batch.next_state, dtype=float), device=self.device, dtype=torch.float
        )
        reward_batch = torch.tensor(
            np.array(batch.reward, dtype=float), device=self.device, dtype=torch.float
        ).unsqueeze(1)
        not_done = np.array([(not don) for don in batch.done])
        not_done_batch = torch.tensor(
            np.array(not_done), device=self.device, dtype=torch.float
        ).unsqueeze(1)
        omega_batch = torch.tensor(np.array(batch.omega, dtype=float), device=self.device, dtype=torch.float)
        return (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            not_done_batch,
            omega_batch,
        )

    def train(self, step):
        '''Training one step

        Parameters
        ----------
        step : int
            Number of current step

        '''

        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            not_done_batch,
            omega_batch,
        ) = self._buffer2batch()
        if state_batch is None:
            return None, None, None

        self._update_critic(
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            not_done_batch,
            omega_batch,
        )
        if step % self.config["algorithm"]["policy_freq"] == 0:
            self._update_actor(state_batch)

            self._update_target()

    def _update_critic(
        self,
        state_batch,
        action_batch,
        next_state_batch,
        reward_batch,
        not_done_batch,
        omega_batch,
    ):
        '''Update critic network

        Parameters
        ----------
        state_batch : torch.Tensor
            state batch
        action_batch : torch.Tensor
            action batch
        next_state_batch : torch.Tensor
            next state batch
        reward_batch : torch.Tensor
            reward batch
        not_done_batch : torch.Tensor
            not done batch
        omega_batch : torch.Tensor
            omega batch

        '''

        with torch.no_grad():
            action_noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(
                -self.noise_clip_policy, self.noise_clip_policy
            )
            next_action_batch = (
                self.policy_target(next_state_batch) + action_noise
            ).clamp(-self.max_action, self.max_action)
            omega_noise = torch.max(
                torch.min(
                    (
                        torch.randn_like(omega_batch)
                        * torch.tensor(
                            self.omega_noise, device=self.device, dtype=torch.float
                        )
                    ),
                    self.noise_clip_omega,
                ),
                -self.noise_clip_omega,
            )
            next_omega_batch = torch.max(
                torch.min((omega_batch + omega_noise), self.max_omega_tensor),
                self.min_omega_tensor,
            )

            targetQ1, targetQ2 = self.critic_target(
                next_state_batch, next_action_batch, next_omega_batch
            )
            targetQ = torch.min(targetQ1, targetQ2)
            targetQ = (
                reward_batch
                + not_done_batch * self.config["algorithm"]["gamma"] * targetQ
            )

        currentQ1, currentQ2 = self.critic_network(state_batch, action_batch, omega_batch)
        critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)

        self.optimizer_critic_p.zero_grad()
        critic_loss.backward()
        self.optimizer_critic_p.step()

    def _update_actor(self, state_batch):
        '''Update actor network

        Parameters
        ----------
        state_batch : torch.Tensor
            state batch
        '''

        worst_policy_loss_index = None
        worst_policy_loss = None
        worst_policy_loss_value = -np.inf
        for hatomega_index in range(self.config["network"]["hatomega_num"]):
            hatomega_batch = self.hatomega_list[hatomega_index](self.hatomega_input_batch)

            policy_loss = -self.critic_network.Q1(
                state_batch, self.policy_network(state_batch), hatomega_batch.detach()
            ).mean()
            if policy_loss.item() >= worst_policy_loss_value:
                self.update_omega = list(
                    self.hatomega_list[hatomega_index](self.hatomega_input)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                worst_policy_loss = policy_loss
                worst_policy_loss_index = hatomega_index
                worst_policy_loss_value = policy_loss.item()

        hatomega_batch = self.hatomega_list[worst_policy_loss_index](self.hatomega_input_batch)

        hatomega_loss = self.critic_network.Q1(
            state_batch, self.policy_network(state_batch).detach(), hatomega_batch
        ).mean()
        self.optimizer_hatomega_list[worst_policy_loss_index].zero_grad()
        hatomega_loss.backward()
        self.optimizer_hatomega_list[worst_policy_loss_index].step()

        self.optimizer_policy.zero_grad()
        worst_policy_loss.backward()
        self.optimizer_policy.step()

        self._update_hatomega_prob(worst_policy_loss_index)

    def _update_target(self):
        '''Update target network

        '''

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic_network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config["algorithm"]["polyak"])
                + param.data * self.config["algorithm"]["polyak"]
            )

        for target_param, param in zip(
            self.policy_target.parameters(), self.policy_network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config["algorithm"]["polyak"])
                + param.data * self.config["algorithm"]["polyak"]
            )

    def _calc_diff(self):
        '''Compute the distance between hatomegas

        '''

        change_hatomega_index_list = []
        hatomega_parameter_list = []
        for i in range(self.config["network"]["hatomega_num"]):
            hatomega_parameter_list.append(
                self.hatomega_list[i](self.hatomega_input)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
        for hatomega_pair in itertools.combinations(hatomega_parameter_list, 2):
            distance = np.linalg.norm(
                np.array(hatomega_pair[0]) - np.array(hatomega_pair[1]), ord=1
            )
            if distance <= self.config["algorithm"]["hatomega_parameter_distance"]:
                change_hatomega_index_list.append(
                    hatomega_parameter_list.index(hatomega_pair[0])
                )
        return change_hatomega_index_list

    def _update_hatomega_prob(self, index):
        '''Update selection probability for hatomega

        Parameters
        ----------
        index : int
            Index of hatomega to be updated

        '''

        p = [0] * self.config["network"]["hatomega_num"]
        p[index] = 1
        coeff = 1 / self.current_episode_len
        for i in range(self.config["network"]["hatomega_num"]):
            self.hatomega_prob[i] = self.hatomega_prob[i] * (1 - coeff) + coeff * p[i]

    def _minimum_prob(self):
        '''Extract the index of the hatomega that has a selection probability below a threshold value.

        '''

        indexes = []
        for index in range(self.config["network"]["hatomega_num"]):
            prob = self.hatomega_prob[index]
            if prob < self.config["algorithm"]["minimum_prob"]:
                indexes.append(index)
        return indexes

    def _init_hatomega(self, index):
        '''Initialize hatomega

        Parameters
        ----------
        index : int
            Index of hatomega to initialize
        '''

        hatomega = HatOmegaNetwork(
            self.omega_dim,
            self.min_omega,
            self.max_omega,
            self.config["network"]["hatomega_hidden_num"],
            self.config["network"]["hatomega_hidden_size"],
            self.rand_state,
            self.device,
        ).to(self.device)
        optimizer_hatomega = optim.Adam(
            hatomega.parameters(), lr=self.config["algorithm"]["ho_lr"]
        )
        self.hatomega_list[index] = hatomega
        self.optimizer_hatomega_list[index] = optimizer_hatomega

    def _init_hatomega_prob(self, index):
        '''Initialize selection probability for hatomega

        Parameters
        ----------
        index : int
            Index of hatomega to initialize
        '''

        self.hatomega_prob[index] = 0
        sum_prob = sum(self.hatomega_prob)
        p = sum_prob / (self.config["network"]["hatomega_num"] - 1)
        self.hatomega_prob[index] = p

    def _select_hatomega(self):
        '''Select hatomega

        '''

        self.hatomega_prob = list(np.array(self.hatomega_prob) / sum(self.hatomega_prob))
        select_index = self.rand_state.choice(
            a=self.element_list, size=1, p=self.hatomega_prob
        )
        return select_index[0]

    def _update_omega_std(self):
        '''Update omega std

        '''

        if self.omega_std >= self.min_omega_std:
            self.omega_std = list(
                np.array(self.omega_std) - self.omega_std_step
            )

    def set_current_episode_len(self, episode_len):
        '''Set length of episode

        Parameters
        ----------
        episode_len : int
            Length of current episode.
        '''
        self.current_episode_len = episode_len

    def get_qvalue_list(self):
        '''Retrieve the Q value for each hatomega

        '''

        qvalue_list = []
        transitions = self.replay_buffer.sample(self.config["algorithm"]["batch_size"])
        for hatomega_index in range(self.config["network"]["hatomega_num"]):
            if transitions is None:
                qvalue_list.append(0)
                continue
            batch = Transition(*zip(*transitions))
            state_batch = torch.tensor(
                batch.state, device=self.device, dtype=torch.float
            )
            q_value = self.critic_network.Q1(
                state_batch,
                self.policy_network(state_batch),
                self.hatomega_list[hatomega_index](self.hatomega_input_batch),
            ).mean()
            qvalue_list.append(q_value.item())
        return qvalue_list

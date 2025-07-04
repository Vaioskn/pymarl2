import copy
import os
import numpy as np
import torch as th
from torch.optim import Adam

from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.modules.critics import REGISTRY as critic_registry
from pymarlzooplus.components.standarize_stream import PopArt


class HAPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.current_episode = 0
        self.n_agents = args.n_agents
        self.device = "cuda" if args.use_cuda else "cpu"

        # We don't use environments with heterogeneous agents, so just copying the same mac is ok
        self.learners = [HAPPO(copy.deepcopy(mac), agent_id, scheme, logger, args) for agent_id in range(self.n_agents)]

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        actions = []
        extra_returns = {"log_probs": [], "values": []}
        if self.learners[0].mac.agent.use_rnn is True:
            extra_returns.update({"hidden_states": []})
        if self.learners[0].mac.critic.use_rnn_critic is True:
            extra_returns.update({"hidden_states_critic": []})

        for agent_id in range(self.n_agents):

            assert self.learners[0].mac.agent.use_rnn == self.learners[agent_id].mac.agent.use_rnn
            assert self.learners[0].mac.critic.use_rnn_critic == self.learners[agent_id].mac.critic.use_rnn_critic

            agent_action, agent_extra_returns = (
                self.learners[agent_id].mac.select_actions(
                    ep_batch,
                    t_ep,
                    t_env,
                    bs=bs,
                    test_mode=test_mode,
                    agent_id=agent_id
                )
            )
            actions.append(agent_action)
            extra_returns["log_probs"].append(agent_extra_returns["log_probs"])
            extra_returns["values"].append(agent_extra_returns["values"])
            if self.learners[agent_id].mac.agent.use_rnn is True:
                extra_returns["hidden_states"].append(agent_extra_returns["hidden_states"])
            if self.learners[agent_id].mac.critic.use_rnn_critic is True:
                extra_returns["hidden_states_critic"].append(agent_extra_returns["hidden_states_critic"])

        actions = th.concat(actions, dim=1)
        extra_returns["log_probs"] = th.concat(extra_returns["log_probs"], dim=1)
        extra_returns["values"] = th.concat(extra_returns["values"], dim=1)
        assert (actions.shape == extra_returns["log_probs"].shape == extra_returns["values"].shape), (
            f"'actions.shape': {actions.shape}, "
            f"\n'extra_returns['log_probs'].shape': {extra_returns['log_probs'].shape}"
            f"\n'extra_returns['values'].shape': {extra_returns['values'].shape}"
        )

        if self.learners[0].mac.agent.use_rnn is True:
            extra_returns["hidden_states"] = th.concat(extra_returns["hidden_states"], dim=1)
        if self.learners[0].mac.critic.use_rnn_critic is True:
            extra_returns["hidden_states_critic"] = th.concat(extra_returns["hidden_states_critic"], dim=1)

        return actions, extra_returns

    def init_hidden(self, batch_size):

        for agent_id in range(self.n_agents):

            assert self.learners[0].mac.agent.use_rnn == self.learners[agent_id].mac.agent.use_rnn
            assert self.learners[0].mac.critic.use_rnn_critic == self.learners[agent_id].mac.critic.use_rnn_critic

            if self.learners[agent_id].mac.agent.use_rnn is True:
                self.learners[agent_id].mac.hidden_states = (
                    self.learners[agent_id].mac.agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)
                )  # shape: [batch_size, agents, hidden_dim]
            if self.learners[agent_id].critic.use_rnn_critic is True:
                self.learners[agent_id].critic.init_hidden(batch_size)

    def get_hidden_states(self):
        hidden_states_dict = {"hidden_states": [], "hidden_states_critic": []}
        for agent_id in range(self.n_agents):

            assert self.learners[0].mac.agent.use_rnn == self.learners[agent_id].mac.agent.use_rnn
            assert self.learners[0].mac.critic.use_rnn_critic == self.learners[agent_id].mac.critic.use_rnn_critic

            if self.learners[agent_id].mac.agent.use_rnn is True:
                hidden_states_dict["hidden_states"].append(self.learners[agent_id].mac.hidden_states.clone().detach())
            if self.learners[agent_id].critic.use_rnn_critic is True:
                hidden_states_dict["hidden_states_critic"].append(self.learners[agent_id].critic.hidden_states)

        if self.learners[0].mac.agent.use_rnn is True:
            hidden_states_dict["hidden_states"] = th.concat(hidden_states_dict["hidden_states"], dim=1)
        if self.learners[0].mac.critic.use_rnn_critic is True:
            hidden_states_dict["hidden_states_critic"] = th.concat(hidden_states_dict["hidden_states_critic"], dim=1)

        return hidden_states_dict

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        bs = batch.batch_size
        max_t = batch.max_seq_length
        factor = th.ones((bs, max_t-1, 1), device=self.device)

        for agent_id in th.randperm(self.n_agents):
            old_log_pi_taken, _, _ = self.learners[agent_id].evaluate_actions(batch)
            self.learners[agent_id].train(batch, t_env, factor)
            log_pi_taken, _, _ = self.learners[agent_id].evaluate_actions(batch)

            factor = factor * th.prod(th.exp(log_pi_taken - old_log_pi_taken), dim=-1, keepdim=True)

    def cuda(self):
        [learner.cuda() for learner in self.learners]

    def save_models(self, path):
        [learner.save_models(path) for learner in self.learners]

    def load_models(self, path):
        [learner.load_models(path) for learner in self.learners]


class HAPPO:
    def __init__(self, mac, agent_id, scheme, logger, args):
        self.args = args
        self.agent_id = agent_id
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.device = "cuda" if args.use_cuda else "cpu"
        assert self.args.standardise_returns is False
        assert self.args.standardise_rewards is False
        assert self.args.obs_last_action is False
        assert self.args.use_rnn == self.args.use_rnn_critic

        # Edit mac
        self.mac = mac
        self.mac.n_agents = 1
        input_shape = self.mac._get_input_shape(scheme)
        self.mac._build_agents(input_shape)

        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.mac.critic = self.critic

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.use_huber_loss = args.use_huber_loss
        self.huber_delta = args.huber_delta
        self.value_normalizer = PopArt(1, device=self.device)
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.data_chunk_length = args.data_chunk_length
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef

    def compute_returns(self, batch):
        """
        Use GAE and value normalizer
        """

        bs = batch.batch_size
        max_t = batch.max_seq_length-1
        returns = th.zeros((bs, max_t, 1), dtype=th.float32, device=self.device)
        gae = 0

        rewards = batch["reward"][:, :-1]
        # We use only the 'filled' and not the 'terminated' since the first will still
        # be True at the last step (and then will be False), while the 'terminated' will
        # be False at the last step onwards.
        mask = batch["filled"][:, :-1].float()
        values = batch["values"][:, :, self.agent_id]

        for step in reversed(range(rewards.shape[1])):
            delta = (
                rewards[:, step]
                + self.gamma * self.value_normalizer.denormalize(values[:, step + 1]) * mask[:, step]
                - self.value_normalizer.denormalize(values[:, step])
            )
            gae = delta + self.gamma * self.gae_lambda * mask[:, step] * gae
            returns[:, step] = gae + self.value_normalizer.denormalize(values[:, step])

        return returns

    def compute_advantages(self, batch, returns):
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        values = batch["values"]
        advantages = returns - self.value_normalizer.denormalize(values[:, :-1, self.agent_id])
        advantages_copy = advantages.clone().detach().cpu().numpy()
        advantages_copy[mask.detach().cpu().numpy() == 0.0] = np.nan
        mean_advantages = th.from_numpy(np.nanmean(advantages_copy, keepdims=True)).to(self.device)
        std_advantages = th.from_numpy(np.nanstd(advantages_copy, keepdims=True)).to(self.device)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        return advantages

    def evaluate_actions(self, batch, during_training=False):

        if during_training is False:

            agent_id = self.agent_id

            # Get the mask
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

            hidden_states = None
            if self.mac.agent.use_rnn is True:
                # Get the hidden states only of the first step (which are 0s)
                hidden_states = batch["hidden_states"][:, 0:1, agent_id]

            # Get the action executed
            actions = batch["actions"][:, :-1]

            mac_out = self.mac.forward(
                batch,
                t=batch.max_seq_length-1,
                hidden_states=hidden_states,
                masks=mask,
                agent_id=agent_id,
                batch_inf=True
            )
        else:
            agent_id = 0

            mask = batch["mask"]
            actions = batch["actions"]
            hidden_states = None
            if self.mac.agent.use_rnn is True:
                hidden_states = batch["hidden_states"]

            mac_out = self.mac.forward(
                batch,
                # zero because we create a minibatch with only one timestep
                # in the 'create_mini_batch' function
                t=0,
                hidden_states=hidden_states,
                masks=mask,
                # zero because we have already kept
                # the right agent in the 'create_mini_batch' function
                agent_id=agent_id,
                batch_inf=False
            ).unsqueeze(2)

        pi = mac_out
        pi = pi[:, :, 0, :].unsqueeze(2)

        # Calculate policy grad with mask
        pi = th.where(mask.unsqueeze(-1).expand_as(pi) == 0, th.ones_like(pi), pi)

        pi_taken = th.gather(
            pi,
            dim=3,
            index=actions[:, :, agent_id, :].unsqueeze(2)
        ).squeeze(3)
        log_pi_taken = th.log(pi_taken + 1e-10)
        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

        return log_pi_taken, entropy, pi

    def train(self, batch, t_env, factor):

        # Calculate GAE returns
        returns = self.compute_returns(batch)

        # Calculate advantages based on the GAE returns
        advantages = self.compute_advantages(batch, returns)

        # Initialize stats
        actor_train_stats = None
        critic_train_stats = None

        for _ in range(self.args.epochs):

            batch_size = batch["batch_size"] * (batch["max_seq_length"]-1)
            data_chunks = batch_size // self.data_chunk_length
            if self.mac.agent.use_rnn is True:
                mini_batch_size = data_chunks // self.num_mini_batch
                rand = th.randperm(data_chunks).numpy()
            else:
                mini_batch_size = batch_size // self.num_mini_batch
                rand = th.randperm(batch_size).numpy()

            sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(self.num_mini_batch)]

            for indices in sampler:

                mini_batch = self.create_mini_batch(batch, returns, advantages, factor, indices, mini_batch_size)

                # Train Actor
                actor_train_stats = self.train_actor(mini_batch, actor_train_stats)

                # Train Critic
                critic_train_stats = self.train_critic(mini_batch, critic_train_stats)

            self.training_steps += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in critic_train_stats.keys():
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)
            ts_logged = len(actor_train_stats["pg_loss"])
            for key in actor_train_stats.keys():
                self.logger.log_stat(key, sum(actor_train_stats[key]) / ts_logged, t_env)

    def train_actor(self, batch, running_log):

        if running_log is None:
            running_log = {
                "advantage_mean": [],
                "pg_loss": [],
                "agent_grad_norm": [],
                "pi_max": [],
                "entropy": [],
                "ratio": []
            }

        mask = batch["mask"]
        adv_targ = batch["advantages"]
        factor = batch["factor"]
        old_log_pi_taken = batch["log_probs"].squeeze(3)

        # Actor loss
        log_pi_taken, entropy, pi = self.evaluate_actions(batch, during_training=True)
        ratios = th.prod(th.exp(log_pi_taken - old_log_pi_taken.detach()), dim=-1, keepdim=True)
        surr1 = ratios * adv_targ
        surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * adv_targ
        pg_loss = -(
                th.sum(
                    factor.detach() * th.min(surr1, surr2),
                    dim=-1,
                    keepdim=True
                ) * mask
        ).sum() / mask.sum()
        entropy = entropy.mean()
        pg_loss -= self.args.entropy_coef * entropy

        # Optimise Actor
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        running_log["advantage_mean"].append((adv_targ * mask).sum().item() / mask.sum().item())
        running_log["pg_loss"].append(pg_loss.item())
        running_log["agent_grad_norm"].append(grad_norm.item())
        running_log["pi_max"].append((pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item())
        running_log["entropy"].append(entropy.item())
        running_log["ratio"].append(ratios.mean().item())

        return running_log

    def train_critic(self, batch, running_log):

        if running_log is None:
            running_log = {
                "critic_loss": [],
                "critic_grad_norm": [],
                "td_error_abs": [],
                "returns_mean": [],
                "q_taken_mean": [],
            }

        returns = batch["returns"].squeeze(2)
        old_values = batch["values"].squeeze(2).squeeze(2)
        mask = batch["mask"].squeeze(2)
        hidden_states = None
        if self.mac.critic.use_rnn_critic is True:
            hidden_states = batch["hidden_states_critic"]

        critic_returns = self.critic(
            batch,
            t=0,
            hidden_states=hidden_states,
            masks=mask
        )
        values = critic_returns[0]
        value_pred_clipped = old_values + \
                             (values - old_values).clamp(-self.args.eps_clip, self.args.eps_clip)
        error_clipped = self.value_normalizer(returns) - value_pred_clipped
        error_original = self.value_normalizer(returns) - values

        # Compute loss
        if self.use_huber_loss is True:
            value_loss_clipped = self.huber_loss(error_clipped)
            value_loss_original = self.huber_loss(error_original)
        else:
            value_loss_clipped = self.mse_loss(error_clipped)
            value_loss_original = self.mse_loss(error_original)

        # Clip value loss
        value_loss = th.max(value_loss_original, value_loss_clipped)

        # Take mean over batch
        value_loss = value_loss.mean() * self.value_loss_coef

        # Optimise Critic
        self.critic_optimiser.zero_grad()
        value_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(value_loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        running_log["q_taken_mean"].append(values.mean().item())
        running_log["returns_mean"].append(returns.mean().item())

        return running_log

    def huber_loss(self, e):
        a = (abs(e) <= self.huber_delta).float()
        b = (e > self.huber_delta).float()
        return a * e ** 2 / 2 + b * self.huber_delta * (abs(e) - self.huber_delta / 2)

    def mse_loss(self, e):
        return e ** 2 / 2

    def create_mini_batch(self, batch, returns, advantages, factor, indices, mini_batch_size):

        max_ts = batch["max_seq_length"]-1

        obs = batch["obs"][:, :max_ts, self.agent_id].unsqueeze(2)
        obs = obs.transpose(1, 0).reshape(-1, 1, *obs.shape[2:])
        state = batch["state"][:, :max_ts]
        state = state.transpose(1, 0).reshape(-1, 1, *state.shape[2:])
        actions = batch["actions"][:, :max_ts, self.agent_id].unsqueeze(2)
        actions = actions.transpose(1, 0).reshape(-1, 1, *actions.shape[2:])
        actions_onehot = batch["actions_onehot"][:, :max_ts, self.agent_id].unsqueeze(2)
        actions_onehot = actions_onehot.transpose(1, 0).reshape(-1, 1, *actions_onehot.shape[2:])
        avail_actions = batch["avail_actions"][:, :max_ts, self.agent_id].unsqueeze(2)
        avail_actions = avail_actions.transpose(1, 0).reshape(-1, 1, *avail_actions.shape[2:])
        reward = batch["reward"][:, :max_ts]
        reward = reward.transpose(1, 0).reshape(-1, 1, *reward.shape[2:])
        terminated = batch["terminated"][:, :max_ts].float()
        terminated = terminated.transpose(1, 0).reshape(-1, 1, *terminated.shape[2:])
        filled = batch["filled"][:, :max_ts].float()
        filled = filled.transpose(1, 0).reshape(-1, 1, *filled.shape[2:])
        mask = batch["filled"][:, :max_ts].float()
        mask[:, 1:] = mask[:, 1:] * (1 - batch["terminated"][:, :max_ts].float()[:, :-1])
        mask = mask.transpose(1, 0).reshape(-1, 1, *mask.shape[2:])
        log_probs = batch["log_probs"][:, :max_ts, self.agent_id].unsqueeze(2)
        log_probs = log_probs.transpose(1, 0).reshape(-1, 1, *log_probs.shape[2:])
        values = batch["values"][:, :max_ts, self.agent_id].unsqueeze(2)
        values = values.transpose(1, 0).reshape(-1, 1, *values.shape[2:])
        returns = returns.transpose(1, 0).reshape(-1, 1, *returns.shape[2:])
        advantages = advantages.transpose(1, 0).reshape(-1, 1, *advantages.shape[2:])
        factor = factor.transpose(1, 0).reshape(-1, 1, *factor.shape[2:])

        mini_batch = {
            "obs": [],
            "state": [],
            "actions": [],
            "actions_onehot": [],
            "avail_actions": [],
            "reward": [],
            "terminated": [],
            "filled": [],
            "mask": [],
            "log_probs": [],
            "values": [],
            "returns": [],
            "advantages": [],
            "factor": [],
            "max_seq_length": 1
        }
        if self.mac.agent.use_rnn is True:
            mini_batch["batch_size"] = mini_batch_size * self.data_chunk_length
            hidden_states = batch["hidden_states"][:, :, self.agent_id].unsqueeze(2)
            hidden_states = hidden_states.transpose(1, 0).reshape(-1, 1, *hidden_states.shape[2:])
            hidden_states_critic = batch["hidden_states_critic"][:, :, self.agent_id].unsqueeze(2)
            hidden_states_critic = hidden_states_critic.transpose(1, 0).reshape(-1, 1, *hidden_states_critic.shape[2:])
            mini_batch.update({"hidden_states": [], "hidden_states_critic": []})
            for index in indices:
                ind = index * self.data_chunk_length
                mini_batch["obs"].append(obs[ind:ind+self.data_chunk_length])
                mini_batch["state"].append(state[ind:ind + self.data_chunk_length])
                mini_batch["actions"].append(actions[ind:ind + self.data_chunk_length])
                mini_batch["actions_onehot"].append(actions_onehot[ind:ind + self.data_chunk_length])
                mini_batch["avail_actions"].append(avail_actions[ind:ind + self.data_chunk_length])
                mini_batch["reward"].append(reward[ind:ind + self.data_chunk_length])
                mini_batch["terminated"].append(terminated[ind:ind + self.data_chunk_length])
                mini_batch["filled"].append(filled[ind:ind + self.data_chunk_length])
                mini_batch["mask"].append(mask[ind:ind + self.data_chunk_length])
                mini_batch["log_probs"].append(log_probs[ind:ind + self.data_chunk_length])
                mini_batch["values"].append(values[ind:ind + self.data_chunk_length])
                mini_batch["returns"].append(returns[ind:ind + self.data_chunk_length])
                mini_batch["advantages"].append(advantages[ind:ind + self.data_chunk_length])
                mini_batch["factor"].append(factor[ind:ind + self.data_chunk_length])
                mini_batch["hidden_states"].append(hidden_states[ind])
                mini_batch["hidden_states_critic"].append(hidden_states_critic[ind])

            def _flatten(x):
                return x.transpose(0, 1).reshape(self.data_chunk_length * mini_batch_size, *x.shape[2:])

            mini_batch["obs"] = _flatten(th.stack(mini_batch["obs"]))
            mini_batch["state"] = _flatten(th.stack(mini_batch["state"]))
            mini_batch["actions"] = _flatten(th.stack(mini_batch["actions"]))
            mini_batch["actions_onehot"] = _flatten(th.stack(mini_batch["actions_onehot"]))
            mini_batch["avail_actions"] = _flatten(th.stack(mini_batch["avail_actions"]))
            mini_batch["reward"] = _flatten(th.stack(mini_batch["reward"]))
            mini_batch["terminated"] = _flatten(th.stack(mini_batch["terminated"]))
            mini_batch["filled"] = _flatten(th.stack(mini_batch["filled"]))
            mini_batch["mask"] = _flatten(th.stack(mini_batch["mask"]))
            mini_batch["log_probs"] = _flatten(th.stack(mini_batch["log_probs"]))
            mini_batch["values"] = _flatten(th.stack(mini_batch["values"]))
            mini_batch["returns"] = _flatten(th.stack(mini_batch["returns"]))
            mini_batch["advantages"] = _flatten(th.stack(mini_batch["advantages"]))
            mini_batch["factor"] = _flatten(th.stack(mini_batch["factor"]))
            mini_batch["hidden_states"] = th.stack(mini_batch["hidden_states"])
            mini_batch["hidden_states"] = mini_batch["hidden_states"].\
                reshape(mini_batch_size, *mini_batch["hidden_states"].shape[2:])
            mini_batch["hidden_states_critic"] = th.stack(mini_batch["hidden_states_critic"])
            mini_batch["hidden_states_critic"] = mini_batch["hidden_states_critic"].\
                reshape(mini_batch_size, *mini_batch["hidden_states_critic"].shape[2:])

        else:
            mini_batch["batch_size"] = mini_batch_size
            mini_batch["obs"] = obs[indices]
            mini_batch["state"] = state[indices]
            mini_batch["actions"] = actions[indices]
            mini_batch["actions_onehot"] = actions_onehot[indices]
            mini_batch["avail_actions"] = avail_actions[indices]
            mini_batch["reward"] = reward[indices]
            mini_batch["terminated"] = terminated[indices]
            mini_batch["filled"] = filled[indices]
            mini_batch["mask"] = mask[indices]
            mini_batch["log_probs"] = log_probs[indices]
            mini_batch["values"] = values[indices]
            mini_batch["returns"] = returns[indices]
            mini_batch["advantages"] = advantages[indices]
            mini_batch["factor"] = factor[indices]

        return mini_batch

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        if not os.path.exists(path+'/agent_'+str(self.agent_id)):
            os.mkdir(path+'/agent_'+str(self.agent_id))
        self.mac.save_models(path+'/agent_'+str(self.agent_id))
        th.save(self.critic.state_dict(), "{}/agent_{}/critic.th".format(path, str(self.agent_id)))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_{}/agent_opt.th".format(path, str(self.agent_id)))
        th.save(self.critic_optimiser.state_dict(), "{}/agent_{}/critic_opt.th".format(path, str(self.agent_id)))

    def load_models(self, path):
        self.mac.load_models(path+'/agent_'+str(self.agent_id))
        self.critic.load_state_dict(
            th.load("{}/agent_{}/critic.th".format(path, str(self.agent_id)),
                    map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_{}/agent_opt.th".format(path, str(self.agent_id)),
                    map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/agent_{}/critic_opt.th".format(path, str(self.agent_id)),
                    map_location=lambda storage, loc: storage))

from .basic_controller import BasicMAC
import torch as th


# This multi-agent controller shares parameters between agents
class happoMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, agent_id=None):

        extra_returns = {}

        if agent_id is None:
            return self.learner.select_actions(ep_batch, t_ep, t_env, bs, test_mode)

        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep, agent_id].unsqueeze(2)

        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, agent_id=agent_id)

        select_actions_returns = self.action_selector.select_action(
            agent_outputs[bs],
            avail_actions[bs],
            t_env,
            test_mode=test_mode
        )
        # Get actions, log probs, and hidden states
        chosen_actions, chosen_log_probs = select_actions_returns
        if self.agent.use_rnn is True:
            extra_returns.update({'hidden_states': self.hidden_states.clone().detach()})

        # Get values and critic hidden states
        critic_returns = self.critic(ep_batch, t_ep, hidden_states=self.critic.hidden_states)
        values = critic_returns[0]
        if self.critic.use_rnn_critic is True:
            self.critic.hidden_states = critic_returns[1]
            extra_returns.update({'hidden_states_critic': self.critic.hidden_states.clone().detach()})

        extra_returns.update({'log_probs': chosen_log_probs.clone().detach(), 'values': values[bs].clone().detach()})

        return chosen_actions, extra_returns

    def forward(self, ep_batch, t, test_mode=False, batch_inf=False, masks=None, agent_id=None, hidden_states=None):

        if batch_inf is True:
            epi_len = t if batch_inf else 1

        agent_inputs = self._build_inputs(ep_batch, t, batch_inf, agent_id)

        if batch_inf is False:
            avail_actions = ep_batch["avail_actions"][:, t, agent_id].unsqueeze(2)
        else:
            avail_actions = ep_batch["avail_actions"][:, :t, agent_id].unsqueeze(2)

        # Create masks
        if masks is None:
            if batch_inf is False:
                masks = 1 - ep_batch["terminated"][:, t]
                assert len(masks.shape) == 2 and masks.shape[1] == 1, f"masks: {masks}"
                masks = masks.repeat(1, self.n_agents).view(-1, 1)
            else:
                raise NotImplementedError

        # Get hidden states
        if hidden_states is None:
            hidden_states = self.hidden_states

        agent_outs, self.hidden_states = self.agent(agent_inputs, hidden_states, masks)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if self.mask_before_softmax is True:
                if batch_inf is False:
                    reshaped_avail_actions = avail_actions.reshape(ep_batch["batch_size"] * self.n_agents,
                                                                   -1)
                else:
                    reshaped_avail_actions = avail_actions.transpose(1, 2).reshape(
                        ep_batch["batch_size"] * self.n_agents, epi_len, -1
                    )
                # Make the logits for unavailable actions very negative to minimise their effect on the softmax
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        if batch_inf is False:
            return agent_outs.view(ep_batch["batch_size"], self.n_agents, -1)
        else:
            return agent_outs.view(ep_batch["batch_size"], self.n_agents, epi_len, -1).transpose(1, 2)

    def init_hidden(self, batch_size):
        self.learner.init_hidden(batch_size)
        return

    def _build_inputs(self, batch, t, batch_inf, agent_id):

        # Assumes homogenous agents.
        # Other MACs might want to, e.g., delegate building inputs to each agent
        bs = batch["batch_size"]

        # Keep only the values corresponding to agent_id
        if agent_id is not None:
            batch_obs = batch["obs"][:, :, agent_id].unsqueeze(2)
            batch_actions_onehot = batch["actions_onehot"][:, :, agent_id].unsqueeze(2)
        else:
            batch_obs = batch["obs"]
            batch_actions_onehot = batch["actions_onehot"]

        if batch_inf is False:

            inputs = [batch_obs[:, t]]

            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch_actions_onehot[:, t]))
                else:
                    inputs.append(batch_actions_onehot[:, t-1])
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

            if self.is_image is False:
                inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
            else:
                img_ch, img_h, img_w = inputs[0].shape[2:]
                inputs = [inputs[0].reshape(bs * self.n_agents, img_ch, img_h, img_w),
                          [] if len(inputs) == 1
                             else
                          th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs[1:]], dim=1)
                          ]
        else:

            inputs = [batch_obs[:, :t]]

            if self.args.obs_last_action:
                last_actions = th.zeros_like(batch_actions_onehot[:, :t])
                last_actions[:, 1:] = batch_actions_onehot[:, :t-1]
                inputs.append(last_actions)
            if self.args.obs_agent_id:
                inputs.append(
                    th.eye(self.n_agents, device=batch.device).
                    view(1, 1, self.n_agents, self.n_agents).
                    expand(bs, t, -1, -1)
                )

            inputs = th.cat([x.transpose(1, 2).reshape(bs * self.n_agents, t, -1) for x in inputs], dim=2)

        return inputs

    def get_hidden_states(self):
        return self.learner.get_hidden_states()


import torch as th
from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class emcMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

        self.use_individual_Q = self.args.use_individual_Q

        assert self.is_image is False, "EMC does not support image obs for the time being!"

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):

        extra_returns = {}

        avail_actions = ep_batch["avail_actions"][:, t_ep]

        if self.use_individual_Q is True:
            agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        # Only select actions for the selected batch elements in bs
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs],
            avail_actions[bs],
            t_env,
            test_mode=test_mode
        )

        return chosen_actions, extra_returns

    def forward(self, ep_batch, t, test_mode=False, batch_inf=False):

        epi_len = t if batch_inf else 1
        agent_inputs = self._build_inputs(ep_batch, t, batch_inf)

        avail_actions = (
                            ep_batch["avail_actions"][:, :t] if batch_inf is True
                                                             else
                            ep_batch["avail_actions"][:, t:t+1]
        )

        if self.use_individual_Q is True:
            agent_outs, self.hidden_states, individual_Q = self.agent(agent_inputs, self.hidden_states)
        else:
           agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if self.mask_before_softmax is True:
                # Make the logits for unavailable actions very negative to minimise their effect on the softmax
                reshaped_avail_actions = avail_actions.transpose(1, 2).reshape(
                    ep_batch.batch_size * self.n_agents,
                    epi_len,
                    -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if self.mask_before_softmax is True:
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=-1, keepdim=True).float()

                agent_outs = (
                        (1 - self.action_selector.epsilon) * agent_outs
                        + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num
                )

                if self.mask_before_softmax is True:
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        if self.use_individual_Q is True:
            return (
                agent_outs.view(ep_batch.batch_size, self.n_agents, -1),
                individual_Q.view(ep_batch.batch_size, self.n_agents, -1)
            )
        else:
            if batch_inf is False:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
            else:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, epi_len, -1).transpose(1, 2)

    def init_hidden(self, batch_size):
        if self.args.agent == "rnn_emc":
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        else:
            raise ValueError(f"self.args.agent: {self.args.agent}")

    def _build_inputs(self, batch, t, batch_inf):

        # Assumes homogenous agents.
        # Other MACs might want to, e.g., delegate building inputs to each agent
        bs = batch.batch_size

        if batch_inf is False:

            inputs = [batch["obs"][:, t]]

            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                else:
                    inputs.append(batch["actions_onehot"][:, t-1])
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

            if self.is_image is False:
                inputs = th.cat([x.reshape(bs * self.n_agents, 1, -1) for x in inputs], dim=2)
            else:
                img_ch, img_h, img_w = inputs[0].shape[2:]
                inputs = [inputs[0].reshape(bs * self.n_agents, img_ch, img_h, img_w),
                          [] if len(inputs) == 1
                             else
                          th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs[1:]], dim=1)
                          ]
        else:

            inputs = [batch["obs"][:, :t]]

            if self.args.obs_last_action:
                last_actions = th.zeros_like(batch["actions_onehot"][:, :t])
                last_actions[:, 1:] = batch["actions_onehot"][:, :t-1]
                inputs.append(last_actions)
            if self.args.obs_agent_id:
                inputs.append(
                    th.eye(self.n_agents, device=batch.device).
                    view(1, 1, self.n_agents, self.n_agents).
                    expand(bs, t, -1, -1)
                )

            inputs = th.cat([x.transpose(1, 2).reshape(bs*self.n_agents, t, -1) for x in inputs], dim=2)

        return inputs

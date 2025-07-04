from .basic_controller import BasicMAC
import torch as th


# This multi-agent controller shares parameters between agents
class cdsMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

        assert self.is_image is False, "CDS does not support image obs for the time being!"

    def forward(self, ep_batch, t, test_mode=False):

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, local_q = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if self.mask_before_softmax is True:
                # Make the logits for unavailable actions very negative to minimise their effect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)

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

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if self.args.agent == "rnn_cds":
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        else:
            raise ValueError(f"self.args.agent: {self.args.agent}")


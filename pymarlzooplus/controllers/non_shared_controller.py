import torch as th

from pymarlzooplus.modules.agents import REGISTRY as agent_REGISTRY
from pymarlzooplus.components.action_selectors import REGISTRY as action_REGISTRY


# This multi-agent controller does not share parameters between agents
class NonSharedMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.is_image = False  # Image input
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):

        # Just for compatibility
        extra_returns = {}

        avail_actions = ep_batch["avail_actions"][:, t_ep]

        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        # Only select actions for the selected batch elements in bs
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs],
            avail_actions[bs],
            t_env,
            test_mode=test_mode
        )

        return chosen_actions, extra_returns

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their effect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to, e.g., delegate building inputs to each agent
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        if self.is_image is False:
            inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        else:
            img_ch, img_h, img_w = inputs[0].shape[2:]
            inputs = [
                inputs[0].reshape(bs * self.n_agents, img_ch, img_h, img_w),
                [] if len(inputs) == 1
                   else
                th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs[1:]], dim=1)
            ]

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if isinstance(input_shape, int):  # vector input
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0]
            if self.args.obs_agent_id:
                input_shape += self.n_agents
        elif isinstance(input_shape, tuple):  # image input
            self.is_image = True
            input_shape = [input_shape, 0]
            if self.args.obs_last_action:
                input_shape[1] += scheme["actions_onehot"]["vshape"][0]
            if self.args.obs_agent_id:
                input_shape[1] += self.n_agents
            input_shape = tuple(input_shape)  # list to tuple

        return input_shape

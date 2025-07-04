# code adapted from https://github.com/oxwhirl/facmac/
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pymarlzooplus.modules.critics.mlp import MLP


class MADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.is_image = False  # Image input
        self.input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        self.critic = MLP(self.input_shape, self.args, 1)

    def forward(self, inputs, actions):

        if self.is_image is False:  # Vector observation
            # action concat
            inputs = th.cat((inputs, actions), dim=-1)
            # get q-values
            q = self.critic(inputs)
        else:  # Image observation
            # action concat
            extra_input_index = 2 if self.args.obs_individual_obs else 1
            if len(inputs) > extra_input_index:  # There are already some extra inputs, so concatenate them
                inputs[extra_input_index] = th.cat([inputs[extra_input_index], actions], dim=-1)
            else:  # There is no extra input to concatenate with, so add the actions as a new input
                inputs.append(actions)
            # get q-values
            q = self.critic(inputs)

        return q

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        if isinstance(input_shape, int):  # vector input
            # observation
            if self.args.obs_individual_obs:
                input_shape += scheme["obs"]["vshape"]
            # actions
            input_shape += self.n_actions * self.n_agents
            # last actions
            if self.args.obs_last_action:
                input_shape += self.n_actions
            # agent id
            if self.args.obs_agent_id:
                input_shape += self.n_agents
        elif isinstance(input_shape, tuple):  # image input
            self.is_image = True
            # state
            input_shape = [input_shape, (), 0]  # state, individual obs, actions / last actions / agent id
            # observations
            if self.args.obs_individual_obs:
                input_shape[1] = scheme["obs"]["vshape"]
                assert input_shape[0][1:] == input_shape[1], f"Image input shape mismatch: {input_shape}"
            # actions
            input_shape[2] += self.n_actions * self.n_agents
            # last actions
            if self.args.obs_last_action:
                input_shape[2] += self.n_actions
            # agent id
            if self.args.obs_agent_id:
                input_shape[2] += self.n_agents
            input_shape = tuple(input_shape)  # list to tuple

        return input_shape

# code adapted from https://github.com/oxwhirl/facmac/
import torch as th
import torch.nn as nn

from pymarlzooplus.modules.critics.mlp import MLP


class MADDPGCriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCriticNS, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.is_image = False  # Image input
        self.input_shape = self._get_input_shape(scheme)
        self.output_type = "q"
        self.critics = [MLP(self.input_shape, self.args, 1) for _ in range(self.n_agents)]

        assert self.args.obs_agent_id is False, "MADDPG-NS does not support 'obs_agent_id'"

    def forward(self, inputs, actions):

        qs = []
        for i in range(self.n_agents):

            if self.is_image is False:  # Vector observation
                # action concat
                agent_inputs = th.cat((inputs[:, :, i], actions[:, :, i]), dim=-1)
                # get q-values
                q = self.critics[i](agent_inputs).unsqueeze(2)
            else:  # Image observation
                bs, max_t, *_ = inputs[0].shape
                # state
                agent_inputs = [inputs[0]]
                # observation
                if self.args.obs_individual_obs is True:
                    agent_inputs.append(inputs[1][:, :, i].unsqueeze(2))
                # action concat
                extra_input_index = 2 if self.args.obs_individual_obs else 1
                if len(inputs) > extra_input_index:  # There are already some extra inputs, so concatenate them
                    agent_inputs.append(
                        th.cat(
                            [inputs[extra_input_index][:, :, i].unsqueeze(2), actions[:, :, i].unsqueeze(2)],
                            dim=-1
                        )
                    )
                else:  # There is no extra input to concatenate with, so add the actions as a new input
                    agent_inputs.append(actions[:, :, i].unsqueeze(2))
                # get q-values
                q = self.critics[i](agent_inputs).view(bs, max_t, 1, 1)

            qs.append(q)

        return th.cat(qs, dim=2)

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
                self.input_shape += self.n_actions
        elif isinstance(input_shape, tuple):  # image input
            self.is_image = True
            # state: Change the number of agents to 1 for compatibility with the way that CNN infer the input shape
            input_shape = list(input_shape)  # tuple to list
            input_shape[0] = 1
            input_shape = tuple(input_shape)  # list to tuple
            input_shape = [input_shape, (), 0]  # state, individual obs, actions / last actions
            # observations
            if self.args.obs_individual_obs:
                input_shape[1] = scheme["obs"]["vshape"]
                assert input_shape[0][1:] == input_shape[1], f"Image input shape mismatch: {input_shape}"
            # actions
            input_shape[2] += self.n_actions * self.n_agents
            # last actions
            if self.args.obs_last_action:
                input_shape[2] += self.n_actions
            input_shape = tuple(input_shape)  # list to tuple

        return input_shape

    def parameters(self):
        params = list(self.critics[0].parameters())
        for i in range(1, self.n_agents):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, c in enumerate(self.critics):
            c.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()

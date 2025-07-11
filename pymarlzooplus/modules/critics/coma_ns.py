import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pymarlzooplus.modules.critics.mlp import MLP


class COMACriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(COMACriticNS, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.is_image = False  # Image input
        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.critics = [MLP(input_shape, args, self.n_actions) for _ in range(self.n_agents)]

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n_agents):

            if self.is_image is False:  # Vector observation
                q = self.critics[i](inputs[:, :, i]).unsqueeze(2)
            else:  # Image observation
                bs, max_t, *_ = inputs[0].shape
                # state
                agent_inputs = [inputs[0]]
                # observation / actions / last actions
                for x_input in inputs[1:]:
                    agent_inputs.append(x_input[:, :, i].unsqueeze(2))
                q = self.critics[i](agent_inputs).view(bs, max_t, 1, self.n_actions)
            qs.append(q)

        return th.cat(qs, dim=2)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        # state
        if self.is_image is False:  # Vector observation
            inputs = [batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1)]
        else:  # Image observation
            inputs = [batch["state"][:, ts]]

        # observation
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(
                    th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                )
            elif isinstance(t, int):
                inputs.append(
                    batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                )
            else:
                last_actions = th.cat(
                    [th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1
                )
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        if self.is_image is False:  # Vector observation
            inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        elif self.is_image is True and self.args.obs_last_action is True:  # Image observation
            action_index = 2 if self.args.obs_individual_obs else 1
            inputs[action_index] = th.cat(
                [x.reshape(bs, max_t, self.n_agents, -1) for x in inputs[action_index:]], dim=-1
            )
            if len(inputs) == action_index + 2:
                del inputs[action_index + 1]  # remove the last input since it is already concatenated

        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        if isinstance(input_shape, int):  # vector input
            # observation
            if self.args.obs_individual_obs:
                input_shape += scheme["obs"]["vshape"]
            # actions
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            # last action
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
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
            input_shape[2] += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            # last actions
            if self.args.obs_last_action:
                input_shape[2] += scheme["actions_onehot"]["vshape"][0] * self.n_agents
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
        for i, a in enumerate(self.critics):
            a.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()

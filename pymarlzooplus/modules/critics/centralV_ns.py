# code adapted from https://github.com/AnujMahajanOxf/MAVEN

import torch as th
import torch.nn as nn

from pymarlzooplus.modules.critics.mlp import MLP


class CentralVCriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCriticNS, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.is_image = False  # Image input
        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.critics = [MLP(input_shape, args, 1) for _ in range(self.n_agents)]

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n_agents):
            q = self.critics[i](inputs)
            qs.append(q.view(bs, max_t, 1, -1))
        q = th.cat(qs, dim=2)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        # state
        inputs = [batch["state"][:, ts]]

        # observations
        if self.args.obs_individual_obs:
            if self.is_image is False:  # Vector observation
                inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))
            else:  # Image observation
                inputs.append(batch["obs"][:, ts])  # shape: (bs, max_t, n_agents, ch, h, w)

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat(
                    [th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1
                )
                last_actions = last_actions.view(bs, max_t, 1, -1)
                inputs.append(last_actions)

        if self.is_image is False:  # Vector observation
            inputs = th.cat([x.reshape(bs * max_t, -1) for x in inputs], dim=1)
        else:  # Image observation
            # state
            img_ch, img_h, img_w = inputs[0].shape[3:]
            inputs[0] = inputs[0].reshape(bs * max_t * self.n_agents, img_ch, img_h, img_w)
            # observations
            if self.args.obs_individual_obs:
                inputs[1] = inputs[1].reshape(bs * max_t * self.n_agents, img_ch, img_h, img_w)
            # last actions
            if self.args.obs_last_action:
                last_action_index = 2 if self.args.obs_individual_obs else 1
                inputs[last_action_index] = inputs[last_action_index].reshape(bs * max_t, -1)

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        if isinstance(input_shape, int):  # vector input
            # observations
            if self.args.obs_individual_obs:
                input_shape += scheme["obs"]["vshape"]
            # last actions
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        elif isinstance(input_shape, tuple):  # image input
            self.is_image = True
            input_shape = [input_shape, (), 0]  # state, individual obs, last actions
            # observations
            if self.args.obs_individual_obs:
                input_shape[1] = scheme["obs"]["vshape"]
                assert input_shape[0][1:] == input_shape[1], f"Image input shape mismatch: {input_shape}"
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

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pymarlzooplus.utils.trainable_image_encoder import TrainableImageEncoder


class COMACritic(nn.Module):
    def __init__(self, scheme, args):
        super(COMACritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_dim = None
        self.is_image = False
        self.cnn_features_dim = args.cnn_features_dim

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"
        if isinstance(input_shape, tuple):  # Image input
            self.state_dim = self.cnn_features_dim * self.n_agents
            if self.args.obs_individual_obs:
                self.state_dim += self.cnn_features_dim
            self.state_dim += input_shape[2]
        else:  # Vector input
            self.state_dim = input_shape

        if self.is_image is True:
            self.cnn = TrainableImageEncoder([input_shape[0][1:]], args)

        # Set up network layers
        self.fc1 = nn.Linear(self.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.n_actions)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)

        if self.is_image is True:
            # States
            channels = inputs[0].shape[3]
            height = inputs[0].shape[4]
            width = inputs[0].shape[5]
            # Reshape the states
            # from [batch size, max steps, n_agents, channels, height, width]
            # to [batch size x max steps x n_agents, channels, height, width]
            inputs[0] = inputs[0].reshape(-1, channels, height, width)
            # CNN
            # from [batch size x max steps x n_agents, channels, height, width]
            # to [batch size x max steps x n_agents, cnn features dim]
            inputs[0] = self.cnn(inputs[0])
            # Reshape the shape of the encoded states
            # to [batch size x max steps, n_agents x cnn features dim]
            inputs[0] = inputs[0].view(bs * max_t, self.n_agents * self.cnn_features_dim)
            # to [batch size x max steps, n_agents, n_agents x cnn features dim]
            inputs[0] = inputs[0].unsqueeze(1).repeat(1, self.n_agents, 1)
            # to [batch size, max steps, n_agents, n_agents x cnn features dim]
            inputs[0] = inputs[0].view(bs, max_t, self.n_agents, self.n_agents * self.cnn_features_dim)

            # Individual observations
            if self.args.obs_individual_obs:
                # from [batch size, max steps, n_agents, channels, height, width]
                # to [batch size x max steps x n_agents, channels, height, width]
                inputs[1] = inputs[1].reshape(-1, channels, height, width)
                # CNN
                # from [batch size x max steps x n_agents, channels, height, width]
                # to [batch size x max steps x n_agents, cnn features dim]
                inputs[1] = self.cnn(inputs[1])
                # Reshape the shape of the state
                # to [batch size, max steps, n_agents, cnn features dim]
                inputs[1] = inputs[1].view(bs, max_t, self.n_agents, self.cnn_features_dim)

            # All inputs
            inputs = th.cat(inputs, dim=-1)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # state
        if self.is_image is False:
            inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        else:
            inputs.append(batch["state"][:, ts])

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
                    [th.zeros_like(batch["actions_onehot"][:, 0:1]),
                     batch["actions_onehot"][:, :-1]],
                    dim=1
                )
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
            )

        if self.is_image is False:
            inputs = th.cat(inputs, dim=-1)
        else:
            extra_feats_index = 2 if self.args.obs_individual_obs else 1
            if len(inputs) > extra_feats_index:
                inputs[extra_feats_index] = th.cat(inputs[extra_feats_index:], dim=-1)
                del inputs[extra_feats_index + 1:]  # Delete the extra inputs since they have already been concatenated

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        if isinstance(input_shape, int):
            # observation
            if self.args.obs_individual_obs:
                input_shape += scheme["obs"]["vshape"]
            # actions
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            # last action
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            # agent id
            if self.args.obs_agent_id:
                input_shape += self.n_agents
        elif isinstance(input_shape, tuple):
            self.is_image = True
            input_shape = [input_shape, (), 0]  # state, individual obs, last actions / actions / agent ids
            # observations
            if self.args.obs_individual_obs:
                input_shape[1] = (1, *scheme["obs"]["vshape"])
                assert input_shape[0][1:] == input_shape[1][1:], f"Image input shape mismatch: {input_shape}"
            # actions
            input_shape[2] += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            # last action
            if self.args.obs_last_action:
                input_shape[2] += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            # agent id
            if self.args.obs_agent_id:
                input_shape[2] += self.n_agents
            input_shape = tuple(input_shape)
        return input_shape

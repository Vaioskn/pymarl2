import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pymarlzooplus.utils.trainable_image_encoder import TrainableImageEncoder


class ACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ACCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_dim = None
        self.is_image = False
        self.cnn_features_dim = args.cnn_features_dim

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"
        if isinstance(input_shape, tuple):  # Image input
            self.obs_dim = self.cnn_features_dim
            self.obs_dim += input_shape[1]
        else:  # Vector input
            self.obs_dim = input_shape

        if self.is_image is True:
            self.cnn = TrainableImageEncoder([input_shape[0][1:]], args)

        # Set up network layers
        self.fc1 = nn.Linear(self.obs_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)

        if self.is_image is True:
            ## Observations
            channels = inputs[0].shape[3]
            height = inputs[0].shape[4]
            width = inputs[0].shape[5]
            # Reshape the observations
            # from [batch size, max steps, n_agents, channels, height, width]
            # to [batch size x max steps x n_agents, channels, height, width]
            inputs[0] = inputs[0].reshape(-1, channels, height, width)
            # CNN
            # from [batch size x max steps x n_agents, channels, height, width]
            # to [batch size x max steps x n_agents, cnn features dim]
            inputs[0] = self.cnn(inputs[0])
            # Reshape the encoded observations
            # from [batch size x max steps x n_agents, cnn features dim]
            # to [batch size, max steps, n_agents, cnn features dim]
            inputs[0] = inputs[0].view(bs, max_t, self.n_agents, self.cnn_features_dim)
            # Concatenate agent ids
            inputs = th.cat(inputs, dim=-1)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        # observations, agent ids
        inputs = [
            batch["obs"][:, ts],
            th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
        ]

        if self.is_image is False:
            inputs = th.cat(inputs, dim=-1)

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        if isinstance(input_shape, int):  # Vector input
            # agent id
            input_shape += self.n_agents
        elif isinstance(input_shape, tuple):  # Image input
            self.is_image = True
            # observations
            input_shape = [(self.n_agents, *input_shape), 0]
            # agent id
            input_shape[1] += self.n_agents
            input_shape = tuple(input_shape)

        return input_shape

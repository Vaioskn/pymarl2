# code adapted from https://github.com/wendelinboehmer/dcg
# and https://github.com/morning9393/HAPPO-HATRPO/tree/master

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pymarlzooplus.utils.trainable_image_encoder import TrainableImageEncoder
from pymarlzooplus.utils.torch_utils import init


class RNNAgentHAPPO(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentHAPPO, self).__init__()
        self.args = args
        self.algo_name = args.name
        self.use_rnn = args.use_rnn
        self.use_orthogonal_init_rnn = args.use_orthogonal_init_rnn
        self.use_feature_normalization = args.use_feature_normalization

        # Use CNN to encode image observations
        self.is_image = False
        if isinstance(input_shape, tuple):  # image input
            self.cnn = TrainableImageEncoder(input_shape, args)
            input_shape = self.cnn.features_dim + input_shape[1]
            self.is_image = True

        assert self.is_image is False, "HAPPO does not support image obs for the time being!"

        assert not (self.use_rnn is False and self.use_orthogonal_init_rnn is True),  (
            f"'self.use_rnn' is {self.use_rnn} "
            f"but 'self.use_orthogonal_init_rnn' is {self.use_orthogonal_init_rnn} !"
        )

        if self.is_image is False and self.use_feature_normalization is True:
            self.feature_norm = nn.LayerNorm(input_shape)

        # Layers initialization
        def init_(m):
            return init(
                m,
                [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal_init_rnn],
                lambda x: nn.init.constant_(x, 0),
                gain=nn.init.calculate_gain('relu')
            )

        # MLP layers
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_shape, args.hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(args.hidden_dim)
        )
        if self.is_image is False:
            self.fc2 = nn.Sequential(
                init_(nn.Linear(args.hidden_dim, args.hidden_dim)),
                nn.ReLU(),
                nn.LayerNorm(args.hidden_dim)
            )

        # RNN layer
        if self.use_rnn is True:
            self.rnn = nn.GRU(
                args.hidden_dim,
                args.hidden_dim,
                num_layers=1
            )
            self.after_rnn_norm = nn.LayerNorm(args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)

        # Output layer
        if self.use_orthogonal_init_rnn is True:
            def init_(m):
                return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.01)
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
                if self.is_image is False:
                    self.fc3 = init_(nn.Linear(args.hidden_dim, args.n_actions))
                else:
                    self.fc2 = init_(nn.Linear(args.hidden_dim, args.n_actions))
        else:
            def init_(m):
                return init(
                    m,
                    nn.init.xavier_uniform_,
                    lambda x: nn.init.constant_(x, 0),
                    gain=0.01
                )
            if self.is_image is False:
                self.fc3 = init_(nn.Linear(args.hidden_dim, args.n_actions))
            else:
                self.fc2 = init_(nn.Linear(args.hidden_dim, args.n_actions))

    def init_hidden(self):
        return th.zeros((1, self.args.hidden_dim)).to(self.fc1[0].weight.device)

    def forward(self, inputs, hidden_states, masks=None):

        if self.is_image is True:
            inputs[0] = self.cnn(inputs[0])
            if len(inputs[1]) > 0:
                inputs = th.concat(inputs, dim=1)
            else:
                inputs = inputs[0]

        if self.is_image is False and self.use_feature_normalization is True:
            inputs = self.feature_norm(inputs)

        if self.is_image is False:
            x = self.fc2(self.fc1(inputs))
        else:
            x = F.relu(self.fc1(inputs))

        if self.use_rnn is True:

            # hidden_state shape: [batch_size, n_agents, hidden_dim]
            assert (
                    len(hidden_states.shape) == 3 and
                    hidden_states.shape[1] == 1 and
                    hidden_states.shape[2] == self.args.hidden_dim
            ), "'hidden_states.shape': {hidden_states.shape}"

            initial_x_shape = x.shape
            if len(initial_x_shape) == 3:
                # shape: [batch_size, timesteps, hidden_dim]
                batch_size = x.shape[0]
                epi_len = x.shape[1]
                # shape: [timesteps, batch_size, hidden_dim] --> [timesteps*batch_size, hidden_dim]
                x = x.transpose(0, 1).reshape(batch_size*epi_len, -1)
                assert (
                        len(masks.shape) == 3 and
                        masks.shape[0] == batch_size and
                        masks.shape[1] == epi_len and
                        masks.shape[2] == 1
                ), f"'masks.shape': {masks.shape}"
                # shape: [timesteps, batch_size, hidden_dim] --> [timesteps*batch_size, hidden_dim]
                masks = masks.transpose(0, 1).reshape(batch_size * epi_len, 1)

            if x.size(0) == hidden_states.size(0):  # Used in inference
                x, h = (
                    self.rnn(
                        x.unsqueeze(0),
                        (hidden_states * masks.repeat(1, 1).unsqueeze(-1)).transpose(0, 1).contiguous()
                    )
                )
                x = x.squeeze(0)
                h = h.transpose(0, 1)
            else:  # Used in inference
                # x is a (T, N, -1) tensor that has been flattened to (T * N, -1)
                N = hidden_states.size(0)
                T = int(x.size(0) / N)

                # unflatten
                x = x.view(T, N, x.size(1))

                # Same deal with masks
                masks = masks.view(T, N)

                # Let's figure out which steps in the sequence have a zero for any agent
                # We will always assume t=0 has a zero in it as that makes the logic cleaner
                has_zeros = (
                    (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
                )

                # +1 to correct the masks[1:]
                if has_zeros.dim() == 0:
                    # Deal with scalar
                    has_zeros = [has_zeros.item() + 1]
                else:
                    has_zeros = (has_zeros + 1).numpy().tolist()

                # add t=0 and t=T to the list
                has_zeros = [0] + has_zeros + [T]

                h = hidden_states.transpose(0, 1)

                outputs = []
                for i in range(len(has_zeros) - 1):
                    # We can now process steps that don't have any zeros in masks together!
                    # This is much faster
                    start_idx = has_zeros[i]
                    end_idx = has_zeros[i + 1]
                    temp = (h * masks[start_idx].view(1, -1, 1).repeat(1, 1, 1)).contiguous()
                    rnn_scores, h = self.rnn(x[start_idx: end_idx], temp)
                    outputs.append(rnn_scores)

                # x is a (T, N, -1) tensor
                x = th.cat(outputs, dim=0)

                if len(initial_x_shape) == 3:
                    # reshape x to (N, T, -1)
                    x = x.transpose(0, 1)
                elif len(initial_x_shape) == 2:
                    x = x.reshape(N*T, -1)
                else:
                    raise ValueError(f"'initial_x_shape': {initial_x_shape}")

                h = h.transpose(0, 1)

            x = self.after_rnn_norm(x)
        else:
            x = F.relu(self.rnn(x))
            h = x.clone()
        if self.is_image is False:
            q = self.fc3(x)
        else:
            q = self.fc2(x)

        return q, h


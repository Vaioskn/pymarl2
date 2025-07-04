# code adapted from https://github.com/AnujMahajanOxf/MAVEN

import numpy as np
import torch as th
import torch.nn as nn

from pymarlzooplus.utils.trainable_image_encoder import TrainableImageEncoder
from pymarlzooplus.utils.torch_utils import init


class HAPPOCritic(nn.Module):
    def __init__(self, scheme, args):
        super(HAPPOCritic, self).__init__()

        self.args = args
        self.algo_name = args.name
        self.n_actions = args.n_actions
        self.n_agents = 1
        self.real_n_agents = args.n_agents
        self.state_dim = None
        self.is_image = False
        self.use_rnn_critic = args.use_rnn_critic
        self.use_orthogonal_init_rnn_critic = args.use_orthogonal_init_rnn_critic
        self.use_feature_normalization_critic = args.use_feature_normalization_critic
        self.cnn_features_dim = args.cnn_features_dim

        assert not (self.use_rnn_critic is False and self.use_orthogonal_init_rnn_critic is True), (
            f"'self.use_rnn_critic' is {self.use_rnn_critic} "
            f"but 'self.use_orthogonal_init_rnn_critic' is {self.use_orthogonal_init_rnn_critic} !"
        )

        input_shape, n_extra_feat = self._get_input_shape(scheme)
        self.output_type = "v"
        if isinstance(input_shape, np.int64) or isinstance(input_shape, int):  # Vector input
            self.state_dim = input_shape
        elif isinstance(input_shape, tuple) and (len(input_shape[0]) == 4) and (input_shape[0][1] == 3):  # Image input
            self.state_dim = (self.cnn_features_dim * input_shape[0][0]) + input_shape[1]  # multiply with n_agents
        else:
            raise ValueError(f"Invalid 'input_shape': {input_shape}")

        if self.is_image is True:
            self.cnn = TrainableImageEncoder([input_shape[0][1:]], args)

        if self.is_image is False and self.use_feature_normalization_critic is True:
            self.feature_norm = nn.LayerNorm(self.state_dim - n_extra_feat)

        # Layers initialization
        def init_(m):
            return init(
                m,
                [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal_init_rnn_critic],
                lambda x: nn.init.constant_(x, 0),
                gain=nn.init.calculate_gain('relu')
            )

        # MLP layers
        self.fc1 = nn.Sequential(
            init_(nn.Linear(self.state_dim, args.hidden_dim)),
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
        self.hidden_states = None
        if self.use_rnn_critic is True:
            self.rnn = nn.GRU(
                args.hidden_dim,
                args.hidden_dim,
                num_layers=1
            )
            self.after_rnn_norm = nn.LayerNorm(args.hidden_dim)
            if self.use_orthogonal_init_rnn_critic is True:
                for name, param in self.rnn.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)

        # Output layer
        if self.use_orthogonal_init_rnn_critic is True:
            self.v_out = init_(nn.Linear(args.hidden_dim, 1))
        else:
            self.v_out = nn.Linear(args.hidden_dim, 1)

    def init_hidden(self, batch_size):
        self.hidden_states = th.zeros(
            (1, self.args.hidden_dim)
        ).to(
            self.fc1[0].weight.device
        ).unsqueeze(0).expand(batch_size, 1, -1)  # shape: [batch_size, timesteps, hidden_dim]

    def forward(self, batch, t=None, hidden_states=None, masks=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)

        if self.is_image is True:
            channels = inputs[0].shape[3]
            height = inputs[0].shape[4]
            width = inputs[0].shape[5]
            # Reshape the states
            # from [batch size, max steps, real_n_agents, channels, height, width]
            # to [batch size x max steps x real_n_agents, channels, height, width]
            inputs[0] = inputs[0].reshape(-1, channels, height, width)
            # to [batch size x max steps x real_n_agents, cnn features dim]
            inputs[0] = self.cnn(inputs[0])
            # to [batch size x max steps, real_n_agents x cnn features dim]
            inputs[0] = inputs[0].view(bs * max_t, self.real_n_agents * self.cnn_features_dim)
            # to [batch size x max steps, n_agents, real_n_agents x cnn features dim]
            inputs[0] = inputs[0].unsqueeze(1).repeat(1, self.n_agents, 1)
            # to [batch size, max steps, n_agents, real_n_agents x cnn features dim]
            inputs[0] = inputs[0].view(bs, max_t, self.n_agents, self.real_n_agents * self.cnn_features_dim)
            # to [batch size, max steps, n_agents, cnn features dim x real_n_agents + extra features]
            inputs = th.cat(inputs, dim=-1)

            x = self.fc1(inputs)

        else:
            x = self.fc2(self.fc1(inputs))

        if self.use_rnn_critic is True:

            # Create masks if not provided
            if masks is None:
                assert t is not None

                masks = 1 - batch["terminated"][:, slice(t, t + 1)]
                assert len(masks.shape) == 3 and masks.shape[1] == masks.shape[2] == 1, f"masks: {masks}"
                masks = masks.squeeze(2)

                assert len(x.shape) == 4 and x.shape[1] == x.shape[2] == 1, f"x: {x}"
                x = x.squeeze(1).squeeze(1)

            if x.size(0) == hidden_states.size(0):  # Used in inference
                x, h = (
                    self.rnn(
                        x.unsqueeze(0),
                        (hidden_states * masks.repeat(1, 1).unsqueeze(-1)).transpose(0, 1).contiguous()
                    )
                )
                x = x.squeeze(0)
                h = h.transpose(0, 1)
            else:  # Used in training
                x = x.squeeze(1).squeeze(1)

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

                # flatten
                x = x.reshape(T*N, -1)
                h = h.transpose(0, 1)

            x = self.after_rnn_norm(x)

        values = self.v_out(x.squeeze())

        if self.use_rnn_critic is True:
            return [values, h]
        else:
            return [values]

    def _build_inputs(self, batch, t=None):
        bs = batch["batch_size"]
        max_t = batch["max_seq_length"] if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        if self.is_image is False:
            if self.use_feature_normalization_critic is True:
                inputs = self.feature_norm(batch["state"][:, ts])
            else:
                inputs = batch["state"][:, ts]
            inputs = [inputs.unsqueeze(2).repeat(1, 1, self.n_agents, 1)]
        else:
            inputs = [batch["state"][:, ts]]

        # individual observations
        assert not (self.args.obs_individual_obs is True and self.is_image is True), (
            "In case of state image, obs_individual_obs is not supported."
        )
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

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
                    [th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]],
                    dim=1
                )
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        if self.is_image is False:
            inputs = th.cat(inputs, dim=-1)
        else:
            if len(inputs) > 2:
                inputs[1] = th.cat(inputs[1:], dim=-1)
                del inputs[2]
                assert len(inputs) == 2, "length of inputs: {}".format(len(inputs))
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):

        n_extra_feat = 0

        # vector-state
        input_shape = scheme["state"]["vshape"]
        if isinstance(input_shape, int):
            # observations
            if self.args.obs_individual_obs:
                n_extra_feat += scheme["obs"]["vshape"] * self.real_n_agents
            # last actions
            if self.args.obs_last_action:
                n_extra_feat += scheme["actions_onehot"]["vshape"][0] * self.real_n_agents
            input_shape += n_extra_feat
        # image-state
        elif isinstance(input_shape, tuple):
            assert self.args.obs_individual_obs is False, (
                "In case of state-image, 'obs_individual_obs' argument is not supported."
            )
            self.is_image = True
            input_shape = [input_shape, 0]
            if self.args.obs_last_action:
                n_extra_feat += scheme["actions_onehot"]["vshape"][0] * self.real_n_agents
            input_shape[1] += n_extra_feat
            input_shape = tuple(input_shape)

        return input_shape, n_extra_feat

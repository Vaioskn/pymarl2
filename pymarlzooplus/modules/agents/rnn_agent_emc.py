# code adapted from https://github.com/wendelinboehmer/dcg
# and https://github.com/lich14/CDS/blob/main/CDS_GRF/modules/agents/rnn_agent.py

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pymarlzooplus.utils.trainable_image_encoder import TrainableImageEncoder


class RNNAgentEMC(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentEMC, self).__init__()
        self.args = args
        self.algo_name = args.name
        self.use_rnn = args.use_rnn
        self.n_agents = args.n_agents

        # Use CNN to encode image observations
        self.is_image = False
        if isinstance(input_shape, tuple):  # image input
            self.cnn = TrainableImageEncoder(input_shape, args)
            input_shape = self.cnn.features_dim + input_shape[1]
            self.is_image = True

        assert self.is_image is False, "EMC does not support image obs for the time being!"
        assert self.use_rnn is True, "EMC is implemented only to use RNN for the time being!"

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.use_rnn is True:
            self.rnn = nn.GRU(
                input_size=args.hidden_dim,
                num_layers=1,
                hidden_size=args.hidden_dim,
                batch_first=True
            )
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):

        if self.is_image is True:
            inputs[0] = self.cnn(inputs[0])
            if len(inputs[1] > 0):
                inputs = th.concat(inputs, dim=1)
            else:
                inputs = inputs[0]

        bs = inputs.shape[0]
        epi_len = inputs.shape[1]
        num_feat = inputs.shape[2]
        inputs = inputs.reshape(bs * epi_len, num_feat)

        x = F.relu(self.fc1(inputs))
        x = x.reshape(bs, epi_len, self.args.hidden_dim)
        h_in = hidden_state.reshape(1, bs, self.args.hidden_dim).contiguous()
        x, h = self.rnn(x, h_in)
        x = x.reshape(bs * epi_len, self.args.hidden_dim)
        q = self.fc2(x)
        q = q.reshape(bs, epi_len, self.args.n_actions)

        return q, h


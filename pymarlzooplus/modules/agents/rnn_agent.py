# code adapted from https://github.com/wendelinboehmer/dcg
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pymarlzooplus.utils.trainable_image_encoder import TrainableImageEncoder


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
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

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.use_rnn is True:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on the same device as the model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):

        if self.is_image is True:
            inputs[0] = self.cnn(inputs[0])
            if len(inputs[1]) > 0:
                inputs = th.concat(inputs, dim=1)
            else:
                inputs = inputs[0]

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)

        return q, h

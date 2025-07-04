import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pymarlzooplus.utils.trainable_image_encoder import TrainableImageEncoder


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = None
        self.is_image = False
        if isinstance(args.state_shape, int): # Vector input
            self.state_dim = int(np.prod(args.state_shape))
        elif (len(args.state_shape) == 4) and (args.state_shape[1] == 3): # Image input
            self.state_dim = args.cnn_features_dim * args.state_shape[0]
            self.is_image = True
        else:
            raise ValueError(f"Invalid 'state_shape': {args.state_shape}")

        self.embed_dim = args.mixing_embed_dim
        # cnn
        if self.is_image is True:
            self.cnn = TrainableImageEncoder([args.state_shape[1:]], args)

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):

        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        if self.is_image is True:

            channels = states.shape[3]
            height = states.shape[4]
            width = states.shape[5]
            # Reshape the states
            # from [batch size, max steps, n_agents, channels, height, width]
            # to [batch size x max steps x n_agents, channels, height, width]
            states = states.reshape(-1, channels, height, width)
            total_samples = states.shape[0]
            n_batches = math.ceil(total_samples / bs)

            # state-images are processed in batches due to memory limitations
            states_new = []
            for batch in range(n_batches):
                # from [batch size, channels, height, width]
                # to [batch size, cnn features dim]
                states_new.append(self.cnn(states[batch*bs: (batch+1)*bs]))

            # to [batch size x max steps x n_agents, cnn features dim]
            states = th.concat(states_new, dim=0)
            states = states.view(-1, self.state_dim)
        else:
            states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

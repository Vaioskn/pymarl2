# code adapted from https://github.com/wendelinboehmer/dcg
# and https://github.com/lich14/CDS/blob/main/CDS_GRF/modules/agents/rnn_agent.py

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pymarlzooplus.utils.trainable_image_encoder import TrainableImageEncoder


class RNNAgentCDS(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentCDS, self).__init__()
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

        assert self.is_image is False, "CDS does not support image obs for the time being!"
        assert self.use_rnn is True, "CDS is implemented only to use RNN for the time being!"

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

        # Create an MLP module list to generate local Q values for each agent
        self.mlp = nn.ModuleList([nn.Linear(args.hidden_dim, args.n_actions) for _ in range(self.n_agents)])

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

        if len(hidden_state.shape) == 2:
            hidden_state = hidden_state.unsqueeze(0)

        hidden_state = hidden_state.contiguous()
        input_shape = inputs.shape

        if len(input_shape) == 2:
            x = F.relu(self.fc1(inputs))
            x = x.unsqueeze(1)
            gru_out, _ = self.rnn(x, hidden_state)
            local_q = th.stack([mlp(gru_out[id, :, :]) for id, mlp in enumerate(self.mlp)], dim=1)
            local_q = local_q.squeeze()

            gru_out = gru_out.squeeze()
            q = self.fc2(gru_out)

        elif len(input_shape) == 4:
            inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
            inputs = inputs.reshape(-1, inputs.shape[-1])
            x = F.relu(self.fc1(inputs))
            x = x.reshape(-1, input_shape[2], x.shape[-1])

            gru_out, _ = self.rnn(x, hidden_state.to(x.device))
            gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])
            q = self.fc2(gru_out_c)

            q = q.reshape(-1, gru_out.shape[1], q.shape[-1])
            q = q.reshape(-1, input_shape[1], q.shape[-2], q.shape[-1]).permute(0, 2, 1, 3)

            gru_out_local = gru_out.reshape(-1, input_shape[1], gru_out.shape[-2], gru_out.shape[-1])
            local_q = th.stack(
                [mlp(gru_out_local[:, id].reshape(-1, gru_out_local.shape[-1])) for id, mlp in enumerate(self.mlp)],
                dim=1
            )
            local_q = local_q.reshape(-1, gru_out_local.shape[-2], local_q.shape[-2], local_q.shape[-1])

        else:
            raise ValueError(f"'input_shape': {input_shape}")

        # Combining global q values with local q values
        q = q + local_q

        return q, gru_out, local_q



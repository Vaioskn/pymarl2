import torch.nn as nn
from pymarlzooplus.modules.agents.rnn_agent import RNNAgent
import torch as th


class RNNNSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNNSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape

        self.is_image = False
        if isinstance(input_shape, tuple):  # image input
            self.is_image = True

        # Create the non-shared agents
        self.agents = th.nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.n_agents)])

    def init_hidden(self):
        # make hidden states on the same device as the model
        return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []

        if (
                (self.is_image is False and inputs.size(0) == self.n_agents) or
                (self.is_image is True and inputs[0].size(0) == self.n_agents)
        ):  # Single sample
            for i in range(self.n_agents):
                if self.is_image is True:  # Image observation
                    agent_inputs = [inputs[0][i].unsqueeze(0), []]
                    if len(inputs[1]) > 0:
                        agent_inputs[1] = inputs[1][i].unsqueeze(0)
                else:  # Vector observation
                    agent_inputs = inputs[i].unsqueeze(0)
                q, h = self.agents[i](agent_inputs, hidden_state[:, i])
                hiddens.append(h)
                qs.append(q)
            return th.cat(qs), th.cat(hiddens).unsqueeze(0)

        else:  # Multiple samples
            if self.is_image is True:  # Image observation
                inputs[0] = inputs[0].view(-1, self.n_agents, *self.input_shape[0])
                if len(inputs[1]) > 0:
                    inputs[1] = inputs[1].view(-1, self.n_agents, self.input_shape[1])
            else:  # Vector observation
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
            for i in range(self.n_agents):
                if self.is_image is True:  # Image observation
                    agent_inputs = [inputs[0][:, i], []]
                    if len(inputs[1]) > 0:
                        agent_inputs[1] = inputs[1][:, i]
                else:  # Vector observation
                    agent_inputs = inputs[:, i]
                q, h = self.agents[i](agent_inputs, hidden_state[:, i])
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
            return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)

    def cuda(self, device="cuda:0"):
        # TODO: Fix this in case of image inputs and different devices. Do the same for RNN agent
        for a in self.agents:
            a.cuda(device=device)

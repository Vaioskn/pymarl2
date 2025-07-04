import torch as th
from torch.autograd import Variable
import torch.nn.functional as F

from pymarlzooplus.modules.agents import REGISTRY as agent_REGISTRY


def onehot_from_logits(logits, eps=0.0):
    """
    Given a batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs


def sample_gumbel(shape, eps=1e-20, tens_type=th.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -th.log(-th.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=-1)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    """

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


# This multi-agent controller shares parameters between agents
class MADDPGMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.is_image = False  # Image input
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = None
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):

        # Just for compatibility
        extra_returns = {}

        # Only select actions for the selected batch elements in bs
        agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1)
        return chosen_actions, extra_returns

    def target_actions(self, ep_batch, t_ep):
        agent_outputs = self.forward(ep_batch, t_ep)
        return onehot_from_logits(agent_outputs)

    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions == 0] = -1e10
        return agent_outs

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def init_hidden_one_agent(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to, e.g., delegate building inputs to each agent
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        if self.is_image is False:
            inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        else:
            img_ch, img_h, img_w = inputs[0].shape[2:]
            inputs = [
                inputs[0].reshape(bs * self.n_agents, img_ch, img_h, img_w),
                [] if len(inputs) == 1
                   else
                th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs[1:]], dim=1)
            ]

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if isinstance(input_shape, int):  # vector input
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0]
            if self.args.obs_agent_id:
                input_shape += self.n_agents
        elif isinstance(input_shape, tuple):  # image input
            self.is_image = True
            input_shape = [input_shape, 0]
            if self.args.obs_last_action:
                input_shape[1] += scheme["actions_onehot"]["vshape"][0]
            if self.args.obs_agent_id:
                input_shape[1] += self.n_agents
            input_shape = tuple(input_shape)  # list to tuple

        return input_shape

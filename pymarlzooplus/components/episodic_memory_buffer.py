import numpy as np
from pymarlzooplus.components.lrn_knn import LRUKNN


class EpisodicMemoryBuffer:
    def __init__(self, args, scheme):

        assert not isinstance(scheme['state']['vshape'], tuple), "EMC does not support image obs for the time being!"

        self.ec_buffer = LRUKNN(args.emdqn_buffer_size, args.emdqn_latent_dim, 'game')
        self.rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
        state_dim = scheme['state']['vshape']
        self.random_projection = self.rng.normal(
            loc=0, scale=1. / np.sqrt(args.emdqn_latent_dim),
            size=(args.emdqn_latent_dim, state_dim)
        )
        self.q_episodic_memory_cwatch = []
        self.args = args
        self.update_counter = 0
        self.qecwatch = []
        self.qec_found = 0
        self.update_counter = 0

    def update_kdtree(self):
        self.ec_buffer.update_kdtree()

    def peek(self, key, value_decay, modify):
        return self.ec_buffer.peek(key, value_decay, modify)

    def update_ec(self, episode_batch):
        ep_state = episode_batch['state'][0, :]
        ep_action = episode_batch['actions'][0, :]
        ep_reward = episode_batch['reward'][0, :]
        rtd = 0.
        for t in range(episode_batch.max_seq_length - 1, -1, -1):
            s = ep_state[t]
            a = ep_action[t]
            r = ep_reward[t]
            z = np.dot(self.random_projection, s.flatten().cpu())
            rtd = r + self.args.gamma * rtd
            z = z.reshape(self.args.emdqn_latent_dim)
            qd = self.ec_buffer.peek(z, rtd, True)
            if qd is None:  # new action
                self.ec_buffer.add(z, rtd)

    def hit_probability(self):
        return 1.0 * self.qec_found / self.args.batch_size / self.update_counter

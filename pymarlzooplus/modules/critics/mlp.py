import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pymarlzooplus.utils.trainable_image_encoder import TrainableImageEncoder


class MLP(nn.Module):
    def __init__(self, input_shape, args, output_dim):
        super(MLP, self).__init__()

        self.n_agents = args.n_agents
        self.obs_individual_obs = args.obs_individual_obs
        self.input_shape = input_shape
        self.critic_type = args.critic_type
        hidden_dim = args.hidden_dim

        # Use CNN to encode image observations
        self.is_image = False
        if isinstance(self.input_shape, tuple):  # image input
            self.cnn = TrainableImageEncoder((self.input_shape[0][1:],), args)
            self.n_agents_for_state = self.input_shape[0][0]
            # state
            if self.critic_type == "ac_critic_ns":  # Individual observation
                mlp_input_shape = self.cnn.features_dim
            else:  # The entire state consists of the individual observations of all agents
                mlp_input_shape = self.cnn.features_dim * self.n_agents
            # observation
            if self.obs_individual_obs is True and len(self.input_shape) > 1 and len(self.input_shape[1]) == 3:
                if self.critic_type == "cv_critic_ns":  # All individual observations
                    mlp_input_shape += self.cnn.features_dim * self.n_agents
                else:  # Single individual observation
                    mlp_input_shape += self.cnn.features_dim
            else:
                self.obs_individual_obs = False
            # action / last action / agent id
            if len(self.input_shape) > 2 and self.input_shape[2] > 0:
                mlp_input_shape += self.input_shape[2]
            input_shape = mlp_input_shape
            self.is_image = True

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):

        if self.is_image is True:

            # Use another variable than 'input' to avoid changing the original input which probably are used repeatedly
            x = [None]
            if self.obs_individual_obs:
                x.append(None)
            if len(self.input_shape) > 2 and self.input_shape[2] > 0:
                x.append(None)

            if len(inputs[0].shape) == 4:
                # shape: (bs * max_t * n_agents, ch, h, w) → (bs * max_t, n_agents * cnn_features_dim)
                x[0] = self.cnn(inputs[0]).view(-1, self.n_agents * self.cnn.features_dim)
            elif len(inputs[0].shape) == 6:
                # shape: (bs, max_t, n_agents_for_state, ch, h, w)
                n_agents_for_state, img_ch, img_h, img_w = inputs[0].shape[2:]
                # shape (reshape): (bs, max_t, n_agents_for_state, ch, h, w) →
                #                  (bs * max_t * n_agents_for_state, ch, h, w)
                # shape (CNN): (bs * max_t * n_agents_for_state, ch, h, w) →
                #              (bs * max_t * n_agents_for_state, cnn_features_dim)
                x[0] = self.cnn(inputs[0].reshape(-1, img_ch, img_h, img_w))
                if self.n_agents_for_state == self.n_agents:
                    # shape (repeat): (bs * max_t * n_agents_for_state, cnn_features_dim) →
                    #                 (bs * max_t * n_agents_for_state, n_agents_for_state * cnn_features_dim)
                    x[0] = x[0].repeat(1, self.n_agents)
                else:
                    assert self.n_agents_for_state == 1, "'self.n_agents_for_state' should be 1 or 'n_agents'"
                    if n_agents_for_state == self.n_agents:
                        # x shape: (bs * max_t * n_agents, cnn_features_dim) →
                        #          (bs * max_t * n_agents_for_state, n_agents * cnn_features_dim)
                        x[0] = x[0].view(-1, self.n_agents * self.cnn.features_dim)
                    else:
                        assert n_agents_for_state == 1, "'n_agents_for_state' should be 1 or 'n_agents'"
            else:
                raise NotImplementedError("Unsupported input shape for CNN: {}".format(inputs[0].shape))

            # observation
            if self.obs_individual_obs:
                if len(inputs[1].shape) == 4:
                    # shape: (bs * max_t * n_agents, ch, h, w) → (bs * max_t, n_agents * cnn_features_dim)
                    x[1] = self.cnn(inputs[1]).view(-1, self.n_agents * self.cnn.features_dim)
                elif len(inputs[1].shape) == 6:
                    # shape: (bs, max_t, n_agents, ch, h, w)
                    img_ch, img_h, img_w = inputs[1].shape[3:]
                    # shape (reshape): (bs, max_t, n_agents_for_state, ch, h, w) →
                    #                  (bs * max_t * n_agents_for_state, ch, h, w)
                    # shape (CNN): (bs * max_t * n_agents_for_state, ch, h, w) →
                    #              (bs * max_t * n_agents_for_state, cnn_features_dim)
                    x[1] = self.cnn(inputs[1].reshape(-1, img_ch, img_h, img_w))
                else:
                    raise NotImplementedError("Unsupported input shape for CNN: {}".format(inputs[1].shape))

            # action / last action / agent id
            if len(self.input_shape) > 2 and self.input_shape[2] > 0:
                extra_feats_index = 2 if self.obs_individual_obs and len(inputs) > 1 else 1
                if len(inputs[extra_feats_index].shape) == 2:
                    # shape: (bs * max_t, n_agents * extra_feats_dim)
                    x[extra_feats_index] = inputs[extra_feats_index].clone()
                elif len(inputs[extra_feats_index].shape) == 4:
                    # shape: (bs, max_t, n_agents_for_state, extra_feats_dim)
                    extra_feats_dim = inputs[extra_feats_index].shape[3]
                    # shape: (bs, max_t, n_agents_for_state, extra_feats_dim) →
                    #        (bs * max_t * n_agents_for_state, extra_feats_dim)
                    x[extra_feats_index] = inputs[extra_feats_index].clone().reshape(-1, extra_feats_dim)
                else:
                    raise NotImplementedError(
                        "Unsupported input shape for MLP: {}".format(inputs[extra_feats_index].shape)
                    )

            if len(inputs) > 1:
                x = th.cat(x, dim=1)
            else:
                x = x[0]

        else:
            x = inputs.clone()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

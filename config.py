class CartPoleConfig():
    n_episodes = 5_000
    policy_network_lr = 1e-4
    baseline_network_lr = 1e-3
    gamma = 1
    use_baselines = True
    advantage_normalization = True
    record_freq = 1_000
    discrete = True
    game = 'CartPole-v0'
    exp_id = 'exp-3'

class HalfCheetahConfig():
    n_episodes = 10_000
    policy_network_lr = 1e-4
    baseline_network_lr = 1e-3
    gamma = 0.9
    use_baselines = True
    advantage_normalization = True
    record_freq = 1_000
    discrete = False
    game = 'HalfCheetah-v4'
    exp_id = 'exp-3'

class InvertedPendulumConfig():
    n_episodes = 5_000
    policy_network_lr = 1e-4
    baseline_network_lr = 1e-3
    gamma = 1
    use_baselines = False
    advantage_normalization = False
    record_freq = 1_000
    discrete = False
    game = 'InvertedPendulum-v4'
    exp_id = 'exp-3'
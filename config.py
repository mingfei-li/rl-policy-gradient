class Config():
    n_batches = 5000
    batch_size = 1
    policy_network_lr = 1e-4
    baseline_network_lr = 1e-3
    gamma = 1
    n_batches_baseline = 0
    advantage_normalization = False
    record_freq = 1_000
    # discrete = True
    # game = 'CartPole-v0'
    discrete = False
    game = 'InvertedPendulum-v4'
    exp_id = 'exp-43'
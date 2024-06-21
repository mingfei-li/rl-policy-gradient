class Config():
    n_batches = 5000
    batch_size = 100
    policy_network_lr = 3e-2
    baseline_network_lr = 1e-3
    gamma = 1
    n_batches_baseline = 2
    record_freq = 1_000
    # discrete = True
    # game = 'CartPole-v0'
    discrete = False
    game = 'InvertedPendulum-v4'
    exp_id = 'exp-35'
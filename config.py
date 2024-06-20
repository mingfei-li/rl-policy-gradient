class Config():
    n_batches = 200
    batch_size = 20
    policy_network_lr = 1e-3
    baseline_network_lr = 1e-3
    gamma = 1
    n_batches_baseline = 10
    record_freq = 1_000
    discrete = True
    game = 'CartPole-v0'
    # discrete = False
    # game = 'InvertedPendulum-v4'
    exp_id = 'exp-20'
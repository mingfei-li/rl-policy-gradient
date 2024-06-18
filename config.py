class Config():
    n_batches = 100
    batch_size = 100
    policy_network_lr = 1e-3
    baseline_network_lr = 1e-2
    gamma = 1
    use_baseline = True
    record_freq = 1_000
    discrete = True
    game = 'CartPole-v0'
    # discrete = False
    # game = 'InvertedPendulum-v4'
    exp_id = 'exp-16'
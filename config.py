class Config():
    n_training_episodes = 5000
    policy_network_lr = 0.1
    baseline_network_lr = 0.001
    gamma = 1
    use_baseline = True
    record_freq = 1000
    discrete = False
    game = 'InvertedPendulum-v4'
    exp_id = 'exp-8'
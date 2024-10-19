agents = {
    "baseline": [{
        "meta": {
            "evaluate_file": "./data/evaluation/dt_baseline.csv",
            "model_file": "./models/dt_baseline.pth",
            "data_file": "./data/trajectories/baseline.csv",
            "device": "cuda",
            "training_iterations": 1,
            "training_steps": 5,
        },
        "hyperparams": {
            "max_episode_len": 1440,
            "target_return": 1000.,
            "dropout": 0.1,
            "discount_factor": 0.99,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "warmup_steps": 100,
            "eval_episodes": 5,
        },
        "architecture": {
            "num_tokens": 64,
            "num_attention_embedding_layers": 4,
            "num_attention_heads": 8,
            "embed_dim": 128
        },
        "data": {
            "random_frequency": 1.0,
            "masked": False,
            "modifiable_buses": 4,
            "modifiable_lines": 4,
            "redispatch_bins": 11,
            "curtail_bins": 11,
        }
    }],

    "alpha": [
        {
            "meta": {
                "evaluate_file": "./data/evaluation/dt_alpha_0.csv",
                "model_file": "./models/dt_alpha_0.pth",
                "data_file": "./data/trajectories/alpha_0.csv",
                "device": "cuda",
            },
            "hyperparams": {
                "max_episode_len": 1440,
                "target_return": 1000.,
                "dropout": 0.1,
                "discount_factor": 0.99,
                "batch_size": 256,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "warmup_steps": 100,
                "eval_episodes": 5,
            },
            "architecture": {
                "num_tokens": 64,
                "num_attention_embedding_layers": 4,
                "attention_heads": 8
            },
            "data": {
                "random_frequency": 1.0,
                "masked": True,
                "modifiable_buses": 1,
                "modifiable_lines": 1,
                "redispatch_bins": 11,
                "curtail_bins": 11,
            }
        },
        {
            "meta": {
                "evaluate_file": "./data/decision_transformer/alpha_1.csv"
            },
            "hyperparams": {
                "max_episode_len": 1440,
                "target_return": 1000.,
                "state_mean": [0.],
                "state_std": [1.],
                "dropout": 0.1
            },
            "architecture": {
                "num_tokens": 64,
                "num_attention_embedding_layers": 4,
                "attention_heads": 8
            },
            "data": {
                "random_frequency": 0.75,
                "masked": True,
                "modifiable_buses": 1,
                "modifiable_lines": 1,
                "redispatch_bins": 11,
                "curtail_bins": 11,
            }
        },
        {
            "meta": {
                "evaluate_file": "./data/decision_transformer/alpha_2.csv"
            },
            "hyperparams": {
                "max_episode_len": 1440,
                "target_return": 1000.,
                "state_mean": [0.],
                "state_std": [1.],
                "dropout": 0.1
            },
            "architecture": {
                "num_tokens": 64,
                "num_attention_embedding_layers": 4,
                "attention_heads": 8
            },
            "data": {
                "random_frequency": 0.75,
                "masked": True,
                "modifiable_buses": 4,
                "modifiable_lines": 4,
                "redispatch_bins": 11,
                "curtail_bins": 11,
            }
        },
    ],

    "bravo": [
        {
            "meta": {
                "evaluate_file": "./data/decision_transformer/bravo_0.csv"
            },
            "hyperparams": {
                "max_episode_len": 1440,
                "target_return": 1000.,
                "state_mean": [0.],
                "state_std": [1.],
                "dropout": 0.1
            },
            "architecture": {
                "num_tokens": 64,
                "num_attention_embedding_layers": 1,
                "attention_heads": 8
            },
            "data": {
                "random_frequency": 1.0,
                "masked": False,
                "modifiable_buses": 4,
                "modifiable_lines": 4,
                "redispatch_bins": 11,
                "curtail_bins": 11,
            }
        },
        {
            "meta": {
                "evaluate_file": "./data/decision_transformer/bravo_1.csv"
            },
            "hyperparams": {
                "max_episode_len": 1440,
                "target_return": 1000.,
                "state_mean": [0.],
                "state_std": [1.],
                "dropout": 0.1
            },
            "architecture": {
                "num_tokens": 64,
                "num_attention_embedding_layers": 2,
                "attention_heads": 8
            },
            "data": {
                "random_frequency": 1.0,
                "masked": False,
                "modifiable_buses": 4,
                "modifiable_lines": 4,
                "redispatch_bins": 11,
                "curtail_bins": 11,
            }
        },
        {
            "meta": {
                "evaluate_file": "./data/decision_transformer/bravo_3.csv"
            },
            "hyperparams": {
                "max_episode_len": 1440,
                "target_return": 1000.,
                "state_mean": [0.],
                "state_std": [1.],
                "dropout": 0.1
            },
            "architecture": {
                "num_tokens": 64,
                "num_attention_embedding_layers": 8,
                "attention_heads": 8
            },
            "data": {
                "random_frequency": 1.0,
                "masked": False,
                "modifiable_buses": 4,
                "modifiable_lines": 4,
                "redispatch_bins": 11,
                "curtail_bins": 11,
            }
        },
    ]
}

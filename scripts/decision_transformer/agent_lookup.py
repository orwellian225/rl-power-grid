agents = {
    "baseline": [{
        "meta": {
            "evaluate_file": "./data/evaluation/dt_baseline.csv",
            "model_file": "./models/dt_baseline.pth",
            "data_file": "./data/trajectories/baseline.csv",
            "device": "cuda",
            "training_iterations": 15,
            "training_steps": 400,
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
                "training_iterations": 15,
                "training_steps": 400,
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
                "masked": True,
                "modifiable_buses": 1,
                "modifiable_lines": 1,
                "redispatch_bins": 11,
                "curtail_bins": 11,
            }
        },
        {
            "meta": {
                "evaluate_file": "./data/evaluation/dt_alpha_1.csv",
                "model_file": "./models/dt_alpha_1.pth",
                "data_file": "./data/trajectories/alpha_1.csv",
                "device": "cuda",
                "training_iterations": 15,
                "training_steps": 400,
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
                "evaluate_file": "./data/evaluation/dt_alpha_2.csv",
                "model_file": "./models/dt_alpha_2.pth",
                "data_file": "./data/trajectories/alpha_2.csv",
                "device": "cuda",
                "training_iterations": 15,
                "training_steps": 400,
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
                "embed_dim": 128,
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
                "evaluate_file": "./data/evaluation/dt_bravo_0.csv",
                "model_file": "./models/dt_bravo_0.pth",
                "data_file": "./data/trajectories/bravo_0.csv",
                "device": "cuda",
                "training_iterations": 15,
                "training_steps": 400,
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
                "num_attention_embedding_layers": 1,
                "num_attention_heads": 8,
                "embed_dim": 128,
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
                "evaluate_file": "./data/evaluation/dt_bravo_1.csv",
                "model_file": "./models/dt_bravo_1.pth",
                "data_file": "./data/trajectories/bravo_1.csv",
                "device": "cuda",
                "training_iterations": 15,
                "training_steps": 400,
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
                "num_attention_embedding_layers": 2,
                "num_attention_heads": 8,
                "embed_dim": 128,
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
                "evaluate_file": "./data/evaluation/dt_bravo_2.csv",
                "model_file": "./models/dt_bravo_2.pth",
                "data_file": "./data/trajectories/bravo_2.csv",
                "device": "cuda",
                "training_iterations": 15,
                "training_steps": 400,
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
                "num_attention_embedding_layers": 8,
                "num_attention_heads": 8,
                "embed_dim": 128,
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

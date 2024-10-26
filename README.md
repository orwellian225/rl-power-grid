# COMS4061A RL Assignment

## Group Members

* Konstantinos Hatzipanis
* Brendan Griffiths
* Lisa Godwin
* Nihal Ranchod

## Depedencies

* grid2op - `pip install grid2op`
* lightsim2grid - `pip install lightsim2grid`
* pytorch
* stable baslines 3 - `pip install stable-baselines3` - PPO and DQN
* transformers - `pip install transformers` - Decision Transformer

## Running

### Decision Transformer

| Available Agent Improvement | Available Agent Configuration |
|-----------------------------|-------------------------------|
| baseline | 0 |
| alpha | 0 |
| alpha | 1 |
| alpha | 2 |
| bravo | 0 |
| bravo | 1 |
| bravo | 2 |

> More Information available in `./scripts/decision_transformer/notes.md`

* Evaluating: `python3 ./scripts/decision_transformer/evaluate.py <agent_improvement> <agent_configuration>`
* Data Generation: `python3 ./scripts/decision_transformer/generate.py <agent_improvement> <agent_configuration> <num_trajectories>`
* Training: `python3 ./scripts/decision_transformer/train.py <agent_improvement> <agent_configuration>`

### DQN

To run baseline DQN Model:
```bash
python train_dqn_baseline.py
```

To run Improvement 1 DQN Model:
```bash
python train_dqn_improvement_1.py
```

To run Improvement 2 DQN Model:
```bash
python train_dqn_improvement_2.py
```

### Running PPO

The train and testing environments are straight to run. 
The reader is required to read the comments in the files
and can adjust the parameters of directories accordingly.

The environments have been duplicated to help with the certain iteration.
`ppo_env.py` is used as the baseline case. 
Please uncomment this in the training and the test file when in this case.
Similarly, use the environemtn `env_2.py` for the 1 iteration.
Finally, use the environemnt `env_2_train.py` for the reward shaping iteration 2.

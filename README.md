# COMS40 RL Assignment

## Group Members

* Konstantinos Hatzipanis
* Brendan Griffiths
* Lisa Godwin
* Nihal Ranchod

## Depedencies

* grid2op - `pip install grid2op`
* lightsim2grid - `pip install lightsim2grid`

## Running

### Decision Transformer

| Available Agent Improvement | Available Agent Configuration |
|-----------------------------|-------------------------------|
| baseline | 0 |
| alpha | 0 |
| alpha | 1 |
| bravo | 0 |
| bravo | 1 |
| bravo | 3 |

> More Information available in `./scripts/decision_transformer/notes.md`

* Evaluating: `python3 ./scripts/decision_transformer/evaluate.py <agent_improvement> <agent_configuration>`
* Data Generation: `python3 ./scripts/decision_transformer/generate.py <agent_improvement> <agent_configuration> <num_trajectories>`
* Training: `python3 ./scripts/decision_transformer/train.py <agent_improvement> <agent_configuration>`
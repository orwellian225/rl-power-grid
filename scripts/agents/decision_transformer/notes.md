# Decision Transformer

## Agents

* Baseline - gym-experiment-powergrid-random_1.0-570939
* Improvement 1: only 1 action unmasked - gym-experiment-powergrid-masked_random_1.0-782768
* Improvement 1: Multiple unmasked actions - gym-experiment-powergrid-masked_random_0.8-741523
* Improvement 1: 1 line, 1 bus change masked actions - 

## Improvement 1

Action space modification and experimentation

* Training period: 500 steps per epoch(not actually epochs but close enough), 10 epochs

## Improvement 2

Mean of the agent is very low, possibly running into architectural issues with vanishing gradients and such

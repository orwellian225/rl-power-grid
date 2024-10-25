# COMS40 RL Assignment

## Group Members

* Konstantinos Hatzipanis
* Brendan Griffiths
* Lisa Godwin
* Nihal Ranchod

## Depedencies

* grid2op - `pip install grid2op`
* lightsim2grid - `pip install limsim2grid`

## Running

Run from the top level train script.

```bash
python train.py
```
## Running PPO

The train and testing environments are straight to run. 
The reader is required to read the comments in the files
and can adjust the parameters of directories accordingly.

The environments have been duplicated to help with the certain iteration.
`ppo_env.py` is used as the baseline case. 
Please uncomment this in the training and the test file when in this case.
Similarly, use the environemtn `env_2.py` for the 1 iteration.
Finally, use the environemnt `env_2_train.py` for the reward shaping iteration 2.
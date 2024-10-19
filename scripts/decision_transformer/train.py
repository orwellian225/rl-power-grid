import sys
import csv
import random

import torch
import wandb
import numpy as np

from agent_lookup import agents
from environment import Gym2OpEnv
from models.decision_transformer import DecisionTransformer
from training.seq_trainer import SequenceTrainer
from evaluation.evaluate_episodes import evaluate_episode_rtg


if len(sys.argv) != 3:
    print("No impovement and configuration specified")
    exit()

agent_improvement = sys.argv[1]
agent_configuration = int(sys.argv[2])

agent_info = agents[agent_improvement][agent_configuration]
hyperparams = agent_info["hyperparams"]

log = False
save = False

def discounted_returns(timesteps, rewards, discount_factor):
    discounted_rewards = discount_factor**timesteps * rewards
    returns = np.cumsum(discounted_rewards)

    for idx in (np.where(timesteps[1:] == 0)[0]) + 1:
        returns[idx:] -= returns[idx - 1]

    return returns

def load_data(filname, _print=False):
    file = open(filname, "r")
    csvr = csv.reader(file)

    num_sar_tuples = 0

    timesteps = []
    states = []
    actions = []
    rewards = []

    for line in csvr:
        num_sar_tuples += 1

        timestep = int(line[0])
        observation = np.array(line[1][2:-1].split(), dtype=np.float32)
        action = np.array(line[2][1:-1].split(), dtype=np.float32)
        reward = float(line[3])

        timesteps.append(timestep)
        states.append(observation)
        actions.append(action)
        rewards.append(reward)

    if _print:
        print(f"Loaded {num_sar_tuples} sar tuples")
    timesteps = np.array(timesteps)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    if _print:
        print(f"State Dim: {states.shape}")
        print(f"Action Dim: {actions.shape}")
        print(f"Reward Dim: {rewards.shape}")

    return timesteps, states, actions, rewards

def get_batch(batch_size=256, max_len=agent_info["architecture"]["num_tokens"]):
    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p= traj_lens / num_timesteps # each trajectory is likely to be chosen by how long it is
    )

    s, a, r, d, rtg, t, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        traj_i = int(batch_inds[i])
        traj_start = trajectories[traj_i]
        traj_end = traj_start + traj_lens[traj_i]
        si = random.randint(traj_start, traj_end)

        if si + max_len > traj_end:
            overflow = si + max_len - traj_end
            s.append(np.concatenate([np.zeros((1, overflow, state_dim)), states[si:traj_end].reshape(1, -1, state_dim)], axis=1))
            a.append(np.concatenate([np.zeros((1, overflow, act_dim)), actions[si:traj_end].reshape(1, -1, act_dim)], axis=1))
            r.append(np.concatenate([np.zeros((1, overflow, 1)), rewards[si:traj_end].reshape(1, -1, 1)], axis=1))
            rtg.append(np.concatenate([np.zeros((1, overflow, 1)), returns[si:traj_end].reshape(1, -1, 1)], axis=1))
            t.append(np.concatenate([np.zeros((1, overflow)), timesteps[si:traj_end].reshape(1, -1)], axis=1))
            d.append(np.ones((1, max_len)))
            mask.append(np.zeros((1, max_len)))

            d[-1][traj_end - si:] = 0
            mask[-1][traj_end - si:] = 1
        else:
            s.append(states[si:si + max_len].reshape(1, -1, state_dim))
            a.append(actions[si:si + max_len].reshape(1, -1, act_dim))
            r.append(rewards[si:si + max_len].reshape(1, -1, 1))
            rtg.append(returns[si:si + max_len].reshape(1, -1, 1))
            d.append(np.zeros((1, max_len)))
            t.append(timesteps[si:si + max_len].reshape(1, -1))
            mask.append(np.ones((1, max_len)))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    t = torch.from_numpy(np.concatenate(t, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a, r, d, rtg, t, mask

def eval_episodes(target_rew):
    def fn(model):
        returns, lengths = [], []
        for _ in range(hyperparams["eval_episodes"]):
            with torch.no_grad():
                ret, length, = evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=hyperparams["max_episode_len"],
                    scale=1,
                    target_return=target_rew/1,
                    mode="normal",
                    device=device
                )
            returns.append(ret)
            lengths.append(length)
        return {
            f'target_{target_rew}_return_mean': np.mean(returns),
            f'target_{target_rew}_return_std': np.std(returns),
            f'target_{target_rew}_length_mean': np.mean(lengths),
            f'target_{target_rew}_length_std': np.std(lengths),
        }
    return fn


env = Gym2OpEnv(
    modifying_bus_count=agent_info["data"]["modifiable_buses"],
    modifying_line_count=agent_info["data"]["modifiable_lines"],
    curtail_bin_counts=agent_info["data"]["redispatch_bins"],
    redispatch_bin_counts=agent_info["data"]["curtail_bins"]
)

timesteps, states, actions, rewards = load_data(agent_info["meta"]["data_file"], _print=False)
trajectories = np.where(timesteps == 0)[0]
num_trajectories = len(trajectories)
traj_lens = np.append(timesteps[trajectories[1:] - 1] + 1, timesteps[-1] + 1)
returns = discounted_returns(timesteps, rewards, hyperparams["discount_factor"])

state_dim = env.observation_space.shape[0]
act_dim = len(env.action_space.sample())

num_timesteps = sum(traj_lens)
max_episode_len = agent_info["hyperparams"]["max_episode_len"]
device = agent_info["meta"]["device"]

print('=' * 50)
print(f'Starting new experiment: Powergrid {agent_improvement}')
print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
print('=' * 50)


model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=agent_info["architecture"]["num_tokens"],
        max_ep_len=max_episode_len,
        hidden_size=agent_info["architecture"]["embed_dim"],
        n_layer=agent_info["architecture"]["num_attention_embedding_layers"],
        n_head=agent_info['architecture']["num_attention_heads"],
        n_inner=4 * agent_info["architecture"]["embed_dim"],
        activation_function="relu",
        n_positions=1024,
        resid_pdrop=hyperparams["dropout"],
        attn_pdrop=hyperparams["dropout"], # dropout
        action_relu=True
    )
model.train()
model = model.to(device='cuda')

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=hyperparams['learning_rate'],
    weight_decay=hyperparams['weight_decay'],
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda steps: min((steps+1)/ hyperparams["warmup_steps"] , 1)
)

trainer = SequenceTrainer(
    model=model,
    optimizer=optimizer,
    batch_size=hyperparams["batch_size"],
    get_batch=get_batch,
    scheduler=scheduler,
    loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
    eval_fns=[eval_episodes(tar) for tar in [hyperparams["target_return"]]],
)


if log:
    wandb.init(
        name=f"{agent_improvement}-{agent_configuration}",
        group=f"{agent_improvement}-powergrid",
        project='rl-powergrid',
        config=agent_info
    )

for i in range(agent_info["meta"]["training_iterations"]):
    outputs = trainer.train_iteration(
        num_steps=agent_info["meta"]["training_steps"],
        iter_num=i + 1,
        print_logs=True
    )

    if log:
        wandb.log(outputs)

if save:
    torch.save(model.state_dict(), agent_info["meta"]["model_file"])
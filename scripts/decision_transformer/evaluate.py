import sys

import torch
import numpy as np

from agent_lookup import agents
from environment import Gym2OpEnv
from models.decision_transformer import DecisionTransformer

if len(sys.argv) != 3:
    print("No impovement and configuration specified")
    exit()

evaluation_episodes = 5

agent_improvement = sys.argv[1]
agent_configuration = int(sys.argv[2])

agent_info = agents[agent_improvement][agent_configuration]

env = Gym2OpEnv(
    modifying_bus_count=agent_info["data"]["modifiable_buses"],
    modifying_line_count=agent_info["data"]["modifiable_lines"],
    curtail_bin_counts=agent_info["data"]["redispatch_bins"],
    redispatch_bin_counts=agent_info["data"]["curtail_bins"]
)

device = 'cuda'
state_dim = env.observation_space.shape[0]
act_dim = len(env.action_space.sample())
max_episode_len = agent_info["hyperparams"]["max_episode_len"]

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
        resid_pdrop=agent_info["hyperparams"]["dropout"],
        attn_pdrop=agent_info["hyperparams"]["dropout"], # dropout
        action_relu=True
    )
model.load_state_dict(torch.load(agent_info["meta"]["model_file"]))
model.eval()
model = model.to(device=device)

episode_returns = np.zeros(evaluation_episodes)
episode_lengths = np.zeros(evaluation_episodes)
for ep_i in range(evaluation_episodes):
    state, info = env.reset()

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    target_return = torch.tensor(agent_info["hyperparams"]["target_return"], device=device, dtype=torch.float32).reshape(1, 1)

    for t in range(max_episode_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy().astype(dtype=np.int64)

        state, reward, terminated, truncated, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_returns[ep_i] += reward
        episode_lengths[ep_i] += 1

        if terminated or truncated:
            break

    del states
    del actions
    del rewards
    del timesteps
    del target_return

print("Mean Length:", np.mean(episode_lengths))
print("Mean Return:", np.mean(episode_returns))
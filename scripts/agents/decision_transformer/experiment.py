import numpy as np
import torch
import wandb

import random

from evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from models.decision_transformer import DecisionTransformer
from training.seq_trainer import SequenceTrainer
from load_dataset import load_data
import util as dtutil

from env import Gym2OpEnv

def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    dataset = variant['dataset']
    model_type = 'decision-transformer'
    group_name = f'{exp_prefix}-powergrid-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    env = Gym2OpEnv()
    max_ep_len = 10 * 24 * 60 // 5
    env_targets = [1000] # what is this?
    scale = 1.

    # load dataset
    timesteps, states, actions, rewards = load_data(f"./data/trajectories/{dataset}.csv", _print=False)
    trajectories = np.where(timesteps == 0)[0]
    num_trajectories = len(trajectories)
    traj_lens = np.append(timesteps[trajectories[1:] - 1] + 1, timesteps[-1] + 1)
    returns = dtutil.discounted_returns(timesteps, rewards, variant["discount_factor"])


    state_dim = states.shape[1]
    act_dim = actions.shape[1]

    # used for input normalization
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: Powergrid {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']

    def get_batch(batch_size=256, max_len=K):
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

            # get sequences from dataset
            # s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            # a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            # r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            # if 'terminals' in traj:
            #     d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            # else:
            #     d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            # timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            # timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            # rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            # if rtg[-1].shape[1] <= s[-1].shape[1]:
            #     rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # # padding and state + reward normalization
            # tlen = s[-1].shape[1]
            # s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            # a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            # r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            # d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            # rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            # timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            # mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

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
            num_illegal, num_ambiguous = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length, n_illegal, n_ambiguous = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode="normal",
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
                num_illegal.append(n_illegal)
                num_ambiguous.append(n_ambiguous)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'target_{target_rew}_num_illegal_mean': np.mean(num_illegal),
                f'target_{target_rew}_num_illegal_std': np.std(num_illegal),
                f'target_{target_rew}_num_ambiguous_mean': np.mean(num_ambiguous),
                f'target_{target_rew}_num_ambiguous_std': np.std(num_ambiguous),
            }
        return fn

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
        action_relu=True
    )

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='rl-powergrid',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    # for eval_fn in trainer.eval_fns:
    #     outputs = eval_fn(trainer.model)
    #     for k, v in outputs.items():
            # logs[f'evaluation/{k}'] = v
            # print(k, v)

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)

    if variant["save"]:
        torch.save(model.state_dict(), f"./scripts/agents/decision_transformer/saved_models/{exp_prefix}.plt")


if __name__ == '__main__':
    variant = {
        "dataset": "reduced_action_masked_random_0.8",
        "K": 64, # Number of tokens
        "batch_size": 256,
        "embed_dim": 128, # Dimension of embedding layer in transformer
        "n_layer": 4, # number of attention-embedding layers
        "n_head": 8, # number of heads in each attention layer
        "activation_function": "relu",
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "discount_factor": 0.99,
        "warmup_steps": 50,
        "num_eval_episodes": 5,
        "max_iters": 10, # num epochs
        "num_steps_per_iter": 1000,
        "device": "cuda",
        "log_to_wandb": True,
        "save": True,
    }
    experiment('gym-experiment', variant=variant)
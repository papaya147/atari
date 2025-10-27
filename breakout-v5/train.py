import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import ale_py
from network import Network
import torch
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer
from device import device
import random
import wandb
import checkpoint

# * hyper params
learning_rate = 0.0001
replay_buffer_size = 1000000
min_replay_buffer_size = 10000
n_episodes = 10000
n_train_steps = 4
epsilon_min = 0.1
epsilon_decay_steps = 2000000
batch_size = 32
gamma = 0.99
target_network_update_steps = 10000
# * hyper params

gym.register_envs(ale_py)
torch.set_num_threads(12)
torch.set_num_interop_threads(12)

env = gym.make('ALE/Breakout-v5', frameskip=4)
env = GrayscaleObservation(env)
env = ResizeObservation(env, shape=(84, 84))
env = FrameStackObservation(env, stack_size=4)
obs, info = env.reset()

def train_step(network, target_network, samples, optimizer, gamma=0.99):
    states, actions, rewards, next_states, dones = samples

    q_values = network(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_states)
        max_next_q_values = next_q_values.max(1)[0]

        targets = rewards + gamma * max_next_q_values * (1 - dones)

    loss = F.huber_loss(q_values, targets)

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)

    optimizer.step()

    return loss.item()

network = Network(obs.shape[0], env.action_space.n).to(device)
target_network = Network(obs.shape[0], env.action_space.n).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

checkpoint_dir = 'breakout-v5/checkpoints'
model_state = checkpoint.load_latest(checkpoint_dir)
if model_state:
    episode = model_state['episode']
    total_steps = model_state['total_steps']
    network.load_state_dict(model_state['network'])
    target_network.load_state_dict(model_state['target_network'])
    optimizer.load_state_dict(model_state['optimizer'])
    replay_buffer = model_state['replay_buffer']

    run = wandb.init(
        project='atari-breakout-v5',
        id=model_state['run_id'],
        resume='must',
        config={
            'learning_rate': learning_rate,
            'replay_buffer_size': replay_buffer_size,
            'min_replay_buffer_size': min_replay_buffer_size,
            'n_episodes': n_episodes,
            'n_train_steps': n_train_steps,
            'epsilon_min': epsilon_min,
            'epsilon_decay_steps': epsilon_decay_steps,
            'batch_size': batch_size,
            'gamma': gamma,
            'target_network_update_steps': target_network_update_steps,
        }
    )
else:
    episode = 0
    total_steps = 0
    target_network.load_state_dict(network.state_dict())
    replay_buffer = ReplayBuffer(replay_buffer_size, pct_recent=0.2, pct_recent_util=0.3)
    
    run = wandb.init(project='atari-breakout-v5', config={
        'learning_rate': learning_rate,
        'replay_buffer_size': replay_buffer_size,
        'min_replay_buffer_size': min_replay_buffer_size,
        'n_episodes': n_episodes,
        'n_train_steps': n_train_steps,
        'epsilon_min': epsilon_min,
        'epsilon_decay_steps': epsilon_decay_steps,
        'batch_size': batch_size,
        'gamma': gamma,
        'target_network_update_steps': target_network_update_steps,
    })
    
run_id = run.id
loss = '-'

for e in range(episode, n_episodes):
    epsilon = max(epsilon_min, 1.0 - (total_steps / epsilon_decay_steps) * (1.0 - epsilon_min))

    total_reward = 0
    env.reset()

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    state, _, done, _, _ = env.step(1) # FIRE before start
    state = state.astype(np.uint8)

    steps = 0
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = network(torch.tensor(state / 255.0, dtype=torch.float32).unsqueeze(0).to(device))
                action = q_values.argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        next_state = next_state.astype(np.uint8)
        reward = np.clip(reward, -1, 1)
        total_reward += reward
        total_steps += 1

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

        if replay_buffer.size() > min_replay_buffer_size:
            for _ in range(n_train_steps):
                samples = replay_buffer.sample(batch_size)
                loss = train_step(network, target_network, samples, optimizer, gamma)

        if total_steps % target_network_update_steps == 0:
            target_network.load_state_dict(network.state_dict())

        state = next_state
        steps += 1

    replay_buffer.add_episode(states, actions, rewards, next_states, dones)

    run.log({
        'episode': e,
        'steps': steps,
        'epsilon': epsilon,
        'loss': loss,
        'total_reward': total_reward,
        'replay_buffer_size': replay_buffer.size()
    })

    model_state = {
        'episode': e,
        'total_steps': total_steps,
        'network': network.state_dict(),
        'target_network': target_network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'replay_buffer': replay_buffer,
        'run_id': run_id
    }

    checkpoint.new(checkpoint_dir, e, model_state)
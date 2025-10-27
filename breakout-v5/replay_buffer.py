from collections import deque
import random
import torch
import numpy as np
from device import device

class ReplayBuffer:
    def __init__(self, cap, pct_recent=0.1, pct_recent_util=0.8, n_frames=4):
        self.cap = cap
        self.pct_recent = pct_recent
        self.pct_recent_util = pct_recent_util
        self.n_frames = n_frames
        self.episodes = []
        self.frames = []
        self.actions = []
        self.rewards = []
        self.next_frames = []
        self.dones = []
        
    def add_episode(self, states, actions, rewards, next_states, dones):
        # assumes that states are sequential within this episode for n_frames = 4: (S1: 1, 2, 3, 4 ; S2: 2, 3, 4, 5 ; S3: 3, 4, 5, 6 ; ...)

        # mark episode start
        episodes = np.zeros(len(dones), dtype=np.uint8)
        episodes[0] = 1

        # for the first sample, add first n-1 frames
        for i in range(self.n_frames - 1):
            self.frames.append(states[0][i])
            self.next_frames.append(next_states[0][i])

        # for all samples, add only the last frame
        for i in range(len(states)):
            self.frames.append(states[i][-1])
            self.next_frames.append(next_states[i][-1])

        self.episodes.extend(episodes.tolist())
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.dones.extend(dones)

        # prune old
        if len(self.dones) > self.cap:
            n_prune = len(self.dones) - self.cap

            episodes_removed = int(np.sum(self.episodes[:n_prune]))

            self.episodes = self.episodes[n_prune:]

            if self.episodes[0] == 0: # old episode still retained
                episodes_removed -= 1
                self.episodes[0] = 1 # mark beginning of episode

            frames_removed = n_prune + (self.n_frames - 1) * episodes_removed
            self.frames = self.frames[frames_removed:]
            self.next_frames = self.next_frames[frames_removed:]

            self.actions = self.actions[n_prune:]
            self.rewards = self.rewards[n_prune:]
            self.dones = self.dones[n_prune:]

    def build_sample_frames(self, samples):
        sample_frames = []
        sample_next_frames = []

        for n in samples:
            episodes = int(np.sum(self.episodes[:n]))
            if self.episodes[n] == 0:
                episodes -= 1

            frames_start = n + (self.n_frames - 1) * episodes
            frames_end = frames_start + self.n_frames

            frames = self.frames[frames_start:frames_end]
            next_frames = self.next_frames[frames_start:frames_end]

            sample_frames.append(frames)
            sample_next_frames.append(next_frames)

        return np.array(sample_frames, dtype=np.uint8), np.array(sample_next_frames, dtype=np.uint8)

    def sample(self, batch_size):
        if len(self.episodes) < batch_size:
            samples = random.sample(range(len(self.episodes)), batch_size)
        else:
            n_recent_samples = int(batch_size * self.pct_recent_util)

            recent_size = int(len(self.episodes) * self.pct_recent)
            recent_end = len(self.episodes)
            recent_start = recent_end - recent_size

            n_recent_samples = min(n_recent_samples, recent_size)
            recent_samples = random.sample(range(recent_start, recent_end), n_recent_samples)

            n_old_samples = batch_size - n_recent_samples

            old_start = 0
            old_end = recent_start

            n_old_samples = min(n_old_samples, old_end - old_start)
            old_samples = random.sample(range(old_start, old_end), n_old_samples)

            samples = recent_samples + old_samples
            random.shuffle(samples)

        states, next_states = self.build_sample_frames(samples)
        actions = np.array(self.actions)[samples]
        rewards = np.array(self.rewards)[samples]
        dones = np.array(self.dones)[samples]
        
        return (
            torch.tensor(states / 255.0, dtype=torch.float32).to(device),
            torch.LongTensor(actions).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(next_states / 255.0, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device)
        )

    def size(self):
        return len(self.episodes)
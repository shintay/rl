from collections import namedtuple, deque
import numpy as np
import random
import copy


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=[
            'state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        xp = self.experience(state, action, reward, next_state, done)
        self.memory.append(xp)

    def sample(self, batch_size=64):
        return random.sample(self.memory, k=batch_size)

    def size(self):
        return len(self.memory)

    def reset(self):
        self.memory.clear()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

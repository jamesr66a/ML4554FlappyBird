import numpy as np

class ReplayMemory:
  def __init__(N):
    self.N = N

    self.memory = [] # FIFO queue

  def insert(xt, at, rt, xt_1):
    self.memory.append((xt, at, rt, xt_1))

    if len(self.memory) > self.N:
      self.memory.pop(0)

  def sample(n):
    if n > self.N:
      raise Exception('Sample length must be <= memory capacity')

    if len(self.memory) < n:
      return None

    idxs = np.random.choice(xrange(N), n, replace=False)
    return [self.memory[idx] for idx in idxs]


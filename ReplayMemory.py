import numpy as np

class ReplayMemory:
  def __init__(self, N):
    self.N = N

    self.memory = [] # FIFO queue

  def insert(self, xt, at, rt, xt_1):
    if rt != 0 or np.random.random() < 0.05:
      a = np.zeros((1, 2))
      a[0, at] = 1.0
      self.memory.append((xt, a, rt, xt_1))

      if len(self.memory) > self.N:
        self.memory.pop(0)

  def sample(self, n):
    if n > self.N:
      raise Exception('Sample length must be <= memory capacity')

    if len(self.memory) < n:
      return None

    idxs = np.random.choice(xrange(len(self.memory)), n, replace=False)
    return [self.memory[idx] for idx in idxs]

if __name__=='__main__':
  rp = ReplayMemory(100)

  for i in xrange(100):
    rp.insert(
      np.random.randn(4, 28, 28, 3),
      np.random.randint(2),
      np.random.randint(100),
      np.random.randn(4, 28, 28, 3)
    )

  print(rp.sample(10))

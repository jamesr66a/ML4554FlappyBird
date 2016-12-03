from FlappyBirdGame import FlappyBirdGame
from ReplayMemory import ReplayMemory
from QNetwork import QNetwork

import numpy as np
import tensorflow as tf

ep_prob = 0.1

def sample_action():
  return np.random.randint(0, 2)

display = True

def step_wrapper(game, action):
  frames, reward, terminal, _ = game.step(action, display=display)
  frames = frames[-5:-1, :, :]
  frames = np.transpose(frames, [1, 2, 0])
  frames = np.expand_dims(frames, 0)
 
  return frames, reward, terminal

with tf.Session() as sess:
  rp = ReplayMemory(1000)
  learning_rate = 5e-2
  Q = QNetwork(84, 4, 2, learning_rate=learning_rate)
  Qhat = QNetwork(84, 4, 2, learning_rate=learning_rate)
  sess.run(tf.initialize_all_variables())
  Qhat.copy_params(Q, sess)

  game = FlappyBirdGame()

  for episode in xrange(1000000):
    t = 0
    game.reset()
    frames, reward, terminal = step_wrapper(game, sample_action())
    while not terminal:
      ep = np.random.random(1)
      if ep < ep_prob:
        action = sample_action()
      else:
        action = np.argmax(Q.predict(frames, sess))

      frames_new, reward, terminal = step_wrapper(game, action)
      rp.insert(frames, action, reward, frames_new)
      frames = frames_new

      mb_size = 10
      minibatch = rp.sample(mb_size)
      if minibatch is not None:
        mb_data = np.zeros((mb_size, 84, 84, 4))
        mb_data_1 = np.zeros((mb_size, 84, 84, 4))
        mb_actions = np.zeros((mb_size, 1), dtype=np.int32)
        mb_rewards = np.zeros((mb_size, 1))
        for idx, (x, a, r, x1) in enumerate(minibatch):
          mb_data[idx, :, :, :] = x
          mb_data_1[idx, :, :, :] = x1
          mb_actions[idx] = a
          mb_rewards[idx] = r

        Qhat_preds = Qhat.predict(mb_data_1, sess)[0]
        max_future_reward = np.amax(Qhat_preds, axis=1)

        ys = np.concatenate((mb_rewards, mb_rewards), axis=1)
        for idx, action in enumerate(mb_actions):
          ys[idx, action] += max_future_reward[idx]

        loss = Q.train(mb_data, ys, sess)
        print('loss', loss)

      t = t + 1
      if t % 100 == 0:
        Qhat.copy_params(Q, sess)


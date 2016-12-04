from FlappyBirdGame import FlappyBirdGame
from ReplayMemory import ReplayMemory
from QNetwork import QNetwork

import numpy as np
import os
import tensorflow as tf

ep_prob = 0.1

def sample_action():
  return np.random.randint(0, 2)

display = True

def step_wrapper(game, action):
  frames, reward, terminal, _ = game.step(action, display=display, steps=4)
  frames = np.transpose(frames, [1, 2, 0])
  frames = np.expand_dims(frames, 0)
 
  return frames, reward, terminal

is_train = False

with tf.Session() as sess:
  rp = ReplayMemory(1000000)
  learning_rate = 0.005
  Q = QNetwork(84, 4, 2, learning_rate=learning_rate)
  Qhat = QNetwork(84, 4, 2, learning_rate=learning_rate)
  sess.run(tf.global_variables_initializer())
  Qhat.copy_params(Q, sess)

  game = FlappyBirdGame()

  saver = tf.train.Saver()

  if os.path.exists('./model.ckpt.index'):
    saver.restore(sess, './model.ckpt')
    print('Model restored from ./model.ckpt')

  t = 0
  try:
    for episode in xrange(1000000):
      game.reset()
      frames, reward, terminal = step_wrapper(game, sample_action())
      while not terminal:
        ep = np.random.random(1)
        if is_train and ep < ep_prob*np.exp(-t/100000.):
          action = sample_action()
        else:
          action = np.argmax(Q.predict(frames, sess))
          if not is_train:
            print(Q.predict(frames, sess))

        frames_new, reward, terminal = step_wrapper(game, action)
        rp.insert(frames, action, reward, frames_new)
        frames = frames_new

        mb_size = 32
        minibatch = rp.sample(mb_size)
        if is_train and minibatch is not None and t % 4 == 0:
          mb_size = len(minibatch)
          print(mb_size)
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

          ys = Q.predict(mb_data, sess)[0]
          for idx, action in enumerate(mb_actions):
            ys[idx, action] = mb_rewards[idx] + 0.99*max_future_reward[idx]

          loss = Q.train(mb_data, ys, sess)
          print('iteration', t, 'loss', loss, 'reward', reward, 'ep', np.exp(-t/100000.))

        t = t + 1
        if t % 100 == 0:
          Qhat.copy_params(Q, sess)

        if not is_train:
          import time
          time.sleep(1/30.)
  except KeyboardInterrupt:
    pass

  print(saver.save(sess, './model.ckpt'))

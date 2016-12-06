from FlappyBirdGame import FlappyBirdGame
from ReplayMemory import ReplayMemory
from QNetwork import QNetwork

import cPickle as pickle
import numpy as np
import os
import tensorflow as tf

ep_prob = 0.5

def sample_action():
  return np.random.randint(0, 2)

display = True

def step_wrapper(game, action):
  frames, reward, terminal, _ = game.step(action, display=display, steps=6)
  frames = np.transpose(frames, [1, 2, 0])
  frames = np.expand_dims(frames, 0)
 
  return frames, reward, terminal

is_train = False

with tf.Session() as sess:
  H = 6
  rp = ReplayMemory(1000)
  learning_rate = 0.5
  Q = QNetwork(84, H, 2, learning_rate=learning_rate)
  Qhat = QNetwork(84, H, 2, learning_rate=learning_rate)
  sess.run(tf.global_variables_initializer())
  Qhat.copy_params(Q, sess)

  game = FlappyBirdGame()

  saver = tf.train.Saver()

  if os.path.exists('./model.ckpt.index') and os.path.exists('./model.ckpt.rm'):
    saver.restore(sess, './model.ckpt')
    if is_train:
      with open('./model.ckpt.rm') as f:
        rp = pickle.load(f)
    print('Model restored from ./model.ckpt')

  t = 0
  try:
    while True:
      game.reset()
      frames, reward, terminal = step_wrapper(game, sample_action())
      while not terminal:
        ep = np.random.random(1)
        if is_train and ep < ep_prob*np.exp(-t/100000.):
          action = sample_action()
        else:
          action = np.argmax(Q.predict(frames, sess))
          if not is_train:
            pred = Q.predict(frames, sess)
            print(np.argmax(pred), pred)

        frames_new, reward, terminal = step_wrapper(game, action)
        rp.insert(frames, action, reward, frames_new)
        frames = frames_new

        mb_size = 32
        minibatch = rp.sample(mb_size)
        if is_train and minibatch is not None:
          mb_size = len(minibatch)
          mb_data = np.zeros((mb_size, 84, 84, H))
          mb_data_1 = np.zeros((mb_size, 84, 84, H))
          mb_actions = np.zeros((mb_size, 2))
          mb_rewards = np.zeros((mb_size, 1))
          for idx, (x, a, r, x1) in enumerate(minibatch):
            mb_data[idx, :, :, :] = x
            mb_data_1[idx, :, :, :] = x1
            mb_actions[idx, :] = a
            mb_rewards[idx] = r

          Qhat_preds = Qhat.predict(mb_data_1, sess)[0]
          max_future_reward = np.amax(Qhat_preds, axis=1)

          ys = np.zeros((mb_size, 1))
          for idx, reward in enumerate(mb_rewards):
            ys[idx] = reward + 0.99*max_future_reward[idx]

          loss = Q.train(mb_data, ys, mb_actions, sess)
          print(
            'iteration', t, 'loss', loss,
            'reward', reward[0], 'ep', np.exp(-t/100000.),
            'avg y', np.average(ys)
          )

        t = t + 1
        if t % 10000 == 0:
          Qhat.copy_params(Q, sess)

        if not is_train:
          import time
          time.sleep(1/30.)
  except KeyboardInterrupt:
    pass

  print(saver.save(sess, './model.ckpt'))
  if is_train:
    with open('./model.ckpt.rm', 'w') as f:
      pickle.dump(rp, f)

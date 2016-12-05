import numpy as np
import tensorflow as tf
import util

# x - DxDxH input, D := dimension, H := history length
# y - Q values \in \mathbb{R}^A
# b - Preprocessing map
# c - 32 8x8 filters, stride 4
# d - ReLU
# e - 64 4x4 filters, stride 2
# f - ReLU
# g - 64 3x3 filters, stride 1
# h - ReLU
# i - 512-unit FC ReLU
# out - A-unit FC linear, A := number of actions
class QNetwork:
  def conv_layer(self, x, shape, stride, cl):
    key = 'kernel{}'.format(shape[0])
    with tf.name_scope(key) as scope:
      self.vars[key] = tf.Variable(
        tf.truncated_normal(shape, stddev=0.01), dtype=tf.float32
      )
   
      self.arch[cl] = tf.nn.conv2d(
        x, self.vars[key], [1, stride, stride, 1],
        padding='SAME' # TODO: check padding
      )
      return tf.nn.relu(self.arch[cl])

  def fc_layer(self, x, outdim, name, relu=True):
    with tf.name_scope(name) as scope:
      shape = x.get_shape().as_list()
      dim = 1
      for d in shape[1:]:
        dim *= d
      x = tf.reshape(x, [-1, dim])

      weights = self.vars[name+'W'] = tf.Variable(
        tf.truncated_normal([dim, outdim], stddev=0.01), dtype=tf.float32
      )
      biases = self.vars[name+'B'] = tf.Variable(
        tf.truncated_normal([outdim], stddev=0.01), dtype=tf.float32
      )

      if relu:
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights), biases))
      else:
        return tf.nn.bias_add(tf.matmul(x, weights), biases)
      
 
  def __init__(self, D, H, A, learning_rate=0.000025):
    self.D = D
    self.H = H
    self.A = A
    self.learning_rate = learning_rate

    self.vars = {}

    self.arch = {}
    self.arch['x'] = tf.placeholder(dtype=tf.float32, shape=(None, D, D, H))
    self.arch['y'] = tf.placeholder(dtype=tf.float32, shape=(None))
    self.arch['action'] = tf.placeholder(dtype=tf.float32, shape=(None, A))
    self.arch['b'] = self.arch['x'] # TODO

    
    l1 = 32
    l2 = 64
    fc = 256
    with tf.name_scope('layers') as scope:
      self.arch['d'] = self.conv_layer(self.arch['b'], [8, 8, 4, l1], 4, 'c')
      self.arch['f'] = self.conv_layer(self.arch['d'], [4, 4, l1, l2], 2, 'e')
      self.arch['h'] = self.conv_layer(self.arch['f'], [3, 3, l2, l2], 1, 'g')

      self.arch['i'] = self.fc_layer(self.arch['h'], 256, 'fc1')
      self.arch['out'] = self.fc_layer(self.arch['i'], A, 'fc2', relu=False)

    with tf.name_scope('optimization') as scope:
      self.arch['readout'] = tf.reduce_sum(
        tf.mul(self.arch['out'], self.arch['action']), axis=1
      )

      self.arch['loss'] = tf.reduce_mean(
       tf.square(self.arch['readout'] - self.arch['y'])
      )

      #self.arch['optim'] = tf.train.RMSPropOptimizer(
      #  learning_rate, momentum=0.95,decay=0.95
      #).minimize(self.arch['loss'])
      self.arch['optim'] = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.arch['loss'])

  def train(self, x, y, actions, sess):
    _, loss = sess.run([
      self.arch['optim'], self.arch['loss']
    ],feed_dict={
        self.arch['x']: x, self.arch['y']: y, self.arch['action']: actions
      })
    return loss

  def predict(self, x, sess):
    pred = sess.run([
      self.arch['out']
    ], feed_dict={self.arch['x']: x})
    return pred

  def copy_params(self, other, sess):
    ops = []
    for key, val in self.vars.iteritems():
      ops.append(tf.assign(val, other.vars[key]))
    sess.run(ops)

if __name__=='__main__':
  """
  qn = QNetwork(84, 4, 2)
  qn2 = QNetwork(84, 4, 2)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    first = np.concatenate((np.ones((100, 1)), np.zeros((100, 1))), axis=1);
    second = np.concatenate((np.zeros((100, 1)), np.ones((100, 1))), axis=1);
    qn.train(
      np.random.randn(100, 84, 84, 4),
      np.random.randn(100, 1),
      first,
      sess
    )
    test_x = np.random.randn(1, 84, 84, 4)
    print(qn.predict(test_x, sess))
    print(qn2.predict(test_x, sess))
    qn2.copy_params(qn, sess)
    print(qn2.predict(test_x, sess))
  """

  with tf.Session() as sess:
    qn = QNetwork(84, 4, 2)
    sess.run(tf.global_variables_initializer())
    data = np.concatenate((np.zeros((100, 84, 84, 4)), np.ones((100, 84, 84, 4))))
    print(data)
    labels = np.concatenate((np.ones((100)), -1.*np.ones((100))))
    print(labels)
    kek = np.ones((200, 2))
    writer = tf.train.SummaryWriter("test", sess.graph)
    for i in xrange(1000):
      qn.train(data, labels, kek, sess)

    print(qn.predict(np.zeros((1, 84, 84, 4)), sess))
    print(qn.predict(np.ones((1, 84, 84, 4)), sess))

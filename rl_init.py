import tensorflow as tf
import numpy as np
from noise import noise
import itertools
from operator import itemgetter
from env1 import ArmEnv

#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.q = tf.placeholder(tf.float32, [None, 1], 'q')
        #self.action_bound = [50, 50, 50, 50]
        self.sample_all_actions()

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            #self.a= tf.multiply(unbounded_a, self.a_bound)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        print "after actor init"
        #with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            #q = self._build_c(self.S, self.a, scope='eval', trainable=False)
            #q = self.calcQ(self.S[:6], self.S[6:9], self.S[9:12], self.a)
            #q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        #self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        #self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea)]
                             for ta, ea in zip(self.at_params, self.ae_params)]

        #q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        #td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        #self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        #self.a_grads = tf.gradients(q, self.a)[0]
        #self.actor_grads = tf.gradients(self.a, self.ae_params, -self.a_grads)
        #q = self.calcQ(self.S[:6], self.S[6:9], self.S[9:12], self.a)

        a_loss = - tf.reduce_mean(self.q)    # maximize the q

        #self.atrain = tf.train.AdamOptimizer(LR_A).apply_gradients(zip(self.actor_grads, self.ae_params))
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        raw_action = self.sess.run(self.a, {self.S: s[None, :]})[0]
        return raw_action

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        #br = bt[:, -self.s_dim - 1: -self.s_dim]
        #bs_ = bt[:, -self.s_dim:]
        bq = []
        for i in range(len(bs)):
            s = bs[i]
            a = ba[i]
            q = self.calcQ(s[:6], s[6:9], s[9:12], a)
            bq.append(q)
        bq = np.array(bq)

        self.sess.run(self.atrain, {self.S: bs, self.q : bq})
        #self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 64, activation=tf.nn.relu, name='l1', trainable=trainable)
            net1 = tf.layers.dense(net,64, activation = tf.nn.relu, name = 'l2', trainable = trainable)
            a = tf.layers.dense(net1, self.a_dim, activation=tf.nn.relu, name='a', trainable=trainable)
            bounded_a = self.a_bound[0] + tf.nn.sigmoid(a) * (self.a_bound[1] - self.a_bound[0])
            return bounded_a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net1 = tf.layers.dense(net, n_l1, activation=tf.nn.tanh, name = 'net1', trainable = trainable)
            return tf.layers.dense(net1, 1, activation = tf.nn.tanh, trainable=trainable)  # Q(s,a)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')

    def frange(self, start, end, step):
        res = []
        i = start
        while i < end:
            res.append(i)
            i += step
        return res

    def sample_all_actions(self):
        action_setp = (self.a_bound[1] - self.a_bound[0]) / 10
        oned_sample_action = self.frange(self.a_bound[0], self.a_bound[1], action_setp)
        self.sample_actions = []
        for action in itertools.product(oned_sample_action, oned_sample_action, oned_sample_action, oned_sample_action):
            self.sample_actions.append(action)

    def calcQ(self, state, linear_acc, angular_acc, real_action):
        sim = ArmEnv(state[:3], state[3:6], linear_acc, angular_acc)
        dic = {}
        min_action_dist = self.a_bound[1]* self.a_dim
        nearest_action = [0] * self.a_dim
        goal = np.array([sim.goal['x'], sim.goal['y'], sim.goal['z']])
        for action in self.sample_actions:
            action_dist = np.linalg.norm(real_action - action)
            if action_dist < min_action_dist:
                nearest_action = action
                min_action_dist = action_dist
            next_state = sim.step(action)
            # if move > 0 means closer to goal
            move = np.linalg.norm(goal - state) - np.linalg.norm(goal - next_state)
            dic[action] = move
            sim.reset()


        idx = 1
        for action, move in sorted(dic.items(), key = itemgetter(1)):
            if action == nearest_action:
                return idx
            idx += 1

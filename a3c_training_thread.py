import tensorflow as tf
import numpy as np
import random
import vtrace
import utils
import time
import sys

def network(x, num_action):
    x = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    actor = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    actor = tf.layers.dense(inputs=actor, units=256, activation=tf.nn.relu)
    actor = tf.layers.dense(inputs=actor, units=num_action, activation=tf.nn.softmax)
    critic = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    critic = tf.layers.dense(inputs=critic, units=256, activation=tf.nn.relu)
    critic = tf.squeeze(tf.layers.dense(inputs=critic, units=1, activation=None), axis=1)
    
    return actor, critic

def build_model(state, next_state, input_shape, output_size, unroll):
    state = tf.reshape(state, [-1, *input_shape])
    next_state = tf.reshape(next_state, [-1, *input_shape])

    with tf.variable_scope('impala'):
        policy, value = network(state, output_size)
    with tf.variable_scope('impala', reuse=True):
        _, next_value = network(next_state, output_size)

    policy = tf.reshape(policy, [-1, unroll, output_size])
    value = tf.reshape(value, [-1, unroll])
    next_value = tf.reshape(next_value, [-1, unroll])

    return policy, value, next_value

class Network(object):
    def __init__(self, input_shape, output_size, unroll, thread_index, device):
        scope_name = "net_" + str(thread_index)
        self.output_size = output_size
        self.input_shape = input_shape
        self.unroll = unroll
        self.discount_factor = 0.99
        self.lr = 0.00025
        self.coef = 0.1

        with tf.device(device), tf.variable_scope(scope_name):
            self.s_ph = tf.placeholder(tf.float32, shape=[None, self.unroll, *self.input_shape])
            self.ns_ph = tf.placeholder(tf.float32, shape=[None, self.unroll, *self.input_shape])
            self.a_ph = tf.placeholder(tf.int32, shape=[None, self.unroll])
            self.d_ph = tf.placeholder(tf.bool, shape=[None, self.unroll])
            self.behavior_policy = tf.placeholder(tf.float32, shape=[None, self.unroll, self.output_size])
            self.r_ph = tf.placeholder(tf.float32, shape=[None, self.unroll])

            self.clipped_reward = tf.clip_by_value(self.r_ph, -1.0, 1.0)

            self.discounts = tf.to_float(~self.d_ph) * self.discount_factor

            self.policy, self.value, self.next_value = build_model(self.s_ph, self.ns_ph, self.input_shape, self.output_size, self.unroll)

            self.transpose_vs, self.transpose_clipped_rho = vtrace.from_softmax(
                behavior_policy_softmax=self.behavior_policy,
                target_policy_softmax=self.policy,
                actions=self.a_ph, discounts=self.discounts, rewards=self.clipped_reward,
                values=self.value, next_value=self.next_value, action_size=self.output_size)

            self.vs = tf.transpose(self.transpose_vs, perm=[1, 0])
            self.rho = tf.transpose(self.transpose_clipped_rho, perm=[1, 0])

            self.vs_ph = tf.placeholder(tf.float32, shape=[None, self.unroll])
            self.pg_advantage_ph = tf.placeholder(tf.float32, shape=[None, self.unroll])

            self.value_loss = vtrace.compute_value_loss(self.vs_ph, self.value)
            self.entropy = vtrace.compute_entropy_loss(self.policy)
            self.pi_loss = vtrace.compute_policy_loss(self.policy, self.a_ph, self.pg_advantage_ph, self.output_size)

            self.total_loss = self.pi_loss + self.value_loss + self.entropy * self.coef
            # self.optimizer = tf.train.RMSPropOptimizer(self.lr, epsilon=0.01, momentum=0.0, decay=0.99)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.total_loss)

class Agent(object):
    def __init__(self, input_shape, output_size, unroll, thread_index, device, global_network, local_network):
        self.thread_index = thread_index
        self.unroll = unroll
        self.output_size = output_size
        self.trajectory = self.unroll + 1
        self.global_network = global_network
        self.local_network = local_network
        self.discount_factor = 0.99

        self.assign_tf = utils.copy_src_to_dst('net_global', 'net_'+str(self.thread_index))

    def set_session(self, sess):
        self.sess = sess

    def assign(self):
        self.sess.run(self.assign_tf)

    def get_policy_and_action(self, state):
        policy = self.sess.run(
            self.local_network.policy,
            feed_dict={self.local_network.s_ph: [[state for i in range(self.unroll)]]})
        policy = policy[0][0]
        action = np.random.choice(self.output_size, p=policy)

        return action, policy, max(policy)

    def update(self, state, next_state, reward, done, action, behavior_policy):
        unrolled_state = np.stack([state[i:i+self.trajectory] for i in range(len(state) - self.trajectory+1)])
        unrolled_next_state = np.stack([next_state[i:i+self.trajectory] for i in range(len(state)-self.trajectory+1)])
        unrolled_reward = np.stack([reward[i:i+self.trajectory] for i in range(len(state)-self.trajectory+1)])
        unrolled_done = np.stack([done[i:i+self.trajectory] for i in range(len(state)-self.trajectory+1)])
        unrolled_behavior_policy = np.stack([behavior_policy[i:i+self.trajectory] for i in range(len(state)-self.trajectory+1)])
        unrolled_action = np.stack([action[i:i+self.trajectory] for i in range(len(state)-self.trajectory+1)])

        unrolled_length = len(unrolled_state)
        sampled_range = np.arange(unrolled_length)
        np.random.shuffle(sampled_range)
        shuffled_idx = sampled_range[:64]

        s_ph = np.stack([unrolled_state[i, 1:] for i in shuffled_idx])
        ns_ph = np.stack([unrolled_next_state[i, 1:] for i in shuffled_idx])
        r_ph = np.stack([unrolled_reward[i, 1:] for i in shuffled_idx])
        d_ph = np.stack([unrolled_done[i, 1:] for i in shuffled_idx])
        b_ph = np.stack([unrolled_behavior_policy[i, 1:] for i in shuffled_idx])
        a_ph = np.stack([unrolled_action[i, 1:] for i in shuffled_idx])

        vs_plus_1 = self.sess.run(
            self.global_network.vs,
            {self.global_network.s_ph: s_ph,
            self.global_network.ns_ph: ns_ph,
            self.global_network.a_ph: a_ph,
            self.global_network.d_ph: d_ph,
            self.global_network.behavior_policy: b_ph,
            self.global_network.r_ph: r_ph})
        
        s_ph = np.stack([unrolled_state[i, :-1] for i in shuffled_idx])
        ns_ph = np.stack([unrolled_next_state[i, :-1] for i in shuffled_idx])
        r_ph = np.stack([unrolled_reward[i, :-1] for i in shuffled_idx])
        d_ph = np.stack([unrolled_done[i, :-1] for i in shuffled_idx])
        b_ph = np.stack([unrolled_behavior_policy[i, :-1] for i in shuffled_idx])
        a_ph = np.stack([unrolled_action[i, :-1] for i in shuffled_idx])

        vs, rho, value = self.sess.run(
            [self.global_network.vs, self.global_network.rho, self.global_network.value],
            {self.global_network.s_ph: s_ph,
            self.global_network.ns_ph: ns_ph,
            self.global_network.a_ph: a_ph,
            self.global_network.d_ph: d_ph,
            self.global_network.behavior_policy: b_ph,
            self.global_network.r_ph: r_ph})

        pg_advantage = rho * (r_ph + self.discount_factor * (1-d_ph) * vs_plus_1 - value)

        feed_dict = {
            self.global_network.s_ph: s_ph,
            self.global_network.ns_ph: ns_ph,
            self.global_network.r_ph: r_ph,
            self.global_network.d_ph: d_ph,
            self.global_network.a_ph: a_ph,
            self.global_network.behavior_policy: b_ph,
            self.global_network.vs_ph: vs,
            self.global_network.pg_advantage_ph: pg_advantage}

        pi_loss, value_loss, entropy, _ = self.sess.run(
            [self.global_network.pi_loss, self.global_network.value_loss, self.global_network.entropy, self.global_network.train_op],
            feed_dict)

        return pi_loss, value_loss, entropy

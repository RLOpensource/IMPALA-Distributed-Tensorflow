import tensorflow as tf
import numpy as np

import vtrace

def network(x, num_action):
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    actor = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    actor = tf.layers.dense(inputs=actor, units=256, activation=tf.nn.relu)
    actor = tf.layers.dense(inputs=actor, units=num_action, activation=tf.nn.softmax)
    critic = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    critic = tf.layers.dense(inputs=critic, units=256, activation=tf.nn.relu)
    critic = tf.squeeze(tf.layers.dense(inputs=critic, units=1, activation=None), axis=1)
    
    return actor, critic

def build_model(state, trajectory_state, num_action, trajectory):

    with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
        policy, _ = network(state, num_action)

    unrolled_first_state = trajectory_state[:, :-2]
    unrolled_middle_state = trajectory_state[:, 1:-1]
    unrolled_last_state = trajectory_state[:, 2:]

    unrolled_first_policy = []
    unrolled_first_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v = network(unrolled_first_state[:, i], num_action)
            unrolled_first_policy.append(p)
            unrolled_first_value.append(v)
    unrolled_first_policy = tf.stack(unrolled_first_policy, axis=1)
    unrolled_first_value = tf.stack(unrolled_first_value, axis=1)
    
    unrolled_middle_policy = []
    unrolled_middle_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v = network(unrolled_middle_state[:, i], num_action)
            unrolled_middle_policy.append(p)
            unrolled_middle_value.append(v)
    unrolled_middle_policy = tf.stack(unrolled_middle_policy, axis=1)
    unrolled_middle_value = tf.stack(unrolled_middle_value, axis=1)

    unrolled_last_policy = []
    unrolled_last_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v = network(unrolled_last_state[:, i], num_action)
            unrolled_last_policy.append(p)
            unrolled_last_value.append(v)
    unrolled_last_policy = tf.stack(unrolled_last_policy, axis=1)
    unrolled_last_value = tf.stack(unrolled_last_value, axis=1)

    return policy, unrolled_first_policy, unrolled_first_value, \
        unrolled_middle_policy, unrolled_middle_value, \
            unrolled_last_policy, unrolled_last_value

class IMPALA:
    def __init__(self, trajectory, input_shape, num_action, discount_factor, start_learning_rate,
                 end_learning_rate, learning_frame, baseline_loss_coef, entropy_coef, gradient_clip_norm,
                 reward_clipping):

        self.input_shape = input_shape
        self.trajectory = trajectory
        self.num_action = num_action
        self.discount_factor = discount_factor
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.learning_frame = learning_frame
        self.baseline_loss_coef = baseline_loss_coef
        self.entropy_coef = entropy_coef
        self.gradient_clip_norm = gradient_clip_norm

        self.s_ph = tf.placeholder(tf.float32, shape=[None, *self.input_shape])
        self.t_s_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory, *self.input_shape])
        self.a_ph = tf.placeholder(tf.int32, shape=[None, self.trajectory])
        self.r_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory])
        self.d_ph = tf.placeholder(tf.bool, shape=[None, self.trajectory])
        self.b_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory, self.num_action])

        if reward_clipping == 'abs_one':
            self.clipped_r_ph = tf.clip_by_value(self.r_ph, -1.0, 1.0)
        elif reward_clipping == 'soft_asymmetric':
            squeezed = tf.tanh(self.r_ph / 5.0)
            self.clipped_r_ph = tf.where(self.r_ph < 0, .3 * squeezed, squeezed) * 5.

        self.discounts = tf.to_float(~self.d_ph) * self.discount_factor

        self.policy, self.unrolled_first_policy, \
            self.unrolled_first_value, self.unrolled_middle_policy, \
                self.unrolled_middle_value, self.unrolled_last_policy, \
                    self.unrolled_last_value = build_model(state=self.s_ph, trajectory_state=self.t_s_ph,
                                                           num_action=self.num_action, trajectory=self.trajectory)

        self.unrolled_first_action, self.unrolled_middle_action, self.unrolled_last_action = vtrace.split_data(self.a_ph)
        self.unrolled_first_reward, self.unrolled_middle_reward, self.unrolled_last_reward = vtrace.split_data(self.clipped_r_ph)
        self.unrolled_first_discounts, self.unrolled_middle_discounts, self.unrolled_last_discounts = vtrace.split_data(self.discounts)
        self.unrolled_first_behavior_policy, self.unrolled_middle_behavior_policy, self.unrolled_last_behavior_policy = vtrace.split_data(self.b_ph)

        self.vs, self.clipped_rho = vtrace.from_softmax(
                                        behavior_policy_softmax=self.unrolled_first_behavior_policy, target_policy_softmax=self.unrolled_first_policy,
                                        actions=self.unrolled_first_action, discounts=self.unrolled_first_discounts, rewards=self.unrolled_first_reward,
                                        values=self.unrolled_first_value, next_values=self.unrolled_middle_value, action_size=self.num_action)

        self.vs_plus_1, _ = vtrace.from_softmax(
                                        behavior_policy_softmax=self.unrolled_middle_behavior_policy, target_policy_softmax=self.unrolled_middle_policy,
                                        actions=self.unrolled_middle_action, discounts=self.unrolled_middle_discounts, rewards=self.unrolled_middle_reward,
                                        values=self.unrolled_middle_value, next_values=self.unrolled_last_value, action_size=self.num_action)

        self.pg_advantage = tf.stop_gradient(
            self.clipped_rho * \
                (self.unrolled_first_reward + self.unrolled_first_discounts * self.vs_plus_1 - self.unrolled_first_value))

        self.pi_loss = vtrace.compute_policy_gradient_loss(
            softmax=self.unrolled_first_policy,
            actions=self.unrolled_first_action,
            advantages=self.pg_advantage,
            output_size=self.num_action)
        self.baseline_loss = vtrace.compute_baseline_loss(
            vs=tf.stop_gradient(self.vs),
            value=self.unrolled_first_value)
        self.entropy = vtrace.compute_entropy_loss(
            softmax=self.unrolled_first_policy)

        self.total_loss = self.pi_loss + self.baseline_loss * self.baseline_loss_coef + self.entropy * self.entropy_coef

        self.num_env_frames = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.polynomial_decay(self.start_learning_rate, self.num_env_frames, self.learning_frame, self.end_learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, momentum=0, epsilon=0.1)
        gradients, variable = zip(*self.optimizer.compute_gradients(self.total_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)
        self.train_op = self.optimizer.apply_gradients(zip(gradients, variable), global_step=self.num_env_frames)

    def train(self, state, reward, action, done, behavior_policy):
        normalized_state = np.stack(state) / 255
        feed_dict={
            self.t_s_ph: normalized_state,
            self.r_ph: reward,
            self.a_ph: action,
            self.d_ph: done,
            self.b_ph: behavior_policy}

        pi_loss, baseline_loss, entropy, learning_rate, _ = self.sess.run(
            [self.pi_loss, self.baseline_loss, self.entropy, self.learning_rate, self.train_op],
            feed_dict=feed_dict)
        
        return pi_loss, baseline_loss, entropy, learning_rate

    def get_policy_and_action(self, state):
        normalized_state = np.stack(state) / 255
        policy = self.sess.run(
            self.policy, feed_dict={self.s_ph: [normalized_state]})[0]
        action = np.random.choice(self.num_action, p=policy)
        return action, policy, max(policy)

    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def test(self):
        batch_size = 3
        state = np.random.rand(batch_size, self.trajectory, 84, 84, 4)
        action = np.random.randint(self.num_action, size=(batch_size, self.trajectory))
        reward = [[1, 1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1, 0]]
        done = [[False, False, False, True, False, False, True], [False, False, False, False, False, False, True], [False, False, False, False, False, False, False]]

        behavior_policy = [[[0.33032224, 0.34375232,  0.32592544],
                            [0.3289143,  0.34173146,  0.3293543 ],
                            [0.3289951,  0.33941314,  0.33159173],
                            [0.32420245, 0.3402337,   0.3355638 ],
                            [0.3259509,  0.34949812,  0.324551  ],
                            [0.32673714, 0.34975535,  0.32350752],
                            [0.32905534, 0.3409173,   0.33002734]],

                            [[0.32442018, 0.34156674, 0.33401304],
                            [0.3260732,  0.3445669,   0.32935992],
                            [0.3263984,  0.34780538,  0.32579625],
                            [0.33317977, 0.3418529,   0.32496738],
                            [0.31805468, 0.350274,    0.33167133],
                            [0.31823072, 0.34361452,  0.33815467],
                            [0.3327084,  0.3373777,   0.32991385]],

                            [[0.32761666, 0.3401868,  0.33219647],
                            [0.3246559,  0.3405778,   0.33476633],
                            [0.33133945, 0.34374413,  0.32491648],
                            [0.32742482, 0.34393698,  0.3286382 ],
                            [0.33089834, 0.3449944,   0.3241072 ],
                            [0.31870508, 0.34764907,  0.33364582],
                            [0.31830227, 0.34426683,  0.33743086]]]
        feed_dict={self.t_s_ph: state, self.a_ph: action, self.r_ph: reward, self.d_ph: done, self.b_ph: behavior_policy}

        vs = self.sess.run(
            self.vs,
            feed_dict=feed_dict)
        print(vs)
        vs = self.sess.run(
            self.clipped_rho,
            feed_dict=feed_dict)
        print(vs)


        

if __name__ == '__main__':
    sess = tf.Session()

    np.random.seed(0)
    tf.set_random_seed(0)

    agent = IMPALA()
    agent.set_session(sess)
    agent.test()

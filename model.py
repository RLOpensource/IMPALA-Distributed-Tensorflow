import tensorflow as tf
import numpy as np

import vtrace

def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation
    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"
    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def attention_CNN(x):
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
    shape = x.get_shape()
    return tf.layers.flatten(x), [s.value for s in shape]

def action_embedding(previous_action, num_action):
    onehot_action = tf.one_hot(previous_action, num_action)
    x = tf.layers.dense(inputs=onehot_action, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
    return x

def lstm(lstm_hidden_size, flatten, initial_h, initial_c):
    initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_c, initial_h)
    cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)
    output, state = tf.nn.dynamic_rnn(
        cell, flatten, dtype=tf.float32,
        initial_state=initial_state)
    c, h = state
    return output, c, h

def fully_connected(x, hidden_list, output_size, final_activation):
    for h in hidden_list:
        x = tf.layers.dense(inputs=x, units=h, activation=tf.nn.relu)
    return tf.layers.dense(inputs=x, units=output_size, activation=final_activation)


def network(image, previous_action, initial_h, initial_c, num_action, lstm_hidden_size):
    image_embedding, _ = attention_CNN(image)
    previous_action_embedding = action_embedding(previous_action, num_action)
    concat = tf.concat([image_embedding, previous_action_embedding], axis=1)
    expand_concat = tf.expand_dims(concat, axis=1)
    lstm_embedding, c, h = lstm(lstm_hidden_size, expand_concat, initial_h, initial_c)
    last_lstm_embedding = lstm_embedding[:, -1]
    actor = fully_connected(last_lstm_embedding, [256, 256], num_action, tf.nn.softmax)
    critic = tf.squeeze(fully_connected(last_lstm_embedding, [256, 256], 1, None), axis=1)
    return actor, critic, c, h

def build_network(state, previous_action, initial_h, initial_c,
                  trajectory_state, trajectory_previous_action, 
                  trajectory_initial_h, trajectory_initial_c,
                  num_action, lstm_hidden_size, trajectory):

    with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
        policy, _, c, h = network(
            image=state, previous_action=previous_action,
            initial_h=initial_h, initial_c=initial_c,
            num_action=num_action, lstm_hidden_size=lstm_hidden_size)

    unrolled_first_state = trajectory_state[:, :-2]
    unrolled_middle_state = trajectory_state[:, 1:-1]
    unrolled_last_state = trajectory_state[:, 2:]

    unrolled_first_previous_action = trajectory_previous_action[:, :-2]
    unrolled_middle_previous_action = trajectory_previous_action[:, 1:-1]
    unrolled_last_previous_action = trajectory_previous_action[:, 2:]

    unrolled_first_initial_h = trajectory_initial_h[:, :-2]
    unrolled_middle_initial_h = trajectory_initial_h[:, 1:-1]
    unrolled_last_initial_h = trajectory_initial_h[:, 2:]

    unrolled_first_initial_c = trajectory_initial_c[:, :-2]
    unrolled_middle_initial_c = trajectory_initial_c[:, 1:-1]
    unrolled_last_initial_c = trajectory_initial_c[:, 2:]

    unrolled_first_policy = []
    unrolled_first_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v, _, _ = network(
                image=unrolled_first_state[:, i],
                previous_action=unrolled_first_previous_action[:, i],
                initial_h=unrolled_first_initial_h[:, i],
                initial_c=unrolled_first_initial_c[:, i],
                num_action=num_action, lstm_hidden_size=lstm_hidden_size)
            unrolled_first_policy.append(p)
            unrolled_first_value.append(v)
    unrolled_first_policy = tf.stack(unrolled_first_policy, axis=1)
    unrolled_first_value = tf.stack(unrolled_first_value, axis=1)

    unrolled_middle_policy = []
    unrolled_middle_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v, _, _ = network(
                image=unrolled_middle_state[:, i],
                previous_action=unrolled_middle_previous_action[:, i],
                initial_h=unrolled_middle_initial_h[:, i],
                initial_c=unrolled_middle_initial_c[:, i],
                num_action=num_action, lstm_hidden_size=lstm_hidden_size)
            unrolled_middle_policy.append(p)
            unrolled_middle_value.append(v)
    unrolled_middle_policy = tf.stack(unrolled_middle_policy, axis=1)
    unrolled_middle_value = tf.stack(unrolled_middle_value, axis=1)

    unrolled_last_policy = []
    unrolled_last_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v, _, _ = network(
                image=unrolled_last_state[:, i],
                previous_action=unrolled_last_previous_action[:, i],
                initial_h=unrolled_last_initial_h[:, i],
                initial_c=unrolled_last_initial_c[:, i],
                num_action=num_action, lstm_hidden_size=lstm_hidden_size)
            unrolled_last_policy.append(p)
            unrolled_last_value.append(v)
    unrolled_last_policy = tf.stack(unrolled_last_policy, axis=1)
    unrolled_last_value = tf.stack(unrolled_last_value, axis=1)

    return policy, c, h, unrolled_first_policy, unrolled_first_value, \
        unrolled_middle_policy, unrolled_middle_value, \
            unrolled_last_policy, unrolled_last_value

class IMPALA:
    def __init__(self, trajectory, input_shape, num_action, lstm_hidden_size, discount_factor, start_learning_rate,
                 end_learning_rate, learning_frame, baseline_loss_coef, entropy_coef, gradient_clip_norm,
                 reward_clipping, model_name, learner_name):

        self.input_shape = input_shape
        self.trajectory = trajectory
        self.num_action = num_action
        self.lstm_hidden_size = lstm_hidden_size
        self.discount_factor = discount_factor
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.learning_frame = learning_frame
        self.baseline_loss_coef = baseline_loss_coef
        self.entropy_coef = entropy_coef
        self.gradient_clip_norm = gradient_clip_norm

        with tf.variable_scope(model_name):

            with tf.device('cpu'):

                self.s_ph = tf.placeholder(tf.float32, shape=[None, *self.input_shape])
                self.pa_ph = tf.placeholder(tf.int32, shape=[None])
                self.initial_h_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_hidden_size])
                self.initial_c_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_hidden_size])

                self.t_s_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory, *self.input_shape])
                self.t_pa_ph = tf.placeholder(tf.int32, shape=[None, self.trajectory])
                self.t_initial_h_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory, self.lstm_hidden_size])
                self.t_initial_c_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory, self.lstm_hidden_size])
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

                self.policy, self.c, self.h, self.unrolled_first_policy, \
                    self.unrolled_first_value, self.unrolled_middle_policy,\
                        self.unrolled_middle_value, self.unrolled_last_policy,\
                            self.unrolled_last_value = build_network(
                                                        state=self.s_ph, previous_action=self.pa_ph, trajectory=self.trajectory,
                                                        initial_h=self.initial_h_ph, initial_c=self.initial_c_ph,
                                                        num_action=self.num_action, lstm_hidden_size=self.lstm_hidden_size,
                                                        trajectory_state=self.t_s_ph, trajectory_previous_action=self.t_pa_ph,
                                                        trajectory_initial_h=self.t_initial_h_ph, trajectory_initial_c=self.t_initial_c_ph)

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

        self.global_to_session = copy_src_to_dst(learner_name, model_name)
        self.saver = tf.train.Saver()

    def save_weight(self, path):
        self.saver.save(self.sess, path)

    def load_weight(self, path):
        self.saver.restore(self.sess, path)

    def parameter_sync(self):
        self.sess.run(self.global_to_session)

    def train(self, state, reward, action, done, behavior_policy, previous_action, initial_h, initial_c):
        normalized_state = np.stack(state) / 255
        feed_dict={
            self.t_s_ph: normalized_state,
            self.t_pa_ph: previous_action,
            self.t_initial_h_ph: initial_h,
            self.t_initial_c_ph: initial_c,
            self.a_ph: action,
            self.d_ph: done,
            self.r_ph: reward,
            self.b_ph: behavior_policy}

        pi_loss, value_loss, entropy, learning_rate, _ = self.sess.run(
            [self.pi_loss, self.baseline_loss, self.entropy, self.learning_rate, self.train_op],
            feed_dict=feed_dict)
        
        return pi_loss, value_loss, entropy, learning_rate

    def test(self):
        batch_size = 2
        trajectory_state = np.random.rand(batch_size, self.trajectory, *self.input_shape)
        trajectory_previous_action = []
        for _ in range(batch_size):
            previous_action = [np.random.choice(self.num_action) for U in range(self.trajectory)]
            trajectory_previous_action.append(previous_action)
        trajectory_initial_h = np.random.rand(batch_size, self.trajectory, self.lstm_hidden_size)
        trajectory_initial_c = np.random.rand(batch_size, self.trajectory, self.lstm_hidden_size)

        first_value, middle_value, last_value = self.sess.run(
            [self.unrolled_first_value, self.unrolled_middle_value, self.unrolled_last_value],
            feed_dict={
                self.t_s_ph: trajectory_state,
                self.t_pa_ph: trajectory_previous_action,
                self.t_initial_h_ph: trajectory_initial_h,
                self.t_initial_c_ph: trajectory_initial_c})

        print(first_value)
        print('#####')
        print(middle_value)
        print('#####')
        print(last_value)
        print('#####')

    def parameter_sync(self):
        self.sess.run(self.global_to_session)

    def get_policy_and_action(self, state, previous_action, h, c):
        normalized_state = np.stack(state) / 255
        policy, result_c, result_h = self.sess.run(
            [self.policy, self.c, self.h], feed_dict={
                                            self.s_ph: [normalized_state],
                                            self.pa_ph: [previous_action],
                                            self.initial_h_ph: [h],
                                            self.initial_c_ph: [c]})
        policy = policy[0]
        result_c = result_c[0]
        result_h = result_h[0]
        action = np.random.choice(self.num_action, p=policy)
        return action, policy, max(policy), result_c, result_h

    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    
    np.random.seed(0)
    tf.set_random_seed(0)

    sess = tf.Session()
    impala = IMPALA(
                trajectory=20,
                input_shape=[84, 84, 4],
                num_action=3,
                discount_factor=0.999,
                start_learning_rate=0.0006,
                end_learning_rate=0,
                learning_frame=1000000000,
                baseline_loss_coef=0.5,
                entropy_coef=0.01,
                gradient_clip_norm=40,
                reward_clipping='abs_one',
                model_name='actor_0',
                learner_name='learner',
                lstm_hidden_size=256)
    impala.set_session(sess)
    impala.test()
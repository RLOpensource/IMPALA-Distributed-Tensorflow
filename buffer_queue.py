import tensorflow as tf
import collections

class FIFOQueue:
    def __init__(self, trajectory, input_shape, output_size,
                 queue_size, batch_size, num_actors):
        
        self.trajectory = trajectory
        self.input_shape = input_shape
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.unrolled_state = tf.placeholder(tf.uint8, [self.trajectory, *self.input_shape])
        self.unrolled_next_state = tf.placeholder(tf.uint8, [self.trajectory, *self.input_shape])
        self.unrolled_reward = tf.placeholder(tf.float32, [self.trajectory])
        self.unrolled_done = tf.placeholder(tf.bool, [self.trajectory])
        self.unrolled_behavior_policy = tf.placeholder(tf.float32, [self.trajectory, self.output_size])
        self.unrolled_action = tf.placeholder(tf.int32, [self.trajectory])
        self.unrolled_previous_action = tf.placeholder(tf.int32, [self.trajectory])

        self.queue = tf.FIFOQueue(
            queue_size,
            [self.unrolled_state.dtype,
            self.unrolled_next_state.dtype,
            self.unrolled_reward.dtype,
            self.unrolled_done.dtype,
            self.unrolled_behavior_policy.dtype,
            self.unrolled_action.dtype,
            self.unrolled_previous_action.dtype], shared_name='buffer')

        self.queue_size = self.queue.size()
        
        self.enqueue_ops = []
        for i in range(num_actors):
            self.enqueue_ops.append(
                self.queue.enqueue(
                    [self.unrolled_state,
                     self.unrolled_next_state,
                     self.unrolled_reward,
                     self.unrolled_done,
                     self.unrolled_behavior_policy,
                     self.unrolled_action,
                     self.unrolled_previous_action]))

        self.dequeue = self.queue.dequeue()

    def append_to_queue(self, task, unrolled_state, unrolled_next_state,
                        unrolled_reward, unrolled_done, unrolled_behavior_policy,
                        unrolled_action, unrolled_previous_action):

        self.sess.run(
            self.enqueue_ops[task],
            feed_dict={
                self.unrolled_state: unrolled_state,
                self.unrolled_next_state: unrolled_next_state,
                self.unrolled_reward: unrolled_reward,
                self.unrolled_done: unrolled_done,
                self.unrolled_behavior_policy: unrolled_behavior_policy,
                self.unrolled_action: unrolled_action,
                self.unrolled_previous_action: unrolled_previous_action})

    def sample_batch(self):
        batch_tuple = collections.namedtuple('batch_tuple',
        ['state', 'next_state', 'reward', 'done', 'behavior_policy', 'action', 'previous_action'])

        batch = [self.sess.run(self.dequeue) for i in range(self.batch_size)]

        unroll_data = batch_tuple(
            [i[0] for i in batch],
            [i[1] for i in batch],
            [i[2] for i in batch],
            [i[3] for i in batch],
            [i[4] for i in batch],
            [i[5] for i in batch],
            [i[6] for i in batch])

        return unroll_data

    def get_size(self):
        size = self.sess.run(self.queue_size)
        return size

    def set_session(self, sess):
        self.sess = sess

if __name__ == '__main__':
    
    queue = FIFOQueue(
        20, [84, 84, 4], 3, 128, 32, 4)

    print(queue.unrolled_state)
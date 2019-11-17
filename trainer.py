import tensorflow as tf
import numpy as np

import tensorboardX
import buffer_queue
import collections
import py_process
import wrappers
import config
import model
import time
import gym

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS



flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_integer('batch_size', 32, 'how many batch learner should be training')
flags.DEFINE_integer('queue_size', 128, 'fifoqueue size')
flags.DEFINE_integer('trajectory', 20, 'trajectory length')
flags.DEFINE_integer('learning_frame', int(1e9), 'trajectory length')

flags.DEFINE_float('start_learning_rate', 0.0006, 'start_learning_rate')
flags.DEFINE_float('end_learning_rate', 0, 'end_learning_rate')
flags.DEFINE_float('discount_factor', 0.99, 'discount factor')
flags.DEFINE_float('entropy_coef', 0.05, 'entropy coefficient')
flags.DEFINE_float('baseline_loss_coef', 0.5, 'baseline coefficient')
flags.DEFINE_float('gradient_clip_norm', 40.0, 'gradient clip norm')

flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'], 'Job name. Ignored when task is set to -1')
flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'], 'Reward clipping.')

def main(_):

    local_job_device = '/job:{}/task:{}'.format(FLAGS.job_name, FLAGS.task)
    shared_job_device = '/job:learner/task:0'
    is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == FLAGS.task
    is_learner = FLAGS.job_name == 'learner'

    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:{}'.format(8001+i) for i in range(FLAGS.num_actors)],
        'learner': ['localhost:8000']})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task)

    filters = [shared_job_device, local_job_device]

    input_shape = [84, 84, 4]
    output_size = 6
    env_name = 'PongDeterministic-v4'

    with tf.device(shared_job_device):
        queue = buffer_queue.FIFOQueue(
            FLAGS.trajectory, input_shape, output_size,
            FLAGS.queue_size, FLAGS.batch_size, FLAGS.num_actors)
        learner = model.IMPALA(
            trajectory=FLAGS.trajectory,
            input_shape=input_shape,
            num_action=output_size,
            discount_factor=FLAGS.discount_factor,
            start_learning_rate=FLAGS.start_learning_rate,
            end_learning_rate=FLAGS.end_learning_rate,
            learning_frame=FLAGS.learning_frame,
            baseline_loss_coef=FLAGS.baseline_loss_coef,
            entropy_coef=FLAGS.entropy_coef,
            gradient_clip_norm=FLAGS.gradient_clip_norm,
            reward_clipping=FLAGS.reward_clipping,
            model_name='learner',
            learner_name='learner')

    with tf.device(local_job_device):
        if not is_learner:
            actor = model.IMPALA(
                trajectory=FLAGS.trajectory,
                input_shape=input_shape,
                num_action=output_size,
                discount_factor=FLAGS.discount_factor,
                start_learning_rate=FLAGS.start_learning_rate,
                end_learning_rate=FLAGS.end_learning_rate,
                learning_frame=FLAGS.learning_frame,
                baseline_loss_coef=FLAGS.baseline_loss_coef,
                entropy_coef=FLAGS.entropy_coef,
                gradient_clip_norm=FLAGS.gradient_clip_norm,
                reward_clipping=FLAGS.reward_clipping,
                model_name='actor_{}'.format(FLAGS.task),
                learner_name='learner')

    sess = tf.Session(server.target)
    queue.set_session(sess)
    learner.set_session(sess)
    
    if not is_learner:
        actor.set_session(sess)

    if is_learner:

        writer = tensorboardX.SummaryWriter('runs/learner')
        train_step = 0

        while True:
            size = queue.get_size()
            if size > 3 * FLAGS.batch_size:
                train_step += 1
                batch = queue.sample_batch()
                s = time.time()
                pi_loss, baseline_loss, entropy, learning_rate = learner.train(
                                                                    state=np.stack(batch.state),
                                                                    reward=np.stack(batch.reward),
                                                                    action=np.stack(batch.action),
                                                                    done=np.stack(batch.done),
                                                                    behavior_policy=np.stack(batch.behavior_policy))
                writer.add_scalar('data/pi_loss', pi_loss, train_step)
                writer.add_scalar('data/baseline_loss', baseline_loss, train_step)
                writer.add_scalar('data/entropy', entropy, train_step)
                writer.add_scalar('data/learning_rate', learning_rate, train_step)
                writer.add_scalar('data/time', time.time() - s, train_step)
    else:
        state = np.ones([84, 84, 4])
        while True:
            time.sleep(1)
            _, learner_policy, _ = learner.get_policy_and_action(state)
            _, actor_policy, _ = actor.get_policy_and_action(state)
            print('shared job name : {}'.format(shared_job_device))
            print('local job name : {}'.format(local_job_device))
            print('learner policy : {}'.format(learner_policy))
            print('actor {} policy : {}'.format(FLAGS.task, actor_policy))
            actor.parameter_sync()

if __name__ == '__main__':
    tf.app.run()

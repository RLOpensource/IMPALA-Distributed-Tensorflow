import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import a3c_training_thread
import argparse
import wrappers
import config
import utils
import copy
import time
import sys
import gym

from tensorboardX import SummaryWriter

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == 'worker':
        device=tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)

        global_network = a3c_training_thread.Network(
            input_shape=config.input_shape,
            output_size=config.output_size,
            unroll=config.unroll,
            thread_index='global',
            device=device)

        local_network = [a3c_training_thread.Network(
            input_shape=config.input_shape,
            output_size=config.output_size,
            unroll=config.unroll,
            thread_index=i,
            device=device) for i in range(len(worker_hosts))]

        agent = a3c_training_thread.Agent(
            input_shape=config.input_shape,
            output_size=config.output_size,
            unroll=config.unroll,
            thread_index=FLAGS.task_index,
            device=device,
            global_network=global_network,
            local_network=local_network[FLAGS.task_index])

        with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
            global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
            global_step_ph=tf.placeholder(global_step.dtype,shape=global_step.get_shape())
            global_step_ops=global_step.assign(global_step_ph)
            init_op=tf.global_variables_initializer()
            saver = tf.train.Saver()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                   global_step=global_step,
                                   logdir=None,
                                   summary_op=None,
                                   saver=saver,
                                   init_op=init_op)

        with sv.managed_session(server.target) as sess:
            writer = SummaryWriter('runs/task_index_{}'.format(FLAGS.task_index))
            agent.set_session(sess)
            agent.assign()

            env = wrappers.make_env('BreakoutDeterministic-v4')
            if FLAGS.task_index == 0:
                env = gym.wrappers.Monitor(env, 'save-mov', video_callable=lambda episode_id: episode_id%10==0)
            done = False
            state = env.reset()
            lives = 5

            episode = 0
            score = 0
            step = 0
            total_max_prob = 0
            train_episode = 0


            while True:

                train_episode += 1

                episode_state = []
                episode_next_state = []
                episode_reward = []
                episode_done = []
                episode_action = []
                episode_behavior_policy = []

                for i in range(128):
                    
                    action, behavior_policy, max_prob = agent.get_policy_and_action(state)

                    step += 1
                    total_max_prob += max_prob

                    next_state, reward, done, info = env.step(action+1)

                    if lives != info['ale.lives']:
                        r = -1
                        d = True
                    else:
                        r = reward
                        d = False

                    score += reward

                    episode_state.append(state)
                    episode_next_state.append(next_state)
                    episode_reward.append(r)
                    episode_done.append(d)
                    episode_action.append(action)
                    episode_behavior_policy.append(behavior_policy)

                    state = next_state
                    lives = info['ale.lives']

                    if done:
                        print(FLAGS.task_index, episode, score, step, total_max_prob / step)
                        writer.add_scalar('score', score, episode)
                        writer.add_scalar('max_prob', total_max_prob / step, episode)
                        writer.add_scalar('step', step, episode)
                        episode += 1
                        score = 0
                        step = 0
                        total_max_prob = 0
                        lives = 5
                        state = env.reset()

                pi_loss, value_loss, entropy = agent.update(
                                                    state=np.stack(episode_state),
                                                    next_state=np.stack(episode_next_state),
                                                    reward=np.stack(episode_reward),
                                                    done=np.stack(episode_done),
                                                    action=np.stack(episode_action),
                                                    behavior_policy=np.stack(episode_behavior_policy))
                agent.assign()
                writer.add_scalar('pi_loss', pi_loss, train_episode)
                writer.add_scalar('value_loss', value_loss, train_episode)
                writer.add_scalar('entropy', entropy, train_episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

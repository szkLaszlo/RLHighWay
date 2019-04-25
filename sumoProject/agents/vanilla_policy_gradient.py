# https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import time
import sumoProject.envs.SUMO_Starter
TRAIN = True

env = gym.make('EPHighWay-v1')

gamma = 0.90


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        # hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden1 = tf.layers.dense(inputs=self.state_in, units=h_size, bias_initializer=None, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=h_size, bias_initializer=None, activation=tf.nn.relu)
        # self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.output = tf.layers.dense(inputs=hidden2, units=a_size, activation=tf.nn.softmax, bias_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


tf.reset_default_graph()  # Clear the Tensorflow graph.

# myAgent = agent(lr=1e-2, s_size=4, a_size=2, h_size=8)  # Load the agent.
myAgent = agent(lr=1e-3, s_size=20, a_size=25, h_size=128)  # Load the agent.

total_episodes = 30000  # Set total number of episodes to train agent on.
update_frequency = 5

init = tf.global_variables_initializer()
saver = tf.train.Saver()
causes={}
# Launch the tensorflow graph
with tf.Session() as sess:
    i = 0
    total_reward = []
    total_lenght = []
    won = 0

    if TRAIN:
        # env.render(mode='noone')
        env.render()
        sess.run(init)
        gradBuffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        while i < total_episodes:
            s = env.reset()
            s = env.state_to_tuple(s)
            running_reward = 0
            ep_history = []
            while True:
                # Probabilistically pick an action given our network outputs.
                da_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})  # a_dist=[0.1 0.2 0.7]
                a = np.random.choice(da_dist[0], p=da_dist[0])  # a = [0.2]
                a = np.argmax(da_dist == a)  # a = 1

                s1, r, d, info = env.step(a)  # Get our reward for taking an action given a bandit.
                s1 = env.state_to_tuple(s1)
                ep_history.append([s, a, r, s1])
                s = s1
                running_reward += r
                if d:
                    # Update the network.
                    ep_history = np.array(ep_history)
                    ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                    feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                                 myAgent.action_holder: ep_history[:, 1], myAgent.state_in: np.vstack(ep_history[:, 0])}
                    grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                    for idx, grad in enumerate(grads):
                        gradBuffer[idx] += grad

                    if i % update_frequency == 0 and i != 0:
                        feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                        _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                        for ix, grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0

                    total_reward.append(running_reward)
                    if info['cause'] is None:
                        won += 1
                    else:
                        try:
                            causes[info['cause']] = causes[info['cause']]+1
                        except KeyError:
                            causes[info['cause']] = 1
                        except AttributeError:
                            causes[info['cause']] = 1
                    break

                # Update our running tally of scores.
            if i % 100 == 0:
                save_path = saver.save(sess, "../save_model/model2.ckpt")
                # print("Episode: "+str(i)+", Won: "+str(won)+", Avg. reward: "+str(np.mean(total_reward[-10:]))+
                #       ", Save path: "+save_path)
                print("Episode: " + str(i) + ", Won: " + str(won) + ", Avg. reward: " + str(np.mean(total_reward[-10:])))

                won = 0
            i += 1
        print(causes)

    else:
        env.render()
        saver.restore(sess, "../save_model/model2.ckpt")
        while i < total_episodes:
            s = env.reset()
            s = env.state_to_tuple(s)
            ep_reward = 0
            while True:
                a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
                a = np.argmax(a_dist)

                s1, r, d, info = env.step(a)  # Get our reward for taking an action given a bandit.
                env.render()
                s1 = env.state_to_tuple(s1)
                s = s1
                ep_reward += 1
                if d:
                    total_reward.append(r)
                    if r > 0 and info['cause'] is None:
                        won += 1
                    break
            if i % 10 == 0:
                # print("Episode: " + str(i) + ", EP reward: " + str(ep_reward) + ", Avg. reward: " + str(
                #     np.mean(total_reward[-100:])))
                #
                print("Episode: " + str(i) + ", Won: " + str(won) + ", Avg. reward: " + str(np.mean(total_reward[-10:])))
                won = 0
            i += 1

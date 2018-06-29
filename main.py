from __future__ import print_function
import os
import time
import logging
import argparse
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers

from models.model import Model

def config_logging(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def pathlength(path):
    return len(path['rewards'])

def discounted_rewards_to_go(rewards, gamma):
    """ state/action-centric policy gradients
    """
    rtgs = [] 
    future_reward = 0
    # start at time step t and use future_reward to calculate current reward
    for r in reversed(rewards):
        future_reward = future_reward * gamma + r
        rtgs.append(future_reward)
    return rtgs[::-1]

def sum_discounted_rewards(rewards, gamma):
    """ trajectory-centric policy gradients
    """
    return sum((gamma**i) * rewards[i] for i in range(len(rewards)))


def create_model(session, discrete, num_observations, num_actions, num_layers,
                    layer_size, logger, optimizer, learning_rate, checkpoint_dir, restore):
    model = Model(discrete, num_observations, num_actions, num_layers,
                  layer_size, logger, optimizer, learning_rate, checkpoint_dir)

    if restore:
        restored = model.load(session)
        if not restored:
            logger.info("Created model with fresh parameters")
            session.run(tf.global_variables_initializer())
    else:
        logger.info("Created model with fresh parameters")
        session.run(tf.global_variables_initializer())

    return model

def preprocess_frame(image):
    """ preprocess 210x160x3 uint8 frame into 6400 (80x80) 1 dim float vector
    """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    return image.astype(np.float).ravel()

def train(env_name='Pong-v0',
             gamma=.99, 
             learning_rate=1e-3, 
             reward_to_go=True, 
             render=True, 
             results_dir=None, 
             checkpoint_dir=None,
             normalize_advantages=True,
             nn_baseline=False, 
             # network arguments
             num_layers=1,
             layer_size=200, 
             optimizer='adam',
             restore=False,
             keep_prob=1,
             batch_size=10
             ):
    tf.reset_default_graph()
    
    with tf.Session() as session:
        env = gym.make(env_name)

        env = wrappers.Monitor(env, results_dir, force=True)

        discrete = isinstance(env.action_space, gym.spaces.Discrete)

        num_observations = 80 * 80
        NOOP, UP, DOWN = 0, 2, 5
        pong_actions = [NOOP, UP, DOWN]
        num_actions = 3
    
        model = create_model(session, discrete, num_observations, num_actions, num_layers,
            layer_size, logger, optimizer, learning_rate, checkpoint_dir, restore)

        file_writer = tf.summary.FileWriter(results_dir, session.graph)

        observation = env.reset()
        prev_frame = None
        observations, actions, rewards, batch_q_n = [], [], [], []
        episode_number = 0
        running_reward = None
        reward_sum = 0
        step = 0
        while True:
            curr_frame = preprocess_frame(observation)
            difference_frame = curr_frame - prev_frame if prev_frame is not None else np.zeros(num_observations)
            prev_frame = curr_frame

            observations.append(difference_frame)
            action = session.run(model.sampled_ac, feed_dict={model.observations : [difference_frame]})
            action = action[0]
            actions.append(action)
            pong_action = pong_actions[action]

            observation, reward, done, _ = env.step(pong_action)
            logger.debug('step:{} action:{} pong_action:{} reward:{}'.format(step, action, pong_action, reward))
            reward_sum += reward
            rewards.append(reward)
            if done:
                episode_number += 1
                q_n = []
                
                if reward_to_go:
                    q_n = np.concatenate([discounted_rewards_to_go(rewards, gamma)])
                else:
                    q_n = np.concatenate([
                            [sum_discounted_rewards(rewards, gamma)] * len(rewards)])

                batch_q_n.append(q_n)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                running_reward_summary = tf.Summary(value=[tf.Summary.Value(tag="running_reward", simple_value=running_reward)])
                file_writer.add_summary(running_reward_summary, global_step=episode_number)

                # update model
                if episode_number % batch_size == 0:
                    step += 1

                    if nn_baseline:
                        # TODO: Compute Baselines
                        pass
                    else:
                        advantages = batch_q_n.copy()

                    if normalize_advantages:
                        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
                        logger.debug('advantages: {}'.format(advantages))

                    loss, training_scalar, histogram_merged = model.update(session, observations, actions, advantages, keep_prob)

                    file_writer.add_summary(training_scalar, step)
                    file_writer.add_summary(histogram_merged, step)

                    logger.info("Epoch %3d Loss %f" %(step, loss))

                    observations, actions, rewards, batch_q_n = [], [], [], []

                # TODO
                if episode_number % 100 == 0:
                    model.save(session)

                # reset
                observation = env.reset()
                reward_sum = 0



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Pong-v0')
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--layer_size', '-s', type=int, default=200)
    parser.add_argument('--restore', '-restore', action='store_true')
    args = parser.parse_args()

    checkpoint_dir = os.path.join(os.getcwd(), 'results', args.env_name)
    results_dir = os.path.join(os.getcwd(), 'results', args.env_name, args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train(env_name=args.env_name,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        reward_to_go=args.reward_to_go,
        render=args.render,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        normalize_advantages=not(args.dont_normalize_advantages),
        nn_baseline=args.nn_baseline, 
        num_layers=args.n_layers,
        layer_size=args.layer_size,
        batch_size=args.batch_size,
        restore=args.restore
        )

if __name__ == "__main__":
    log_file = os.path.join(os.getcwd(), 'results', 'train_out.log')
    logger = config_logging(log_file)

    main()

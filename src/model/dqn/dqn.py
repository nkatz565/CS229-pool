import math
import pickle
import numpy as np

from . import dqn_agent
import torch

def save_model(filepath, model):
    torch.save(model.qnetwork_local.state_dict(), filepath)

def load_model(modelpath, model_params):
    state_size = model_params['s_dim']
    #print('State size is ' + str(state_size))
    action_buckets = model_params['buckets']
    #print('Action buckets are ' + str(action_buckets))
    action_size = action_buckets[0] * action_buckets[1]
    #print('Action size is ' + str(action_size))

    agent = dqn_agent.Agent(state_size, action_size, seed = 229)
    agent.qnetwork_local.load_state_dict(torch.load(modelpath))
    #print(agent.qnetwork_local.state_dict())

    return agent


def action_to_tuple(action, action_buckets):
    #print('Action buckets are ' + str(action_buckets))
    return(float(int(action) % action_buckets[0]),\
        int(action/action_buckets[0]))

def choose_action(state, model, action_space, epsilon = 0.):
    action = action_to_tuple(model.act(state, epsilon), action_space.buckets)
    print('action was ' + str(action))
    return action

def train(env, model_path, episodes=200, episode_length=50):
    print('DQN training')

    # Initialize DQN Agent
    state_size = env.state_space.n
    action_buckets = [360, 1]
    env.set_buckets(action=action_buckets)
    action_size = action_buckets[0] * action_buckets[1]

    agent = dqn_agent.Agent(state_size, action_size, seed = 229)

    # Learning related constants; factors should be determined by trial-and-error
    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25))) # epsilon-greedy, factor to explore randomly; discounted over time
    get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25))) # learning rate; discounted over time
    gamma = 0.8 # reward discount factor

    # Q-learning
    for i_episode in range(episodes):
        epsilon = get_epsilon(i_episode)
        lr = get_lr(i_episode)

        state = env.reset() # reset environment to initial state for each episode
        rewards = 0 # accumulate rewards for each episode
        done = False
        for t in range(episode_length):
            # Agent takes action using epsilon-greedy algorithm, get reward
            #print('Episode ' + str(i_episode))
            #print('Iter ' + str(t))
            #for i in range(2, 2, len(state)):
            #    state[i] -= state[0]
            #    state[i+1] -= state[1]
            action = agent.act(state, epsilon)
            #print('The action number is ' + str(action))
            #print('The action is ' + str(action_to_tuple(action, action_buckets)))
            next_state, reward, done = env.step(action_to_tuple(action, action_buckets))
            rewards += reward
            #print('The reward is ' + str(reward))
            #print('The next state is ' + str(next_state))

            # Agent learns over New Step
            agent.step(state, action, reward, next_state, done)

            # Transition to next state
            state = next_state



            if done:
                print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, t+1, rewards))
                with open("output\\dqn-log.txt", "a") as myfile:
                    myfile.write('Episode {} finished after {} timesteps, total rewards {}\n'.format(i_episode, t+1, rewards))
                break
        if not done:
            print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, episode_length, rewards))
            with open("output\\dqn-log.txt", "a") as myfile:
                myfile.write('Episode {} finished after {} timesteps, total rewards {}\n'.format(i_episode, episode_length, rewards))

        save_model(model_path, agent)
        #print(agent.qnetwork_local.state_dict())

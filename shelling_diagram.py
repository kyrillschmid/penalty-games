from env import CoinGame
from coord_comp import CoordComp
from sd import SocialDilemma
from n_player_prisoners_dilemma import NPlayerPrisonersDilemma
from agent import Agent
from market import Market
import pandas as pd
import os
import json
from dotmap import DotMap
import numpy as np
from collections import defaultdict

def main(params, params_q):
    episodes = params.episodes
    #settings = {'Markov Game': (0.0, None), 'Stochastic Market Game': (1.0, Market(price=params.price))}
    settings = {'Stochastic Market Game': (1.0, Market(price=params.price))}
    env = NPlayerPrisonersDilemma(None, params)
    step = 0
    stats = pd.DataFrame(columns=['run', 'episode', 'setting', 'agent', 'reward', 'price', 'action'])
    
    for run in range(params.runs):
        for setting_key, setting in zip(settings.keys(), settings.values()):
            env = NPlayerPrisonersDilemma(setting[1], params)
            agents = []
            nb_sanctioneers = params.n_player_prisoners_dilemma.nb_sanctioneers
            nb_non_sanctioneers = env.N - params.n_player_prisoners_dilemma.nb_sanctioneers
            for _ in range(nb_sanctioneers):
                agent = Agent(env, params_q)
                agents.append(agent)
            for _ in range(nb_non_sanctioneers):
                agent = Agent(env, params_q, action_space_n=2)
                agents.append(agent)

            for i_episode in range(episodes):
                state = env.reset()
                episode_return = 0
                done = False
                while not done:
                    actions = []
                    for agent in agents:
                        action = agent.get_action(i_episode, state)
                        actions.append(action)
                    next_state, reward, done, info = env.step(actions)
                    episode_return_no_market = np.sum(reward)
                    if setting[1] is not None:
                        if env.name == 'n_player_prisoners_dilemma':
                            reward = setting[1].compute_trading_nppd(actions, reward)
                        else:
                            reward = setting[1].compute_trading(actions, reward) 
                    episode_return += np.sum(reward)
                    step += 1
                    for i_ag, agent in enumerate(agents):
                        agent.update_q(state, actions[i_ag], reward[i_ag], next_state)
                    if done:
                        break
                    state = next_state
                if i_episode % 100 == 0:
                    stats.loc[len(stats.index)] = [run, i_episode, '{}'.format(nb_sanctioneers), 'sanctioneers', np.sum(reward[:nb_sanctioneers]), params.price, info['share_cooperators']]
                    stats.loc[len(stats.index)] = [run, i_episode, '{}'.format(nb_sanctioneers), 'non-sanctioneers', np.sum(reward[nb_sanctioneers:]), params.price, info['share_cooperators']]

    if params.log:
        if setting[1] is not None:
            stats.to_csv(os.path.join('exps', 'shelling-{}-nb-sanctioneers-{}-nb-agents-{}-alpha-{}.csv'.format(env.name, nb_sanctioneers, env.N, params_q.alpha)))
        # sns.lineplot(x="i_episode", y="reward_rolling", hue='setting', data=data)
        # plt.savefig('plots/coord-comp-{}.png'.format(params.comp))
        # plt.close()


def save_params(params, output_path):
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        json.dump(params, f)

if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'params_pd.json'), 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)
    with open(os.path.join(os.path.dirname(__file__), 'params_q_learning.json'), 'r') as f:
        params_json = json.load(f)
    params_q = DotMap(params_json)

    for nb_sanctioneers in [i for i in range(params.n_player_prisoners_dilemma.N + 1)]:
        params.n_player_prisoners_dilemma.nb_sanctioneers = nb_sanctioneers
        print(nb_sanctioneers)
        #main(params, params_q)
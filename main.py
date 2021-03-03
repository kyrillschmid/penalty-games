from social_dilemma import make_social_dilemma
from params import get_params
from agent import Agent
import pandas as pd
import os
import json
import numpy as np
import argparse


def main(params):
    episodes = params.episodes
    stats = pd.DataFrame(columns=params.logging_columns)
    
    step = 0
    for run in range(params.nb_runs):    
        env = make_social_dilemma(params)
        agents = []
        for _ in range(env.N):
            agent = Agent(env, params)
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
                
                if params.penalization:
                    reward = env.penalization.compute_penalties(actions, reward)
                    
                episode_return += np.sum(reward)
                step += 1
                for i_ag, agent in enumerate(agents):
                    agent.update_q(state, actions[i_ag], reward[i_ag], next_state)
                if done:
                    break
                state = next_state

            if i_episode % 100 == 0 or (i_episode == episodes-1):                    
                succ_c = 0 
                succ_d = 0
                if env.penalization is not None:
                    succ_c = env.penalization.succ_sanction_c
                    succ_d = env.penalization.succ_sanction_d
                game_type = "{} penalties {}".format(env.name, params.penalization)
                stats.loc[step] = [run, i_episode, game_type, 'sum', np.sum(reward), params.penalty, env.share_cooperators] + \
                                  [succ_c, succ_d] + \
                                   env.get_action_frequencies(actions)


    if params.log:
        stats.to_csv(os.path.join('exps', env.name, '{}-player-{}-penalization-{}.csv'.format(env.N, env.name, params.penalization)))


def save_params(params, output_path):
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        json.dump(params, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env", metavar="ENV", type=str, help="{} or {}".format('PD', 'SH', 'CH', 'NPIPD'))
    parser.add_argument("--eval", metavar="EVAL", type=str, help="{}".format('single'))
    args = parser.parse_args()
    assert args.env in ['PD', 'SH', 'CH', 'NPIPD']

    params = get_params()
    params.env = args.env
    if args.eval == 'single':
        main(params)
    else:
        for penalization in [True, False]:
            params.penalization = penalization
            main(params)
        
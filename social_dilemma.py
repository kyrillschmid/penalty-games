import numpy as np
import gym
from penalty import Penalty
import pandas as pd


def make_social_dilemma(params):
    if params.env in ['PD', 'SH', 'CH']:
        game = TwoPlayerSocialDilemma(params.env, params)
    elif params.env in ['NPIPD']:
        game = NPlayerPrisonersDilemma(params.env, params)
    if params.penalization:
        game.set_penalization(params)
    return game


class SocialDilemma():
    def __init__(self, env, params):
        self.actions = np.array([[0, 0, 0], [1, 0, 0]])
        self.action_labels = ['C-', 'D-', 'CC', 'CD', 'DC', 'DD']
        self.action_space = gym.spaces.Discrete(n=len(self.actions))

        self.name = env
        self.rewards = []
        self.N = None
        self.k = None
        self.share_cooperators = None
        self.penalization = None

    def step(self, actions):
        raise NotImplementedError

    def set_penalization(self, params):
        self.actions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]])  # penalty_game
        self.action_space = gym.spaces.Discrete(n=len(self.actions))
        self.penalization = Penalty(penalty=params.penalty)
        self.penalization.actions = self.actions

    def get_action_frequencies(self, actions):
        df = pd.DataFrame(actions, columns=['action'])
        df = dict(df.groupby(df['action']).size())
        frequencies = [0 for _ in range(len(self.action_labels))]
        for i_a in range(len(self.action_labels)):
            if i_a in df.keys():
                frequencies[i_a] = df[i_a]
        return frequencies

    def reset(self):
        return self.observation

    @property
    def observation(self):
        return 0



class TwoPlayerSocialDilemma(SocialDilemma):
    def __init__(self, env, params):
        super(TwoPlayerSocialDilemma, self).__init__(env, params)

        self.N = 2
        self.k = 0.0
        if env == 'PD':
            self.rewards = [[[1, 1], [-0.5, 1.5]],
                             [[1.5, -0.5], [0, 0]]]

        if env == 'SH':
            self.rewards = [[[4, 4], [0, 3]],
                             [[3, 0], [1, 1]]]

        if env == 'CH':
            self.rewards = [[[3, 3], [1, 4]],
                             [[4, 1], [0, 0]]]
        

    def step(self, actions):
        info = {}
        env_actions = []        
        for action in actions:
            env_actions.append(self.actions[action])
        actions = env_actions

        rewards = self.rewards[actions[0][0]][actions[1][0]]

        self.k = float(self.N - np.sum([i[0] for i in actions]))
        self.share_cooperators = self.k / self.N
        
        return self.observation, rewards, True, info

    def reset(self):
        return self.observation

    @property
    def observation(self):
        return 0


class NPlayerPrisonersDilemma(SocialDilemma):
    def __init__(self, env, params):
        super(NPlayerPrisonersDilemma, self).__init__(env, params)
        self.f = params.npipd.f
        self.p = None
        self.N = params.npipd.N
        self.k = 0.0
        
        

    def step(self, actions):
        info = {}
        env_actions = []
        for action in actions:
            env_actions.append(self.actions[action])
            
        actions = env_actions
        k = self.N - np.sum([i[0] for i in actions])
        p = k
        r_d = (self.f * k * p) / float(self.N)
        r_c = r_d - p
        self.k = k

        rewards = np.zeros(self.N)
        for i_ag in range(self.N):
            if actions[i_ag][0] == 1:
                rewards[i_ag] += r_d
            else:
                rewards[i_ag] += r_c

        info['share_cooperators'] = k / self.N

        return self.observation, rewards, True, info

    def reset(self):
        return self.observation

    @property
    def observation(self):
        return 0
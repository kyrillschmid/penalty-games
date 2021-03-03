import numpy as np


class Penalty:

    def __init__(self, penalty):
        self.actions = None
        self.penalty = penalty
        self.succ_sanction_c = 0
        self.succ_sanction_d = 0

    def compute_penalties(self, actions, reward):
        self.succ_sanction_c = 0
        self.succ_sanction_d = 0
        env_actions = []
        offers = []
        n_agents = len(actions)
        executed = [np.zeros(2) for i in range(n_agents)]

        for i_ac, action in enumerate(actions):
            ac = self.actions[action]
            env_actions.append(ac)
            offers.append(ac[1:3])
            if ac[0] == 0:
                executed[i_ac][0] = 1.0
            if ac[0] == 1:
                executed[i_ac][1] = 1.0
            
        actions = env_actions
        rewards = np.zeros(n_agents)
        
        indices = np.random.choice([i for i in range(n_agents)], replace=False, size=n_agents).reshape(int(n_agents / 2), 2)
        for i, j in indices:
            if offers[i][0] == 1 and executed[j][0] == 1 or offers[j][0] == 1 and executed[i][0] == 1:
                self.succ_sanction_c += 1
            if offers[i][1] == 1 and executed[j][1] == 1 or offers[j][1] == 1 and executed[i][1] == 1:
                self.succ_sanction_d += 1

            rewards[i] += reward[i] - self.penalty * np.sum(offers[i] * executed[j]) + self.penalty * np.sum(offers[j] * executed[i])
            rewards[j] += reward[j] - self.penalty * np.sum(offers[j] * executed[i]) + self.penalty * np.sum(offers[i] * executed[j])

        return rewards
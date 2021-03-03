import numpy as np
from collections import defaultdict

class Agent:
    def __init__(self, env, params, action_space_n=None):
        if action_space_n is not None:
            self.nA = action_space_n
            self.Q = defaultdict(lambda: np.zeros(action_space_n))
        else:
            self.nA = env.action_space.n
            self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.gamma = params.q_learning.gamma
        self.alpha = params.q_learning.alpha
        # linear annealed epsilon
        self.epsilon = params.q_learning.epsilon_init
        self.epsilon_min = params.q_learning.epsilon_min
        self.a = params.q_learning.epsilon_init
        self.b = (params.q_learning.epsilon_min - params.q_learning.epsilon_init) / params.q_learning.nb_steps_epsilon

    def policy_fn(self, i_step, observation):
        self.epsilon = np.maximum(self.a + (self.b * i_step), self.epsilon_min)
        A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        if np.max(self.Q[observation]) == np.min(self.Q[observation]):
            best_action = np.random.randint(0, self.nA)
        else:
            best_action = np.argmax(self.Q[observation])
        A[best_action] += (1.0 - self.epsilon)
        return A

    def get_action(self, i_step, state):
        action_probs = self.policy_fn(i_step, state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update_q(self, state, action, reward, new_state):
        if np.max(self.Q[new_state]) == np.min(self.Q[new_state]):
            best_next_action = np.random.randint(0, self.nA)
        else:
            best_next_action = np.argmax(self.Q[new_state])
        td_target = reward + self.gamma * self.Q[new_state][best_next_action]
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta


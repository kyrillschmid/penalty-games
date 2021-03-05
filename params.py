from dotmap import DotMap

def get_params():
    params = DotMap()

    # SH = Stag Hunt, CH = Chicken, PD = Prisoner's Dilemma, NPIPD = N-Player Prisoner's Dilemma

    # global
    params.penalization = True
    params.nb_runs = 25
    params.episodes = 4000 
    params.log = True
    params.logging_columns = ['run', 'episode', 'game type', 'agent', 'reward', 'penalty',
                              'share_cooperators', 'successful_sanction_c', 'successful_sanction_d',
                              'C-', 'D-', 'CC', 'CD', 'DC', 'DD']

    # Q-Learning
    params.q_learning.nb_steps_epsilon = 500
    params.q_learning.epsilon_init = 1.0
    params.q_learning.epsilon_min = 0.0001
    params.q_learning.gamma = 0.9
    
    # Two Player Social dilemmas
    params.sd.q_learning.alpha = 0.2
    params.sd.penalty = -2.0    

    # N-Player Prisoner's Dilemma
    params.npipd.q_learning.alpha = 0.008
    params.npipd.penalty = {"32": -70, "64": -160, "128": -300}

    return params
    
from dotmap import DotMap

def get_params():
    params = DotMap()

    # global
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
    params.q_learning.alpha = 0.2 # [SH=0.2, CH=-0.2, PD=0.2, NPIPD = 0.008]

    # N Player Prisoner's Dilemma 
    params.npipd.N = 64 # 32 # 64 # 128
    params.npipd.f = 2.0
    
    # penalization
    params.penalization = True
    params.penalty = -160 # [SH=-2.0, CH=-2.0, PD=-2.0, NPIPD_32=-70, NPIPD_64=-160, NPIPD=128=-300]

    return params
    
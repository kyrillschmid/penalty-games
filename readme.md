## Learning to Penalize Other Learning Agents
A repository to reproduce the experiments from the Paper "Learning to Penalize Other Learning Agents"

#### To reproduce the results do the following steps
    - Clone repository
    - Requires python3.x, gym, dotmap, pandas, numpy, matplotlib, notebook, seaborn

#### Install requirements:
    - To install via pip run: pip install -r requirements.txt 
    - To install via pipenv: pipenv install

#### Run experiments:
    - Set parameters in parameters.py for the respective setting
    - Run python3 main.py |PD|SH|CH for two player social dilemmas 
    - Run python3 main.py NPIPD --nb_agents=|32|64|128| for N-Player Prisoner's dilemma
    - To see results in notebook run: jupyter notebook Penalty-Games.ipynb
    * SH = Stag Hunt, CH = Chicken, PD = Prisoner's Dilemma, NPIPD = N-Player Prisoner's Dilemma
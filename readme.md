## Learning to Penalize Other Learning Agents
A repository to reproduce the experiments from the Paper "Learning to Penalize Other Learning Agents"

#### SH = Stag Hunt, CH = Chicken, PD = Prisoner's Dilemma, NPIPD = N-Player Prisoner's Dilemma

#### To reproduce the results do the following steps
    - Clone repository
    - Requires python3.x, gym, dotmap, pandas, numpy, matplotlib, notebook, seaborn

#### Install requirements:
    - To install via pip run: pip install -r requirements.txt 
    - To install via pipenv: pipenv install

#### Run experiments:
    - Set parameters in parameters.py for the respective setting
    - Run python3 main.py [PD, SH, CH, NPIPD]
    - Run jupyter notebook 
    - To see results in notebook run: jupyter notebook Penalty-Games.ipynb
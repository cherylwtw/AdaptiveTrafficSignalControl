# AdaptiveTrafficSignalControl
This project explores applying Reinfrocement Learning approaches to traffic signal control

Acknowledgement:<br />
State, action definitions are implemented based on the understanding of the paper 'Adaptive Traffic Signal Control : Exploring Reward Definition For Adaptive Traffic Signal Control : Exploring Reward Definition For Reinforcement Learning' written by Saad Touhbia, Mohamed Ait Babrama, Tri Nguyen-Huub, Nicolas Marilleaub, Saad Touhbi , Mohamed Ait Babram , Tri Nguyen-Huu , Nicolas Marilleau, Moulay L. Hbida, Christophe Cambierb, Serge Stinckwichb (https://ac.els-cdn.com/S1877050917309912/1-s2.0-S1877050917309912-main.pdf?_tid=dd25f67a-c877-4538-a81e-8f044412a2da&acdnat=1524859146_4fbe7f74db4fd49aa0e9d9d7b4b23dab)

Envrionment and Packages:<br />
A OpenAI Gym (https://gym.openai.com/envs/#classic_control) environment is defined to simulate the environment using the following packages
  - Simulation of Urban Mobility, SUMO (http://sumo.dlr.de/index.html) for microscopic traffic simulation
  - TraCi (http://sumo.dlr.de/wiki/TraCI) for communication between agents and SUMO in Python

Follow instruction to install SUMO and add environment variable SUMO_HOME (http://sumo.dlr.de/wiki/Installing)

###Files###
four_intersects.py<br />
definition of RL agents, control of traffic lights using fixed control Q learning and dyna_Q

graph_results.py<br />
graph experiment result based on the output of four_intersects.py

gym_env
  - four_intersects_env.py<br />
  definition of Gym environment, different reward defintion is implemented here (please fix directory as you need)

  - register.py<br />
  register the new Gym environment. 10 version of the environment is registered for 10 random sets of traffic  scenario (please fix directory as you need)

  - import.py<br />
  import the new Gym environment

sumo_env
  - includes all files for traffic simulation


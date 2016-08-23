import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.stats
import math

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.valid_actions = [None, 'forward', 'left', 'right']
        self.valid_TraficLights = ['green','red'] 
        self.Q={}
        self.alpha=0.8
        self.gamma=0.1
        self.epsilon=1
        self.previous_state=None
        self.previous_action=None
        self.number_of_updates=0
        #self.reset_Q_table(self.stateSpace,  self.valid_actions)
        self.number_of_successful_trials=0
        self.number_of_trials=0
        self.success_rates = [] 
        self.successful_trials=[]
    def reset(self, destination=None):
        self.planner.route_to(destination)
        #TODO: Prepare for a new trip; reset any variables here, if required
        self.previous_state=None
        self.previous_action=None
        self.epsilon = (float)(1)/(self.number_of_trials+1)
    def choose_next_action(self, state_id):
        actions=[]
        #Simulated annealing: draw a random action with probability self.epsilon
        if scipy.stats.bernoulli.rvs(1-self.epsilon):
            QValues=list(self.get_actions(state_id).values())
            actions=list(self.get_actions(state_id).keys())
            actions = [action for action in actions if self.Q[state_id][action] == max(QValues)]
        else:
            actions= list(self.get_actions(state_id).keys())   
        return random.choice(actions)
    def get_actions(self,state_id):
        #lazily expand Q-table once unknown states arrive. 
        #This is not the savest way, since its impossible to validate new states completely
        #but it saves time and space
        if state_id not in self.Q.keys():
            self.Q[state_id]=dict(zip(self.valid_actions,[0]*len(self.valid_actions)))
        return self.Q[state_id]
    def get_q_value(self,state_id,action):
        #use get_actions to initialize Q values if necessary
        return self.get_actions(state_id)[action]
    def set_q_value(self,state_id,action,q_value):
        self.Q[state_id][action]=q_value
    def update(self, t):
        self.next_waypoint = self.planner.next_waypoint()  
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.state=str(inputs)
        action = self.choose_next_action(self.state)
        reward = self.env.act(self, action)
        if self.previous_action is not None and self.previous_state is not None:
            new_q_value=self.alpha*(reward+self.gamma*max([self.get_q_value(self.state,act) for act in self.env.valid_actions])) 
            self.set_q_value(self.previous_state, self.previous_action,new_q_value)
        self.previous_action = action
        self.previous_state = self.state
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        self.calculate_statistics()
    def calculate_statistics(self):
        if self.env.get_deadline(self)==0 or self.env.done:
             self.number_of_trials=self.number_of_trials+1
             if self.env.done:
                 self.number_of_successful_trials= self.number_of_successful_trials+1
             self.success_rates.append((float)(self.number_of_successful_trials)/self.number_of_trials)  
             self.successful_trials.append(self.number_of_successful_trials)  
def display_evaluation_statistics(best_candidates):
    for candidate in best_candidates:
        plt.plot(candidate.successful_trials,label='alpha: '+str(candidate.alpha)+' gamma: '+str(candidate.gamma))
    plt.xlabel('trial')
    plt.ylabel('succesful trials')
    plt.title=('Test')
    plt.legend(loc = 'best').draggable()
    plt.show()
def run():
    """Run the agent for a finite number of trials."""
    evaluation_statistics={}
    best_candidates=[]
    for alpha in np.arange(0.1,1,0.1):
        for gamma in np.arange(0.1,1,0.1):
            e = Environment()  
            a = e.create_agent(LearningAgent) 
            e.set_primary_agent(a, enforce_deadline=True)  
            a.gamma=gamma
            a.alpha=alpha
            sim = Simulator(e, update_delay=0, display=False)  
            sim.run(n_trials=100) 
            if len(best_candidates)<3:
                best_candidates.append(a)
            else:
                min_candidate = min([(i,candidate.successful_trials[-1]) for i,candidate in enumerate(best_candidates)],key=lambda t:t[1])
                if a.successful_trials[-1]>min_candidate[1]:
                    best_candidates[min_candidate[0]]=a
    display_evaluation_statistics(best_candidates)
if __name__ == '__main__':
    run()

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.stats
import math
import types

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  
        self.planner = RoutePlanner(self.env, self)  
        self.Q={}
        self.alpha=0.5
        self.gamma=0.5
        self.epsilon=1
        self.previous_state=None
        self.previous_action=None
        self.number_of_successful_trials=0
        self.number_of_trials=0
        self.successful_trials=[]
        self.number_of_driving_errors=0
    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.previous_state=None
        self.previous_action=None
        self.previous_reward=None
        self.number_of_trials+=1
        self.successful_trials.append(self.number_of_successful_trials)  
        self.number_of_driving_errors=0
    def get_epsilon(self):
        return (float)(1)/(self.number_of_trials)
    def choose_next_action(self, state_id):
        actions=[]
        #Simulated annealing: draw a random action with probability self.epsilon
        if scipy.stats.bernoulli.rvs(1-self.get_epsilon()):
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
            self.Q[state_id]=dict(zip(self.env.valid_actions,[0]*len(self.env.valid_actions)))
        return self.Q[state_id]
    def get_q_value(self,state_id,action):
        #use get_actions to initialize Q values if necessary
        return self.get_actions(state_id)[action]
    def set_q_value(self,state_id,action,q_value):
        self.Q[state_id][action]=q_value
    def calculate_statistics(self):
        if self.env.get_deadline(self)>0 and self.env.done:
             self.number_of_successful_trials+=1
    def update(self, t):
        self.next_waypoint = self.planner.next_waypoint()  
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.state=str(inputs)
        action = self.choose_next_action(self.state)
        reward = self.env.act(self, action)
        if reward>0: self.number_of_driving_errors+=1 
        if self.previous_action is not None:
            new_q_value=self.alpha*(self.previous_reward+self.gamma*max([self.get_q_value(self.state,act) for act in self.env.valid_actions])) 
            self.set_q_value(self.previous_state, self.previous_action,new_q_value)
        self.previous_action = action
        self.previous_state = self.state
        self.previous_reward = reward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        self.calculate_statistics()
def display_evaluation_statistics(best_candidates,figure_name='stats'):
    fig = plt.figure()
    fig.canvas.set_window_title(figure_name) 
    ax = fig.add_subplot(111)
    for candidate in best_candidates:
        ax.plot(candidate.successful_trials,label='alpha: '+str(candidate.alpha)+' gamma: '+str(candidate.gamma))
    ax.set_xlabel('trial')
    ax.set_ylabel('succesful trials')
    #ax.set_title=('Test')
    ax.legend(loc = 'best').draggable()
    plt.show()
def run_with_random_actions():
    e = Environment() 
    a = e.create_agent(LearningAgent)
    #Override agents "get_epsilon"-method such that the actions are chosen completely at random in each iteration
    #This is clearly a hack, but it works and it avoids a seperate agent implementation
    def get_epsilon(self):
        return 1
    a.get_epsilon=types.MethodType(get_epsilon, a) 
    e.set_primary_agent(a, enforce_deadline=False) 
    sim = Simulator(e, update_delay=0, display=False)  
    sim.run(n_trials=100)  
    display_evaluation_statistics([a],"With random action selection")
def run_with_q_learning():
    best_candidates=[]
    for alpha in np.arange(0.1,1,0.1):
        for gamma in np.arange(0.1,1,0.1):
            e = Environment()  
            a = e.create_agent(LearningAgent) 
            e.set_primary_agent(a, enforce_deadline=True)  
            def get_epsilon(self):
                return (float)(1)/(self.number_of_trials)
            a.get_epsilon=types.MethodType(get_epsilon, a) 
            a.gamma=gamma
            a.alpha=alpha
            sim = Simulator(e, update_delay=0, display=False)  
            sim.run(n_trials=100) 
            if len(best_candidates)<5:
                best_candidates.append(a)
            else:
                min_candidate = min([(i,candidate.successful_trials[-1]) for i,candidate in enumerate(best_candidates)],
                                    key=lambda t:t[1])
                if a.successful_trials[-1]>min_candidate[1]:
                    best_candidates[min_candidate[0]]=a
    display_evaluation_statistics(best_candidates,"100 trials for all parameter combinations with Q-learning")
    for i,candidate in  enumerate(best_candidates):
        e = Environment()  
        a = e.create_agent(LearningAgent) 
        e.set_primary_agent(a, enforce_deadline=True)  
        a.gamma=candidate.gamma
        a.alpha=candidate.alpha
        sim = Simulator(e, update_delay=0, display=False)  
        sim.run(n_trials=1000) 
        best_candidates[i]=a
    display_evaluation_statistics(best_candidates,"1000 trials for n best agents with Q-learning")
def run():
    #run_with_random_actions()
    run_with_q_learning()
    """Run the agent for a finite number of trials."""
   
if __name__ == '__main__':
    run()

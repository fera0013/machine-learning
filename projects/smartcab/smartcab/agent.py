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
        self.alpha=0.2
        self.gamma=0.2
        self.epsilon=1
        self.previous_state=None
        self.previous_action=None
        self.number_of_successful_trials=0
        self.number_of_trials=0
        self.successful_trials=[]
        self.number_of_driving_errors=0
        self.driving_errors=[]
    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.previous_state=None
        self.previous_action=None
        self.previous_reward=None
        self.number_of_trials+=1
        self.successful_trials.append(self.number_of_successful_trials)  
        self.driving_errors.append(self.number_of_driving_errors)
        self.number_of_driving_errors=0
    def get_epsilon(self):
        return (float)(1)/(self.number_of_trials)
    def choose_next_action(self, state_id):
        actions=[]
        if scipy.stats.bernoulli.rvs(1-self.get_epsilon()):
            q_values=list(self.get_actions(state_id).values())
            actions=list(self.get_actions(state_id).keys())
            actions = [action for action in actions if self.Q[state_id][action] == max(q_values)]
        else:
            actions= list(self.get_actions(state_id).keys())   
        return random.choice(actions)
    def get_actions(self,state_id):
        #lazily expand Q-table once unknown states arrive. 
        if state_id not in self.Q.keys():
            self.Q[state_id]=dict(zip(self.env.valid_actions,[0]*len(self.env.valid_actions)))
        return self.Q[state_id]
    def get_q_value(self,state_id,action):
        #use get_actions to initialize Q values if necessary
        return self.get_actions(state_id)[action]
    def set_q_value(self,state_id,action,q_value):
        self.Q[state_id][action]=q_value
    def get_state(self):
        can_reach_next_waypoint=False
        if self.next_waypoint=='forward' or self.env.sense(self)[self.next_waypoint]==None:
            can_reach_next_waypoint=True 
        return str((self.env.sense(self)['light'],self.next_waypoint,can_reach_next_waypoint))
    def calculate_statistics(self):
        if self.env.get_deadline(self)>0 and self.env.done:
             self.number_of_successful_trials+=1
    def update(self, t):
        self.next_waypoint = self.planner.next_waypoint()  
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.state= self.get_state() 
        action = self.choose_next_action(self.state)
        reward = self.env.act(self, action)
        if reward<0: self.number_of_driving_errors+=1 
        if self.previous_action is not None:
            new_q_value=self.alpha*(self.previous_reward+self.gamma*max([self.get_q_value(self.state,act) for act in self.env.valid_actions])) 
            self.set_q_value(self.previous_state, self.previous_action,new_q_value)
        self.previous_action = action
        self.previous_state = self.state
        self.previous_reward = reward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        self.calculate_statistics()

def display_evaluation_statistics(agents,figure_name='stats'):
    if hasattr(agents, "__len__"):
        n_columns=3
        if len(agents) < 3: n_columns=len(agents)
        n_rows=int(math.ceil(len(agents)/float(n_columns)))
        f, axarr = plt.subplots(n_rows, n_columns,sharex=True, sharey=True)
        if axarr.ndim==2: axarr=axarr.flat
        f.suptitle(figure_name)
        for i,ax in enumerate(axarr):
            if i<len(agents):
                ax.plot(agents[i].successful_trials)
                ax.plot(agents[i].driving_errors)
                ax.set_title('alpha: '+str(agents[i].alpha)+' gamma: '+str(agents[i].gamma)+' epsilon: '+str(agents[i].get_epsilon()))
                ax.set_xlabel('number of trials')
                if i==0: ax.legend(['#succesful trials (accumulated)','#driving errors'],loc= 'upper left' ).draggable()
        f.tight_layout() 
    else:
        plt.plot(agents.successful_trials)
        plt.plot(agents.driving_errors)
        plt.xlabel('number of trials')
        plt.title('alpha: '+str(agents.alpha)+' gamma: '+str(agents.gamma)+' epsilon: '+str(agents.get_epsilon()))
        plt.suptitle(figure_name)
        plt.legend(['#succesful trials (accumulated)','#driving errors'],loc= 'upper left' ).draggable()
    plt.show()

#Different implementations of agent methods which can be patched to agent objects to create different models
def get_constant_epsilon(self):
    return 1
def get_decaying_epsilon(self):
    return (float)(1)/(self.number_of_trials)
def get_sense_state(self):
<<<<<<< HEAD
    return str(self.env.sense(self))
=======
    return str((self.env.sense(self)))
>>>>>>> 30a19be99a2ca3b3e810cc3abf49c6ee2256d1c7
def get_state_including_next_waypoint(self):
    return str((self.env.sense(self),self.planner.next_waypoint()))


def run_with_random_actions(n_trials):
    e = Environment() 
    a = e.create_agent(LearningAgent)
    a.get_epsilon=types.MethodType(get_constant_epsilon, a) 
    e.set_primary_agent(a, enforce_deadline=False) 
    sim = Simulator(e, update_delay=0, display=False)  
    sim.run(n_trials=n_trials)  
    return a
def run_with_sense_state(n_trials):
    trials=100
    e = Environment() 
    a = e.create_agent(LearningAgent)
    a.get_epsilon=types.MethodType(get_decaying_epsilon, a) 
    a.get_state=types.MethodType(get_sense_state, a) 
    e.set_primary_agent(a, enforce_deadline=True) 
    sim = Simulator(e, update_delay=0, display=False)  
    sim.run(n_trials=n_trials)  
    return a
def run_with_state_including_next_waypoint(n_trials):
    trials=100
    e = Environment() 
    a = e.create_agent(LearningAgent)
    a.get_epsilon=types.MethodType(get_decaying_epsilon, a) 
    a.get_state=types.MethodType(get_state_including_next_waypoint, a) 
    e.set_primary_agent(a, enforce_deadline=True) 
    sim = Simulator(e, update_delay=0, display=False)  
    sim.run(n_trials=n_trials)  
    return a
def run_with_parameter_finetuning(n_trials):
    agents=[]
    for alpha in np.arange(0.1,0.9,0.2):
        for gamma in np.arange(0.1,0.9,0.2):
            e = Environment()  
            a = e.create_agent(LearningAgent) 
            e.set_primary_agent(a, enforce_deadline=True)  
            a.get_epsilon=types.MethodType(get_decaying_epsilon, a )
            a.get_state=types.MethodType(get_state_including_next_waypoint, a) 
            a.gamma=gamma
            a.alpha=alpha
            sim = Simulator(e, update_delay=0, display=False)  
            sim.run(n_trials= n_trials) 
            agents.append(a)
    return agents

def get_best_agents(agents):
    max_success= max([agent.successful_trials[-1] for agent in agents])
    min_traffic_violations = min([agent.driving_errors[-1] for agent in agents])
    return  max_success, min_traffic_violations, [agent for agent in agents if agent.successful_trials[-1]== max_success and agent.driving_errors[-1]== min_traffic_violations]

def run():
    n_trials=100
    dumbcab= run_with_random_actions(n_trials)
    display_evaluation_statistics(dumbcab,"Random actions")
    agent_with_sense_state = run_with_sense_state(n_trials)
    display_evaluation_statistics([dumbcab,  agent_with_sense_state],"Q-learning with 'sense' state")
    agent_with_state_including_next_waypoint = run_with_state_including_next_waypoint(n_trials)
    display_evaluation_statistics([dumbcab, agent_with_state_including_next_waypoint],"Q-learning with state including next waypoint")
    finetuned_agents = run_with_parameter_finetuning(n_trials)
    display_evaluation_statistics([dumbcab]+finetuned_agents,"Q-learning - parameter fine tuning")
    max_success,min_driving_errors, best_agents = get_best_agents(finetuned_agents)
    print "The maximum success rate of {} with the lowest driving errors of {} can be achieved with the following combinations of parameters:".format(max_success, min_driving_errors)
    for agent in best_agents:
        print "alpha: {} - gamma: {}".format(agent.alpha,agent.gamma) 
if __name__ == '__main__':
    run()

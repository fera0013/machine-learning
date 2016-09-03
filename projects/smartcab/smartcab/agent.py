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
            actions= self.env.valid_actions  
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
        return str((self.env.sense(self)))
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
            #The bellman equation
            new_q_value=(1-self.alpha)*self.get_q_value(self.previous_state,self.previous_action)+self.alpha*(self.previous_reward+self.gamma*max([self.get_q_value(self.state,act) for act in self.env.valid_actions])) 
            self.set_q_value(self.previous_state, self.previous_action,new_q_value)
        self.previous_action = action
        self.previous_state = self.state
        self.previous_reward = reward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        self.calculate_statistics()

def display_evaluation_statistics(agents,figure_name='stats'):
    plt.rcParams.update({'font.size': 22})
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
def get_traffic_state(self):
    return str((self.env.sense(self)))
def get_traffic_and_waypoint_state(self):
    return str((self.env.sense(self),self.planner.next_waypoint()))
def get_full_state(self):
    return str((self.env.sense(self),self.planner.next_waypoint(),self.env.get_deadline(self)))

#Generic method to train q-learning models that can be defined through the specification of the parameters
def train_agent(n_trials,
                       state_method, 
                       epsilon_method,
                       alpha=0.2,
                       gamma=0.2,
                       agent=None,
                       enforce_deadline=True,
                       display=False,
                       update_delay=0):
    e = Environment() 
    if agent is None: agent = e.create_agent(LearningAgent)
    agent.get_epsilon=types.MethodType(epsilon_method, agent) 
    agent.get_state=types.MethodType(state_method, agent) 
    e.set_primary_agent(agent, enforce_deadline=True) 
    sim = Simulator(e, update_delay=0, display=False)  
    sim.run(n_trials=n_trials)  
    return agent

def parameter_grid_search(n_trials,state_method,epsilon_method):
    agents=[]
    for alpha in np.arange(0.1,0.9,0.2):
        for gamma in np.arange(0.1,0.9,0.2):
            a=train_agent(n_trials=n_trials,state_method=state_method,epsilon_method=epsilon_method,alpha=alpha,gamma=gamma)
            agents.append(a)
    return agents

def get_best_agents(agents):
    max_success= max([agent.successful_trials[-1] for agent in agents])
    min_traffic_violations = min([agent.driving_errors[-1] for agent in agents])
    return  max_success, min_traffic_violations, [agent for agent in agents if agent.successful_trials[-1]== max_success and agent.driving_errors[-1]== min_traffic_violations]

def run():
    n_trials=100
    dumbcab= train_agent(n_trials=n_trials,state_method=get_traffic_state,epsilon_method=get_constant_epsilon,enforce_deadline=False)
    display_evaluation_statistics(dumbcab,"Random actions")
    agent_with_traffic_state = train_agent(n_trials=n_trials,state_method=get_traffic_state,epsilon_method=get_decaying_epsilon)
    display_evaluation_statistics([dumbcab,agent_with_traffic_state],"Q-learning with traffic state")
    agent_with_traffic_and_waypoint_state =  train_agent(n_trials=n_trials,state_method=get_traffic_and_waypoint_state,epsilon_method=get_decaying_epsilon)
    display_evaluation_statistics([dumbcab,agent_with_traffic_and_waypoint_state],"Q-learning with traffic and waypoint state")
    agent_with_full_state =  train_agent(n_trials=n_trials,state_method=get_full_state,epsilon_method=get_decaying_epsilon)
    display_evaluation_statistics([dumbcab, agent_with_full_state],"Q-learning with full state")
    display_evaluation_statistics([agent_with_traffic_state,agent_with_traffic_and_waypoint_state,agent_with_full_state],"Performance of different state implementations")
    grid_search_agents = parameter_grid_search(n_trials=n_trials,state_method=get_traffic_and_waypoint_state,epsilon_method=get_decaying_epsilon)
    display_evaluation_statistics([dumbcab]+grid_search_agents,"Q-learning - parameter grid search")
    max_success,min_driving_errors, best_agents = get_best_agents(grid_search_agents)
    print "The maximum success rate of {} with the fewest driving errors of {} can be achieved with the following combinations of parameters:".format(max_success, min_driving_errors)
    for agent in best_agents:
        print "alpha: {} - gamma: {}".format(agent.alpha,agent.gamma) 
if __name__ == '__main__':
    run()

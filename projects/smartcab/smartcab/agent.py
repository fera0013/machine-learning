import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.stats

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.valid_actions = [None, 'forward', 'left', 'right']
        self.valid_TraficLights = ['green','red'] 
        self.stateSpace  = {'light': self.valid_TraficLights, 'oncoming': self.valid_actions, 'left': self.valid_actions, 'right': self.valid_actions,'dist' : range(15)}
        self.Q={}
        self.alpha=0.8
        self.gamma=0.1
        self.epsilon=0.1
        self.previousState=None
        self.previousAction=None
        #self.reset_Q_table(self.stateSpace,  self.valid_actions)
        self.number_of_successful_trials=0
        self.number_of_trials=0
        self.success_rates = [] 
    def reset(self, destination=None):
        self.planner.route_to(destination)
        #TODO: Prepare for a new trip; reset any variables here, if required
        self.number_of_successful_trials=0
        self.number_of_trials=0
        self.previousState=None
        self.previousAction=None
    def choose_next_action(self, state):
        actions=[]
        #Simulated annealing: draw a random action with probability self.epsilon
        if scipy.stats.bernoulli.rvs(1-self.epsilon):
            QValues=list(self.get_actions(state).values())
            actions=list(self.get_actions(state).keys())
            actions = [action for action in actions if self.Q[str(state)][action] == max(QValues)]
        else:
            actions= list(self.get_actions(state).keys())   
        return random.choice(actions)
    def get_actions(self,state):
        #lazily expand Q-table once unknown states arrive. 
        #This is not the savest way, since its impossible to validate new states completely
        #but it saves time and space
        assert len(state.keys())>0, "Invalid state!"  #There are more elegant ways to validate parameter types (e.g. decorators), I know ... 
        stateID = str(state)
        if stateID not in self.Q.keys():
            self.Q[stateID]=dict(zip(self.valid_actions,[0]*len(self.valid_actions)))
        return self.Q[stateID]
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        # TODO: Update state
        self.state=inputs
        self.state['dist']=self.env.compute_dist(self.env.agent_states[self]['location'], self.env.agent_states[self]['destination'])
        currentStateID= str(self.state)
        # TODO: Select action according to your policy
        action = self.choose_next_action(self.state)
        # Execute action and get reward
        reward = self.env.act(self, action)
        # TODO: Learn policy based on state, action, reward
        if self.previousAction is not None and self.previousState is not None:
            self.Q[self.previousState][self.previousAction]=self.alpha*(reward+self.gamma*max([self.Q[currentStateID][action] for action in self.env.valid_actions])) 
        self.previousAction = action
        self.previousState =  currentStateID
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        self.calculate_statistics()
    def calculate_statistics(self):
        if self.env.get_deadline(self)==0 or self.env.done:
             self.number_of_trials=self.number_of_trials+1
             if self.env.done:
                 self.number_of_successful_trials= self.number_of_successful_trials+1
             self.success_rates.append((float)(self.number_of_successful_trials)/self.number_of_trials)  
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    #e.enforce_deadline=False
    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    trial_success_rates=[]
    for alpha in np.arange(0,1,0.2):
        for gamma in np.arange(0,1,0.2):
            a.gamma=gamma
            a.alpha=alpha
            a.success_rates = []
            sim.run(n_trials=2)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
            plt.plot(a.success_rates,label='alpha: '+str(alpha)+' gamma: '+str(gamma))
    plt.xlabel('number_of_trials')
    plt.ylabel('success_rate')
    plt.title=('Test')
    plt.legend(loc = 'best').draggable()
    plt.show()
if __name__ == '__main__':
    run()

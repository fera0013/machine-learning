import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import itertools

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.valid_actions = [None, 'forward', 'left', 'right']
        self.valid_TraficLights = ['green','red'] 
        self.stateSpace  = {'light': self.valid_TraficLights, 'oncoming': self.valid_actions, 'left': self.valid_actions, 'right': self.valid_actions}
        self.Q={}
        self.alpha=0
        self.gamma=0
        self.previousState=None
        self.previousAction=None
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.reset_Q_table(self.stateSpace,  self.valid_actions)
        self.alpha=0
        self.gamma=0
        self.previousState=None
        self.previousAction=None
    def choose_action(self, state):
        assert state in self.Q.keys(), "Unknown state!"
        QValues=list(self.Q[state].values())
        actions=list(self.Q[state].keys())
        actionsWithMaxQ = [action for action in actions if self.Q[state][action] == max(QValues)]
        action =  random.choice(actionsWithMaxQ)
        return action
    def reset_Q_table(self,states,actions):
        listOfPossibleStateValues = [states[stateCategory] for stateCategory in states.keys()]
        actionsDict=dict(zip(actions,[0]*len(actions)))
        stateID={}
        for state in itertools.product(*listOfPossibleStateValues):
            for i,partState in enumerate(state):
                stateID[states.keys()[i]]=partState
            self.Q[str(stateID)]=actionsDict.copy()
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        # TODO: Update state
        self.state=inputs
        # TODO: Select action according to your policy
        stateID=str(self.state)
        action = self.choose_action(stateID)
        # Execute action and get reward
        reward = self.env.act(self, action)
        # TODO: Learn policy based on state, action, reward
        if self.previousAction is not None and self.previousState is not None:
            self.alpha=1/2
            self.gamma=1/2
            self.Q[self.previousState][self.previousAction]=self.alpha*(reward+self.gamma*max([self.Q[ stateID][action] for action in self.env.valid_actions])) 
        self.previousAction = action
        self.previousState = stateID  
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    e.enforce_deadline=False
    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

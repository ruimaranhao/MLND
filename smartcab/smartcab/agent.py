import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'yellow'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.preward = 0 #previous reward
        self.paction = None #previous action

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.preward = 0
        self.paction = None

    def selectAction(self, env, naive = False):
        if naive:
            return random.choice([None, 'forward', 'left', 'right'])

        lactions = []

        if(env['light'] == 'red'):
            if(env['oncoming'] != 'left'):
                lactions = [None, 'right']
            else:
                lactions = [None] #I was expecting this to be correct, but it leads to -1 rewards... if lactions = [] then it does not happen... which makes sense (for naive = False). Any ideia why?
        else:
            # traffic ligh is gree and now check for oncoming
            if(env['oncoming'] == 'forward'): #if no oncoming
                lactions = [ 'forward', 'right' ]
            else: #no incoming traffic
                lactions = ['right', 'forward', 'left']

        return random.choice(lactions)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        current_env_state = self.env.sense(self)

        # TODO: Select action according to your policy
        action = self.selectAction(current_env_state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if action:
            self.paction = action
            self.preward = reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

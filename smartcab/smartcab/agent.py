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
        # TODO: Initialize any additional variables here
        self.valid_actions = {}
        self.posreward = 0
        self.negreward = 0
        self.nreached = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None

    def policy(self, state, rnd = False):
        if rnd or len(self.valid_actions) == 0:
            return random.choice(Environment.valid_actions)

        return random.choice(self.valid_actions.get(state, Environment.valid_actions))

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (('nwp', self.next_waypoint),
                      ('left', inputs['light']),
                      ('oncoming', inputs['oncoming']),
                      ('right', inputs['right']),
                      ('left', inputs['left']))

        # TODO: Select action according to your policy
        action = self.policy(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if reward >= 0:
            self.valid_actions[self.state] = list(set(self.valid_actions.get(self.state, []) + [action]))
            self.posreward += 1.0
        else:
            self.negreward += 1.0

        if self.env.get_done():
            self.nreached += 1.0

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print("Deadline: {}, Neg: {} , Pos: {}, Reach: {}".format(deadline,
                            self.negreward / (self.negreward + self.posreward),
                            self.posreward / (self.negreward + self.posreward),
                            self.nreached))


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

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

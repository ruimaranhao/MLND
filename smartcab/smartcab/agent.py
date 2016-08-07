import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, qlearn = False, alpha = 0.5, gamma = 0.2, epsilon = 0.05):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'yellow'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.posreward = 0
        self.negreward = 0
        self.nreached = 0
        self.budget = []
        self.qlearn = qlearn
        if self.qlearn:
            alpha = 0.5
            gamma = 0.2
            epsilon = 0.07
            self.ql = QLearner(Environment.valid_actions, alpha, gamma, epsilon)
        else:
            self.valid_actions = {}


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

        # Update state
        self.state = (('nwp', self.next_waypoint),
                      ('left', inputs['light']),
                      ('oncoming', inputs['oncoming']),
                      ('right', inputs['right']),
                      ('left', inputs['left']))

        # Select action according to your policy
        if self.qlearn:
            action = self.ql.select_action(self.state)
        else:
            action = self.policy(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        if self.qlearn:
            ninputs = self.env.sense(self) #env after action / and state
            nstate = (('nwp', self.next_waypoint),
                      ('left', inputs['light']),
                      ('oncoming', inputs['oncoming']),
                      ('right', inputs['right']),
                      ('left', inputs['left']))
            self.ql.learn(self.state, nstate, action, reward)
        else:
            if reward >= 0:
                self.valid_actions[self.state] = list(set(self.valid_actions.get(self.state, []) + [action]))

        #Collect statistics
        if reward >= 0:
            self.posreward += 1.0
        else:
            self.negreward += 1.0

        if self.env.done: #check if reached destination
            self.nreached += 1.0
            self.budget = self.budget + [deadline]

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print("Deadline: {}, Neg: {} , Pos: {}, Reach: {}".format(deadline,
                            self.negreward / (self.negreward + self.posreward),
                            self.posreward / (self.negreward + self.posreward),
                            self.nreached))


class QLearner():
    def __init__(self, actions, alpha=0.5, gamma=0.2, epsilon=0.05):
        self.q = {}
        self.valid_actions = actions
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount factor
        self.epsilon = epsilon #exploration rate

    def poisson(self, eps):
        return random.random() < eps

    def select_action(self, state):
        if self.poisson(self.epsilon): #select random action
            action = random.choice(self.valid_actions)
        else: # select the best available action
            q = [self.get_q(state, a) for a in self.valid_actions]
            maxq = max(q)

            if q.count(maxq) > 1:
                best = [i for i in range(len(self.valid_actions)) if q[i] == maxq]
                i = random.choice(best)
            else:
                i = q.index(maxq)

            action = self.valid_actions[i]

        return action

    def learn_q(self, state, action, reward, value):
        ovalue = self.q.get((state, action), None)

        if ovalue == None:
            nvalue = reward
        else:
            nvalue = ovalue + self.alpha * (value - ovalue)

        self.set_q(state, action, nvalue) #update

    def learn(self, state, new_state, action, reward):
        q = [self.get_q(new_state, a) for a in self.valid_actions]
        delayed_reward = int(max(q))

        self.learn_q(state, action, reward, reward - self.gamma * delayed_reward)

    #get table value with key
    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    #set table value at key
    def set_q(self, state, action, q):
        self.q[(state, action)] = q

def run():
    """Run the agent for a finite number of trials."""

    qlearner = True

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)

    if qlearner:
        alpha = 0.5
        gamma = 0.2
        epsilon = 0.05
        a = e.create_agent(LearningAgent, qlearner, alpha, gamma, epsilon)
    else:
        a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0005, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylabel('Budget')
    ax.set_xlabel('Trials')
    ax.bar(range(1, len(e.primary_agent.budget) + 1), e.primary_agent.budget, color="blue")
    fig.savefig('performance.png')
    plt.close()


if __name__ == '__main__':
    run()

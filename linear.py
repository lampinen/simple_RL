import numpy as np
import matplotlib.pyplot as plot

class linear_problem(object):
    """Class implementing the linear problem"""
    def __init__(self, length=6, rewards=[0, 0, 0, 0, 0, 1], max_lifetime=100):
       self.min_state = 0 
       self.max_state = length - 1
       if len(rewards) != length:
           raise ValueError("The number of rewards does not match the length... Put zeros on the positions where you don't want a reward")
       self.rewards = rewards # assume that the reward depends only on the state you end up in
       self.max_lifetime = max_lifetime

       self.reset_state()

    def get_state(self):
        """Returns tuple of current state, which in this problem is just position"""
        return (self.x,)

    def reset_state(self):
        """Resets state variables to initial conditions"""
        self.x = 0

    def update_state(self, action):
        """Updates state, returns reward of this state"""
        if action == "left":
            if self.x > self.min_state:
                self.x -= 1
        else: #action == "right"
            self.x += 1

        return self.rewards[self.x]

    def terminal(self):
        """Checks if state is end"""
        return self.x == self.max_state

    def run_trial(self, controller, testing=False):
        self.reset_state()
        total_reward = 0.
        for i in range(self.max_lifetime):
            this_state = self.get_state()
            this_action = controller.choose_action(this_state)
            reward = self.update_state(this_action)
            total_reward += reward
            new_state = self.get_state()

            terminal = self.terminal()
            if not testing:
                controller.update(this_state, this_action, new_state, reward)

            if terminal:
                break

        if testing:
            print("Ran testing trial with %s controller, achieved a total reward of %.2f in %i steps" % (controller.name, total_reward, i+1)) 

        return total_reward, i+1

    def run_k_trials(self, controller, k):
        """Runs k trials, using the specified controller. Controller must have
           a choose_action(state) method which returns one of "left" and
           "right," and must have an update(state, action, next state, reward)
           method (if training=True)."""
        avg_tr = 0.
        avg_time = 0
        for i in range(k):
            (tr, time) = self.run_trial(controller)
            avg_tr += tr
            avg_time += time

        avg_tr /= k
        avg_time /= k
        print("Ran %i trials with %s controller, achieved an average total reward of %.2f in an average of %i steps" % (k, controller.name, avg_tr, avg_time)) 

            

class random_controller(object):
    """Random controller/base class for fancier ones."""
    def __init__(self):
        self.name = "Random"
        self.testing = False

    def set_testing(self):
        """Can toggle exploration, for instance."""
        self.testing = True

    def set_training(self):
        """Can toggle exploration, for instance."""
        self.testing = False

    def choose_action(self, state):
        """Takes a state and returns an action, "left" or "right," to take.
           this method chooses randomly, should be overridden by fancy
           controllers."""
        return np.random.choice(["left", "right"])

    def update(self, prev_state, action, new_state, reward):
        """Update policy or whatever, override."""
        pass

class tabular_Q_controller(random_controller):
    """Tabular Q-learning controller for the linear problem."""
    def __init__(self, possible_states=range(6), epsilon=0.05, gamma=0.9, eta=0.1):
        """Epsilon: exploration probability (epsilon-greedy)
           gamma: discount factor
           eta: update rate"""
        super().__init__()
        self.name = "Tabular Q"
        self.Q_table = {(x,):  {"left": 0.01-np.random.rand()/50, "right": 0.01-np.random.rand()/50} for x in possible_states} 
        self.possible_states = possible_states
        self.str_possible_states = [str(x) for x in possible_states] # for printing
        self.terminal_state = possible_states[-1]
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
 

    def choose_action(self, state):
        """Epsilon-greedy w.r.t the current Q-table."""
        if not self.testing and np.random.rand() < self.epsilon:
            return np.random.choice(["left", "right"])
        else:
            curr_Q_vals = self.Q_table[state]
            if curr_Q_vals["left"] > curr_Q_vals["right"]:
                return "left"
            return "right"

    def update(self, prev_state, action, new_state, reward):
        """Update Q table."""
        if new_state == self.terminal_state:
            target = reward 
        else:
            target = reward + self.gamma * max(self.Q_table[new_state].values())

        self.Q_table[prev_state][action] = (1 - self.eta) * self.Q_table[prev_state][action] + self.eta * target

    def print_pretty_Q_table(self):
        """Prints a Q-table where the L-R dimension represents state and the
           top row represents the Q-value of the "right" action, the bottom row
           represents the Q-value of the "left" action."""
        print("x:\t" + "\t".join(self.str_possible_states))
        right_Qs = map(lambda x: "%.2f" % self.Q_table[(x,)]["right"], self.possible_states[:-1])
        print("right:\t"+ "\t".join(right_Qs) + "\tend") 
        left_Qs = map(lambda x: "%.2f" % self.Q_table[(x,)]["left"], self.possible_states[:-1])
        print("left:\t"+ "\t".join(left_Qs) + "\tend") 


        
if __name__ == "__main__":
    lp = linear_problem()
    np.random.seed(0)
    rc = random_controller()
    rc.set_testing()
    lp.run_trial(rc, testing=True)


    np.random.seed(0)
    tq = linear_tabular_Q_controller()
    for i in range(10):
        tq.set_testing()
        lp.run_trial(tq, testing=True)
        tq.print_pretty_Q_table()
        tq.set_training()
        lp.run_k_trials(tq, 5)
    tq.set_testing()
    lp.run_trial(tq, testing=True)
    tq.print_pretty_Q_table()

    np.random.seed(0)
    tq = linear_tabular_Q_controller()
    lp.run_k_trials(tq, 10000)
    tq.print_pretty_Q_table()

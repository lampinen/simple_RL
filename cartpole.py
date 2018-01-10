import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import deque

class cartpole_problem(object):
    def __init__(self):
        self.delta_t = 0.05
        self.gravity = 9.8
        self.force = 1.
        self.cart_mass = 1.
        self.pole_mass = 0.2
        self.mass = self.cart_mass + self.pole_mass
        self.pole_half_length = 1.

        self.reset_state()

    def get_state(self):
        """Returns current state as a tuple"""
        return (self.x, self.x_dot, self.phi, self.phi_dot)

    def reset_state(self):
        """Reset state variables to initial conditions"""
        self.x = 0.
        self.x_dot = 0.
        self.phi = 0.
        self.phi_dot = 0.

    def tick(self, action):
        """Time step according to EOM and action."""

        if action == "left":
            action_force = self.force
        else:
            action_force = -self.force

        dt = self.delta_t
        self.x += dt * self.x_dot 
        self.phi += dt * self.phi_dot 

        sin_phi = np.sin(self.phi)
        cos_phi = np.cos(self.phi)

        F = action_force + sin_phi * self.pole_mass * self.pole_half_length * (self.phi_dot**2)
        phi_2_dot = (sin_phi * self.gravity - cos_phi * F/ self.mass) / (0.5 * self.pole_half_length * (4./3 - self.pole_mass * cos_phi**2 / self.mass))
        x_2_dot = (F - self.pole_mass * self.pole_half_length * phi_2_dot) / self.mass 
        
        self.x_dot += dt * x_2_dot 
        self.phi_dot += dt * phi_2_dot 
        

    def loses(self):
        """Loses if not within 2.5 m of start and 15 deg. of vertical"""
        return not (-2.5 < self.x < 2.5 and -0.262 < self.phi < 0.262)

    def run_trial(self, controller, testing=False):
        self.reset_state()
        i = 0
        while True:
            i += 1
            this_state = self.get_state()
            this_action = controller.choose_action(this_state)
            self.tick(this_action)
            new_state = self.get_state()

            loss = self.loses()
            reward = -1. if loss else 0.
            if not testing:
                controller.update(this_state, this_action, new_state, reward)

            if loss:
                break

        if testing:
            print("Ran testing trial with %s Controller, achieved a lifetime of %i steps" % (controller.name, i))

        return i


    def run_k_trials(self, controller, k):
        """Runs k trials, using the specified controller. Controller must have
           a choose_action(state) method which returns one of "left" and
           "right," and must have an update(state, action, next state, reward)
           method (if training=True)."""
        avg_lifetime = 0.
        for i in range(k):
            avg_lifetime += self.run_trial(controller)

        avg_lifetime /= k
        print("Ran %i trials with %s Controller, (average lifetime of %f steps)" % (k,  controller.name, avg_lifetime))


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
    """Tabular Q-learning controller."""

    def __init__(self, epsilon=0.05, gamma=0.95, eta=0.1): 
        """Epsilon: exploration probability (epsilon-greedy)
           gamma: discount factor
           eta: update rate"""
        super().__init__()
        self.name = "Tabular Q"
        disc = [-1, 0, 1]
        disc_dot = [-2, -1, 0, 1, 2]
        self.Q_table = {(x, x_dot, phi, phi_dot): {"left": 0.01-np.random.rand()/50, "right": 0.01-np.random.rand()/50} for x in disc for x_dot in disc_dot for phi in disc for phi_dot in disc_dot}
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon

    def discretize_state(self, state):
        """Convert continuous state into discrete with 3 possible values of each
           parameter"""
        x, x_dot, phi, phi_dot = state
        if x > 1.:
            x = 1
        elif x < -1.:
            x = -1
        else: 
            x = 0

        if x_dot < -0.1:
            x_dot = -2
        elif x_dot > 0.1:
            x_dot = 2
        elif x_dot < -0.03:
            x_dot = -1
        elif x_dot > 0.03:
            x_dot = 1
        else:
            x_dot = 0

        if phi > 0.1:
            phi = 1
        elif phi < -0.1:
            phi = -1
        else: 
            phi = 0

        if phi_dot < -0.1:
            phi_dot = -2
        elif phi_dot > 0.1:
            phi_dot = 2
        elif phi_dot < -0.03:
            phi_dot = -1
        elif phi_dot > 0.03:
            phi_dot = 1
        else:
            phi_dot = 0
        
        return (x, x_dot, phi, phi_dot)

    def choose_action(self, state):
        """Epsilon-greedy w.r.t the current Q-table."""
        state = self.discretize_state(state)
        if not self.testing and np.random.rand() < self.epsilon:  
            return np.random.choice(["left", "right"])
        else:
            curr_Q_vals = self.Q_table[state]
            if curr_Q_vals["left"] > curr_Q_vals["right"]:
                return "left"
            return "right"

    def update(self, prev_state, action, new_state, reward):
        """Update Q table."""
        prev_state = self.discretize_state(prev_state)
        new_state = self.discretize_state(new_state)
        if reward != 0.:
            target = reward # reward states are terminal in this task
        else:
            target = self.gamma * max(self.Q_table[new_state].values())

        self.Q_table[prev_state][action] = (1 - self.eta) * self.Q_table[prev_state][action] + self.eta * target 


class dqn_controller(random_controller):
    def __init__(self, epsilon=0.1, gamma=0.95, eta=0.001, nh1=100, nh2=100, replay_buffer=True): 
        """Epsilon: exploration probability (epsilon-greedy)
           gamma: discount factor
           eta: learning rate,
           nh1: number of hidden units in first hidden layer,
           nh2: number of hidden units in second hidden layer,
           replay_buffer: whether to use a replay buffer"""
        super().__init__()
        self.name = "DQN"
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon

        if replay_buffer:
            self.replay_buffer = deque()
            self.replay_buffer_max_size = 1000
        else:
            self.replay_buffer = None

        self.input = tf.placeholder(tf.float32, [1, 4])
        h1 = slim.layers.fully_connected(self.input, nh1, activation_fn=tf.nn.tanh)
        h2 = slim.layers.fully_connected(h1, nh2, activation_fn=tf.nn.tanh)
        self.Q_vals = slim.layers.fully_connected(h2, 2, activation_fn=tf.nn.tanh)

        self.target =  tf.placeholder(tf.float32, [1, 2])
        self.loss = tf.nn.l2_loss(self.Q_vals - self.target)
        optimizer = tf.train.AdamOptimizer(self.eta)
        self.train = optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def choose_action(self, state):
        """Takes a state and returns an action, "left" or "right," to take.
           epsilon-greedy w.r.t current Q-function approx."""
        if not self.testing and np.random.rand() < self.epsilon:  
            return np.random.choice(["left", "right"])
        else:
            curr_Q_vals = self.sess.run(self.Q_vals, feed_dict={self.input: np.array(state, ndmin=2)}) 
            if curr_Q_vals[0, 0] > curr_Q_vals[0, 1]:
                return "left"
            return "right"

    def update(self, prev_state, action, new_state, reward):
        """Update policy or whatever, override."""
        if self.replay_buffer is not None:
            self.replay_buffer.append((prev_state, action, new_state, reward))
            (prev_state, action, new_state,reward) = numpy.random.choice(replay_buffer) 

            if len(self.replay_buffer) > self.replay_buffer_max_size:
                self.replay_buffer.popleft()

        if reward != 0.:
            target_val = reward # reward states are terminal in this task
        else:
            new_Q_vals = self.sess.run(self.Q_vals, feed_dict={self.input: np.array(new_state, ndmin=2)}) 
            target_val = self.gamma * np.max(new_Q_vals)

        # hacky way to update only the correct Q value: make the target for the
        # other its current value
        target_Q_vals = self.sess.run(self.Q_vals, feed_dict={self.input: np.array(prev_state, ndmin=2)})
        if action == "left":
            target_Q_vals[0, 0] = target_val
        else:
            target_Q_vals[0, 1] = target_val
        
        self.sess.run(self.train, feed_dict={self.input: np.array(prev_state, ndmin=2), self.target: target_Q_vals.reshape([1,2])})

             

if __name__ == "__main__":
    np.random.seed(0)
    tf.set_random_seed(0)
    cpp = cartpole_problem()

    cpc = random_controller()
    cpp.run_trial(cpc, testing=True)

    dqn = dqn_controller(replay_buffer=False)
    dqn.set_testing()
    cpp.run_trial(dqn, testing=True)
    for i in range(10):
        dqn.set_training()
        cpp.run_k_trials(dqn, 1000)
        dqn.set_testing()
        cpp.run_trial(dqn, testing=True)
    exit() 
    tqc = tabular_Q_controller()
    tqc.set_testing()
    cpp.run_trial(tqc, testing=True)
    tqc.set_training()
    cpp.run_k_trials(tqc, 10000)
    tqc.set_testing()
    cpp.run_trial(tqc, testing=True)



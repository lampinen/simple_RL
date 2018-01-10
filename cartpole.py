import numpy as np


class cartpole_controller(object):
    """Random controller/base class for fancier ones."""
    def __init__(self):
        self.name = "Random"

    def choose_action(self, state):
        """Takes a state and returns an action, "left" or "right," to take.
           this method chooses randomly, should be overridden by fancy
           controllers."""
        return np.random.choice(["left", "right"])

    def update(self, prev_state, action, new_state, reward):
        """Update policy or whatever, override."""
        pass

class cartpole_problem(object):
    def __init__(self):
        self.delta_t = 0.01

        self.gravity = 9.8

        self.force = 1.
        self.cart_mass = 1.
        self.pole_mass = 0.2
        self.mass = self.cart_mass + self.pole_mass
        self.pole_half_length = 1.

        self.reset_state()

    def get_state(self):
        return [self.x, self.x_dot, self.phi, self.phi_dot]

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

    def run_trial(self, controller, training):
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
            if training:
                controller.update(this_state, this_action, new_state, reward)

            if loss:
                break

        return i


    def run_k_trials(self, controller, k, training=False):
        avg_lifetime = 0.
        for i in range(k):
            avg_lifetime += self.run_trial(controller, training=training)

        avg_lifetime /= k
        print("Ran %i trials with %s Controller, with an average length of %i steps" % (k, controller.name, avg_lifetime))


if __name__ == "__main__":
    cpp = cartpole_problem()
    cpc = cartpole_controller()
    cpp.run_k_trials(cpc, 1000)

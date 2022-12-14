import random

import numpy as np


class StochasticInventoryProblem:
    def __init__(self, T=1000, x_max=100):
        # Number of time steps (days)
        self.T = T

        # State space
        self.x_max = x_max

        # Probability of demand = 1 by time step
        self.P = [_ * 0.001 for _ in range(T)]

        # Simulated demand
        self.D = [int(random.uniform(0, 1) < self.P[_]) for _ in range(T)]

        # Holding cost
        self.h = 0.01

        # Order cost
        self.K = 10

        # Dynamic buying price and static selling price
        self.selling_price = 20
        self.purchase_cost = lambda _: 10 if _ <= 500 else 15

        # Value for each time and state
        self.V = np.zeros((self.x_max + 1, self.T))

        # Optimal policy for each time and state
        self.alpha = np.zeros((self.x_max + 1, self.T))

        # Value for each action at a given time and state
        self.Q = np.zeros((self.x_max + 1))

        # The value at the last time step is the value of the inventory
        self.V[0 : self.x_max, self.T - 1] = [self.h * _ for _ in range(self.x_max)]

    # Reset values to run again with different parameters
    def reset(self):
        self.__init__(T=self.T)

    # Apply dynamic programming to find the optimal policy and its value
    def find_optimal_policy(self):
        # For each time step, going backwards from t = T - 1 to t = 0
        for t in reversed(range(0, self.T - 1)):
            # print(f"[{t}/{self.T - 2}]")

            # For each state (number of items in inventory)
            for x in range(0, self.x_max + 1):
                # Determine the action space A
                a_min = max(self.D[t] - x, 0)
                a_max = self.x_max - x + self.D[t]

                # Buy nothing starting from t=900
                if t >= 900:
                    a_max = 1

                # For each legal action (number of items ordered)
                for a in range(a_min, a_max):
                    # current stock * holding cost
                    holding_costs = x * self.h

                    # if action do order_costs
                    order_costs = self.K * int(a > 0)

                    # t + 1 = next time step (looking forward)
                    next_value = self.V[x - self.D[t] + a, t + 1]

                    # purchase_cost = (t<500: 10, t> 500: 15)
                    # cost to purchase new stock * action
                    acquire_cost = self.purchase_cost(t) * a

                    # earnings (selling price * demand if you have inventory)
                    earnings = self.selling_price * self.D[t] if x > 0 else 0

                    # Calculate value for this time step
                    self.Q[a] = (
                        holding_costs
                        + order_costs
                        + next_value
                        + acquire_cost
                        - earnings
                    )

                # Initialize current best value and optimal policy
                self.V[x, t] = self.Q[a_min]
                self.alpha[x, t] = a_min

                # Choose best action and optimal policy for this state and time step
                for a in range(a_min + 1, a_max):
                    if self.Q[a] < self.V[x, t]:
                        self.alpha[x, t] = a
                        self.V[x, t] = self.Q[a]

    # Perform n simulations of the demand over time
    def simulate(self, n=10):
        for _ in range(n):
            print(f"current run: {_+1}")
            self.reset()
            self.find_optimal_policy()
            self.export(run_nr=_)

    # Export the value function and optimal policy
    def export(self, run_nr=0):
        np.savetxt(f"value_{run_nr}.csv", self.V, fmt="%d", delimiter=",")
        np.savetxt(f"policy_{run_nr}.csv", self.alpha, fmt="%d", delimiter=",")
        np.savetxt(f"demand_{run_nr}.csv", self.D, fmt="%d", delimiter=",")


if __name__ == "__main__":
    s = StochasticInventoryProblem()
    s.simulate()

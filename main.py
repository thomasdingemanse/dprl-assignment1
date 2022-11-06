import numpy as np
import random

# T     = Time periods
# x_max = max inventory
# d     = demand
# h     = holding cost
# K     = order cost
# V     = value by time and state
# alpha = optimal policy by time and state
# Q     = vector with value by action

# Number of time steps (days)
T = 1000

# State space
x_max = T

# Probability of demand = 1 by time step
P = [_ * 0.001 for _ in range(T)]

# Simulated demand
D = [int(random.uniform(0, 1)< P[_]) for _ in range(T)]

# Holding costs
h = 0.01

# Order cost
K = 10

selling_price = 20
purchase_cost = lambda _ : 10 if _ <= 500 else 15

# Value by time and state
V = np.zeros((T, x_max))

# Optimal policy by time and state
alpha = np.zeros((T, x_max))

# Value by action at a given time and state
Q = np.zeros((x_max))

# Set value for last time step (value = items sold * price)
V[0:,T-1] = [h * _  for _ in range(x_max)]

# For each time step, going backwards from t = T to t = 0
for t in reversed(range(0, T-1)):
    # For each state (number of items in inventory)
    for x in range(0, x_max):
        # Determine the action space A
        a_min = max(D[t]-x, 0)
        a_max = x_max - x + D[t]

        # For each legal action (number of items ordered)
        for a in range(a_min, a_max):
            # Calculate value for the given action
            holding_costs = x * h

            # No purchases possible after time 900
            order_costs = K * int(a > 0) if t <= 900 else 0
            
            # t + 1 = next time interval (looking forward)
            Q[a] = holding_costs + order_costs + V[x - D[t] + a, t + 1]
        
        # Initialize current best value and optimal policy
        V[x,t] = Q[a_min]
        alpha[x,t] = a_min
        
        # Choose best action and optimal policy for this state and time step
        for a in range(a_min, a_max):
            if Q[a] < V[x,t]:
                alpha[x,t] = a
                V[x,t] = Q[a]

# Export the value function and optimal policy
np.savetxt("actions.csv", V, fmt='%d', delimiter=",")
np.savetxt("optimal.csv", alpha, fmt='%d', delimiter=",")

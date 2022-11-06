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

T = 10

# We can buy any number of items, but since demand is at most 1 for each time
# step
x_max = T

# Probability of d=1 by time step
P = [_ * 0.001 for _ in range(T)]
D = [int(random.uniform(0, 1)< P[_]) for _ in range(T)]

h = 0.01
selling_price = 20
purchase_cost = lambda _ : 10 if _ <= 500 else 15
K = 10

# Value function
V = np.zeros((T, x_max))

# Optimal policy
alpha = np.zeros((T, x_max))

# vector with value of actions
Q = np.zeros((x_max))

# Set value for last time step (value = items sold * price)
V[0:,T-1] = [h * _  for _ in range(x_max)]

# For each time step (going backwards from t=T)
for t in reversed(range(1, T-1)):
    # For each state (number of items in inventory)
    d = D[t]
    for x in range(0, x_max):
        # For each possible action (number of items ordered)
        a_min = max(d-x, 0)
        a_max = x_max - x + d
        for a in range(a_min, a_max):
            # Calculate value for the given action
            holding_costs = x * h

            # no purchases possible after time 900
            order_costs = K * int(a > 0) if t <= 900 else 0
            # t+1 = next time interval (looking forward)
            Q[a] = holding_costs + order_costs + V[x - d + a, t + 1]
        
        # Initialize current best value and policy based on first action
        V[x,t] = Q[a_min]
        alpha[x,t] = a_min
        
        # Choose best action for this state and time step
        for a in range(a_min, a_max):
            if Q[a] < V[x,t]:
                alpha[x,t] = a
                V[x,t] = Q[a]

print(V[0:,5:10].astype(int))
print(alpha[0:,5:10].astype(int))
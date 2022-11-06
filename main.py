import numpy as np

# T     = Time periods
# x_max = max inventory
# d     = demand
# h     = holding cost
# K     = order cost
# V     = value by time and state
# alpha = optimal policy by time and state
# Q     = vector with value by action

T = 10
x_max = 10

# Parameters
d = 2
h = 1
K = 5

# Value function
V = np.zeros((T, x_max))

# Optimal policy
alpha = np.zeros((T, x_max))

# vector with value of actions
Q = np.zeros((100))

# Set value for last time step (value = items sold * price)
V[0:,T-1] = [h * _  for _ in range(x_max)]

# For each time step (going backwards from t=T)
for t in reversed(range(1, T-1)):
    # For each state (number of items in inventory)
    for x in range(0, x_max):
        # For each possible action (number of items ordered)
        a_min = max(d-x, 0)
        a_max = x_max - x + d
        for a in range(a_min, a_max):
            # Calculate value for the given action
            holding_costs = x * h
            order_costs = K * int(a > 0)
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
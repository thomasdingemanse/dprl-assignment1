import pandas as pd
from numpy import genfromtxt

df_data = {}

for _ in range(10):
    df = pd.read_csv(f"Data/policy_{_}.csv")
    df.columns = [str(_) for _ in range(1000)]
    demand = genfromtxt(f'Data/demand_{_}.csv', delimiter=',')

    inventory = []
    actions = []
    missed_sales = []
    cur_inv = 0
    missed = 0
    for i, dem in enumerate(demand):
        inventory.append(cur_inv)

        # get time possibilities
        temp = df[f"{i}"]

        # get action amount
        action = temp[cur_inv]
        actions.append(action)

        cur_inv = max(cur_inv + action - dem, 0)

        if cur_inv + action < dem:
            missed += 1
        missed_sales.append(missed)

    df_data[f"inventory {_}"] = inventory
    df_data[f"actions {_}"] = actions
    df_data[f"missed sales {_}"] = missed_sales

res = pd.DataFrame(df_data)
res.to_csv(f"Data/missed.csv")
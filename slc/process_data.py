# %%
import pandas as pd

# %%
df = pd.read_csv('data/train_data.csv', sep='|',
                 #  nrows=10,
                 )


# %%
df.iloc[0]

# %%

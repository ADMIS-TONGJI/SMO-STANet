import numpy as np
import pandas as pd

df_raw = pd.read_csv('./chlor/final_south.csv')

cols_data = df_raw.columns[2:2306]
data = np.array(df_raw[cols_data])
print(np.mean(data))
print(np.median(data))
print(np.max(data))
print(data.shape)
print(data.reshape(data.shape[0]*data.shape[1]).shape)
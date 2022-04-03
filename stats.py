import subprocess
import pandas as pd
import numpy as np

data=pd.read_csv('data_train.csv')
data_2=pd.read_csv('data_dev.csv')
data_3=pd.read_csv('data_test.csv')
data = pd.concat([data, data_2, data_3], axis=0)
data = data.replace(np.nan, '', regex=True)

print("="*20)
print('Ilość wierszy w zbiorze: ',len(data))

print("="*10, ' data["department"].value_counts() ', 10*'=')
print(data["department"].value_counts())

print("="*10, ' data.median() ', 10*'=')
print(data.median())

print("="*10, ' data.describe(include="all") ', 10*'=')
print(data.describe(include='all'))

data.describe(include="all").to_csv(r'stats.txt', header=None, index=None, sep='\t', mode='a')
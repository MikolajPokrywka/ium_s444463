import subprocess
import pandas as pd
import numpy as np
import os


path = ''

all_files = ['column_titles.csv', 'data_train.csv', 'data_dev.csv', 'data_test.csv']

data_file = open("data.csv", "w")
for name in all_files:
    f = open(name, "r")
    data_file.write(f.read())
    f.close()

data_file.close()
data=pd.read_csv('data.csv')
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
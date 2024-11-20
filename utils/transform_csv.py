import pandas as pd

dataframe = pd.read_csv('metadata.csv')

dataframe.to_csv('metadata2.csv', index=False, sep=';')
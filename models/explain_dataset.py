import pandas as pd
import numpy as np

def createDatasets(folder="google"):
  df_p = pd.read_csv(folder+'/positive.csv')
  df_n = pd.read_csv(folder+'/negative.csv')
  df_merged = df_p.append(df_n, ignore_index=True)
  y = df_merged['relevant']
  X = df_merged[['local_tuple', 'external_tuple']]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
  train = pd.concat([X_train, y_train], axis=1)
  test = pd.concat([X_test, y_test], axis=1)
  return train, test



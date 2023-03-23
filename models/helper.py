import numpy as np
import pandas as pd



path="datasets/"
def createDatasets(folder="google"):
  df_p = pd.read_csv(folder+'/positive.csv')
  df_n = pd.read_csv(folder+'/negative.csv')
  print("Length of",folder,"positive",len(df_p))
  print("Length of",folder,"negative",len(df_n))




def main():
  createDatasets(path + "google")
  createDatasets(path + "news")
  createDatasets(path + "drugs")

if __name__== "__main__":
    main()
import numpy as np
import pandas as pd
import stock_data

def pca(df):
  df = df.apply(lambda x: (x-x.mean())/x.std())
  cov = np.cov(df.T)
  eigvals,eigvecs = np.linalg.eig(cov)
  weights = eigvals / sum(eigvals) 
  return (eigvals,eigvecs,weights)


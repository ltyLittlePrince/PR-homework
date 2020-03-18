import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pre-processing
df = pd.read_csv('data.csv')
N = len(df)
df.index = range(1,N+1) # edit index
df = df.reindex(columns=df.columns.difference(['index']))
# sort by predicted values
df = df.sort_values(by=['score'], ascending=False)
n_positive = len(df[df.label==1]) # count m+, m-
n_negative = 500 - n_positive

# Caculate updated TP and FP predictions
df = df.reindex(columns=list(df.columns)+['TP','FP'])
df.TP = df.label==1
df.TP = df.TP.cumsum()
df.FP = df.label==2
df.FP = df.FP.cumsum()

# Caculate always up-to-date Precision and Recall
df = df.reindex(columns=list(df.columns)+['Precision','Recall'])
df.Precision = df.TP / (df.TP + df.FP)
df.Recall = df.TP / n_positive

# Calculate AUC_PR
lp, lr, AUC_PR = 1.0, 0.0, 0.0
for row_index, row in df.iterrows():
	AUC_PR += (row['Recall']-lr)*(row['Precision']+lp)/2
	lp, lr = row['Precision'], row['Recall']
print('AUC_PR: %f' % (AUC_PR))

# Calculate AP
lr, AP = 0.0, 0.0
for row_index, row in df.iterrows():
	AP += (row['Recall']-lr)*row['Precision']
	lr = row['Recall']
print('AP: %f' % (AP))
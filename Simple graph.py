import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('/home/student/data.csv')
df.sample(5)
print(df.shape)
print(df['Make'].value_counts())
new_df=df[df['Make']=='Volkswagen']
print(new_df.shape)
print(new_df.isnull().sum())
new_df=new_df.dropna()
new_df.shape
new_df.isnull().sum()
new_df.sample(2)
new_df=new_df[['Engine HP','MSRP']]
new_df.sample(5)
x=np.array(new_df[['Engine HP']])
y=np.array(new_df[['MSRP']])
print(x.shape)
print(y.shape)
plt.scatter(x,y,color="red")
plt.title('HP vs MSRP')
plt.xlabel('HP')
plt.ylabel('MSRP')
plt.show()


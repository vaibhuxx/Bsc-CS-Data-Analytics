Practical 1

Set A

Q.1. Create ‘sales’ Data set having 5 columns namely: ID, TV, Radio, Newspaper and Sales.(random 500 entries).
Build a linear regression model by identifying independent and target variable. 
Split the variables into training and testing sets then divide the training and testing sets into a 7:3 ratio, respectively and print them. 
Build a simple linear regression model. 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
ID=random.sample(range(0,500),500)
flat=random.sample(range(200,800),500)
houses=random.sample(range(100,900),500)
purchases=random.sample(range(100,600),500)
data=list(zip(ID,flat,houses,purchases))
df=pd.DataFrame(data,columns=['ID','TV','Radio','Newspaper','Sales'])
print(df)
x=np.array(df[['flat']])
x=np.array(df[['purchases']])
print(x.shape)
print(y.shape)
plt.scatter(x,y,color="red")
plt.title('flat vs purchases')
plt.xlabel('flat')
plt.ylabel('purchases')
plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=15)
regressor=LinearRegression()
regressor.flt(x_train,y_train)
plt.scatter(x_test,y_test,color="green")
plt.plot(x_train,regressor.predict(x_train),color="red",linewidth=3)
plt.title('Regression(trainingSet)')
plt.xlabel('flat')
plt.ylabel('purchases')
plt.show()
y_pred=regressor.predict(x_test)
print('R2 Score:%.2f'%r2_score(y_test,y_pred))
print('MeanError:',mean_squared_error(y_test,y_pred))
def flat_price(flat):
    result=regressor.predict(np.array(flat).reshape(1,1))
    return(result[0,0])
flat_unit=int(input('Enter number of flat:'))
print('This flat price will be:',int(flat_price(flat_unit))*10,'Rs')



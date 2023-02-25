Practical 1

Set A

Q.2.Create ‘realestate’ Data set having 4 columns namely: ID,flat, houses and purchases (random 500 
entries). Build a linear regression model by identifying independent and target variable. Split the 
variables into training and testing sets and print them. Build a simple linear regression model for 
predicting purchases

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
ID=random.sample(range(0,500),500)
flat=random.sample(range(200,800),500)
house=random.sample(range(100,900),500)
purchase=random.sample(range(100,600),500)
realestate=list(zip(ID,flat,house,purchase))
df=pd.DataFrame(realestate,columns=['ID','flat','house','purchase'])
print(df)
X=np.array(df[['flat']])
y=np.array(df[['purchase']])
print(X.shape)
print(y.shape)
plt.scatter(X,y,color="red")
plt.title('flat vs purchase')
plt.xlabel('flat')
plt.ylabel('purchase')
plt.show()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=15)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
plt.scatter(X_test,y_test,color="green")
plt.plot(X_train,regressor.predict(X_train),color="red",linewidth=3)
plt.title('Regression(TestSet)')
plt.xlabel('flat')
plt.ylabel('purchase')
plt.show()
plt.scatter(X_train,y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="red",linewidth=3)
plt.title('Regression(trainingSet)')
plt.xlabel('flat')
plt.ylabel('purchase')
plt.show()
y_pred=regressor.predict(X_test)
print('R2 score:%.2f'%r2_score(y_test,y_pred))
print('MeanError:',mean_squared_error(y_test,y_pred))
def flat_price():
    result=regressor.predict(np.array(tv).reshape(1,-1))
    return(result[0,0])
flat_unit=int(input('Enter number of flat :'))
print('This flat Prplt.scatter(X_train,y_train,color="blue")')
plt.plot(X_train,regressor.predict(X_train),color="red",linewidth=3)
plt.title('Regression(trainingSet)')
plt.xlabel('flat')
plt.ylabel('purchase')
plt.show()
y_pred=regressor.predict(X_test)
print('R2score:%.2f'%r2_score(y_test,y_pred))
print('MeanError:',mean_squared_error(y_test,y_pred))
  def TV_price(tv):result=regressor.predict(np.array(tv).reshape(1,-1))
   return(result[0,0])
flat_unit=int(input('Enter number of TVs :'))
print('This TV Price will be :' ,int(flat_price(flat_unit))*10,'₹')

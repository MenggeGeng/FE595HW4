#part 1: Assume the data in the Boston Housing data set fits a linear model.
#When fit, how much impact does each factor have on the value of a house in Boston?
#Display these results from greatest to least.


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#load boston dataset
boston = load_boston()
#print(boston)

data = pd.DataFrame(boston['data'], columns= boston['feature_names'])
print(data.head())

data['target'] = boston['target']
print(data.head())

# The correlation map
correlation = data.corr()
sns.heatmap(data = correlation, annot = True)
plt.show()
# The correlation coefficient between target and features
print(correlation['target'])

#prepare the regression objects
X = data.iloc[:, : -1]
Y = data['target']
print(X.shape)
print(Y.shape)

#data segmentation
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5, random_state= 1)

#build regression model
reg_model = LinearRegression()
reg_model.fit(X,Y)
print('coefficient : \n ', reg_model.coef_)

#The impact of each factor have on the value of a house in Boston
coef = pd.DataFrame(data= reg_model.coef_, index = boston['feature_names'], columns=["value"])
print(coef)

#Display these results from greatest to least.
coef_abs = abs(reg_model.coef_)
coef_1 = pd.DataFrame(data= coef_abs , index = boston['feature_names'], columns=["value"])
print("Display these coefficient from greatest to least:")
print(coef_1.sort_values("value",inplace=False, ascending=False))




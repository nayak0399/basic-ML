#import_libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import csv
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from numpy import arange,array,ones
from scipy import stats
from scipy.interpolate import *
import statistics as st

#####################################################################################
#please mention data file name here
filename ='oxygen-2.csv' 
dataset = pd.read_csv(filename)

### check if any data is missing
dataset.isna().sum() 
### remove the line with missing data if any 
dataset = dataset.dropna()
#Remove the column with only text  and target column in X_1 by using their header
X_1 = dataset.drop(['Adsorbate','Surface Material','EXX+RPA'], axis=1)
#mention the column header of target here
Y = dataset['EXX+RPA']
X = preprocessing.scale(X_1)

testing_data_fraction=0.25

for n in range (1,10,1):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = testing_data_fraction, random_state=n)

##################################################################################################
### ## model Here

#    model = LinearRegression(fit_intercept=True)

    model = PLSRegression(n_components=5)

#    model = Ridge(alpha=0.0000001, max_iter=10000, tol=0.001)

#    model = Lasso(alpha=0.00001,max_iter=100000)

#    model = KernelRidge(kernel='polynomial', alpha=0.000000001, degree= 5)
#    model = KernelRidge(kernel='rbf', gamma=0.001, alpha=0.0001)

#    model = GradientBoostingRegressor(n_estimators=500, max_depth = 10, learning_rate=0.01)

#    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#    model = GaussianProcessRegressor(kernel='RBF', n_restarts_optimizer=9)

#    model = GradientBoostingRegressor(n_estimators=1000, max_depth = 4, learning_rate=0.01)
#    model = RandomForestRegressor(n_estimators=1000, max_features=2,max_depth=8)
###ANN with scikit-learn
    #model=MLPRegressor(activation='logistic',solver='adam',hidden_layer_sizes=(1000,),max_iter=1000,random_state=4) 

    model.fit(X_train, Y_train)
    
####################################################################################################
### ## model evaluation for training set
    y_train_predict = model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    r2 = r2_score(Y_train, y_train_predict)
    y_test_predict = model.predict(X_test)
    rmse_test = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    r2_test = r2_score(Y_test, y_test_predict)
####################################################################################################
# print 
# print '%s' %n, '{}'.format(rmse) ,'{}'.format(r2), '{}'.format(rmse_test), '{}'.format(r2_test)
    


##########################################################################################################
### ## write output in a file named out.csv in same directory
### ## here we import the out.csv file and print the statistics
##write output in a file named out.csv in same directory
    with open('out.csv', mode='a') as out_file:
      out_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)      
      out_writer.writerow(['%s' %n, '{}'.format(rmse) ,'{}'.format(r2), '{}'.format(rmse_test), '{}'.format(r2_test)])
      out_file.close()

from shutil import copyfile
copyfile('out.csv', 'cross-validation-out.csv')
########################################################################################################################
###here we import the out.csv file and printthe statistics

   
data_stat=pd.read_csv('out.csv',  names = ["RMSE-Train", "R^square train", "RMSE-test", "R^square test"])  
print data_stat.describe()
data_stat.describe().to_csv("stat-results.csv")
#Removing out.csv file
try:
    os.remove("out.csv")
except OSError:
    pass

##########################################################################################################
##########################################################################################################
############ This section of the script is for an individual random state 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = testing_data_fraction, random_state=88)

#print(X_train.shape, Y_train.shape)
#print(X_test.shape, Y_test.shape)

##########################################################################################################
#####Define the regression model here
 
model = model
model.fit(X_train, Y_train)

##### model evaluation for training set
y_train_predict = model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
#####################################################
# model evaluation for testing set
y_test_predict = model.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2_test = r2_score(Y_test, y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse_test))
print('R2 score is {}'.format(r2_test))
#print (Y_test,y_test_predict)

##########################################################################################################

######printing coefficient of the regression
print X_1.columns.values.tolist()
print("-----------------------------------------------")
print("-----------------------------------------------")

try:
    print ("The Dual coefficients are", model.dual_coef_)    
except AttributeError:
    pass
try:
    print ("The model coefficients are {}".format(model.coef_))
except AttributeError:
    pass
try:
    print ("The model coefficients are", model.feature_importances_)
except AttributeError:
    pass
############################################################################################################

#### define your data plot here       
plt.figure(figsize=(4,4),dpi=300)
plt.title('ANN-Scikit Learn', fontsize=13) 
#### define your data plot here       
plt.xlim((-3,5))
plt.xticks(np.linspace(-3,5,5,endpoint=True))
plt.ylim((-3,5))
plt.yticks(np.linspace(-3,5,5,endpoint=True))
plt.tick_params(axis='both', which='major', labelsize=13)
plt.scatter(Y_train, y_train_predict, label='Training',facecolors='green',alpha=0.8, edgecolors='none',s=60,marker='^')
plt.scatter(Y_test, y_test_predict, label='Testing',facecolors='blue',alpha=0.75, edgecolors='none',s=60, marker='o')
plt.xlabel("$E_{ads}$ RPA calculated (eV)",fontsize=13)
plt.ylabel("$E_{ads}$ ML predicted (eV)",fontsize=13)

#####fitting with a y=x line here (polynomial approach)
#p1 = np.polyfit(Y_train, y_train_predict,1)
plt.plot((-3,5),(-3,5), 'r-',linestyle='dashed',label='y=x line')
plt.tight_layout()
plt.legend(loc=2, prop={'size': 13}, frameon=False)
#plt.text(1, -1, 'RMSE=0.19 eV \n R$^2$=0.96',fontsize=13)
plt.savefig ('mol-ANN-SK.eps')
plt.show()
##########################################################################################################



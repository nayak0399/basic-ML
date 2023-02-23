#import_libraries
##https://www.tensorflow.org/tutorials/keras/basic_regression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import csv
import pathlib
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from numpy import arange,array,ones
from scipy import stats
from scipy.interpolate import *
import statistics as st
###############################
### ## tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#####################################################################################
#please mention data file name here
filename ='oxygen-2.csv' 
dataset = pd.read_csv(filename)
dataset = dataset.drop(['Adsorbate','Surface Material'], axis=1)
### check if any data is missing
dataset.isna().sum() 
### remove the line with missing data if any 
dataset = dataset.dropna()
#####################################################################################

train_dataset = dataset.sample(frac=0.75,random_state=50)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('EXX+RPA')
test_labels = test_dataset.pop('EXX+RPA')

### ## Normalising the data here

normed_train_data = preprocessing.scale(train_dataset)
normed_test_data = preprocessing.scale(test_dataset)

#print normed_test_data

### ## Defining the ANN model here

def build_model():
       model = keras.Sequential([
       layers.Dense(100, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
       layers.Dense(100, activation=tf.nn.sigmoid),
       layers.Dense(1)])

       optimizer = tf.keras.optimizers.RMSprop(0.001)

       model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
       return model
model = build_model()
print model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print(".")

EPOCHS = 1000
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.25, verbose=0)
#  epochs=EPOCHS, validation_split = 0.25, verbose=0,  callbacks=[PrintDot()])    
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print hist.tail()

#################################################################################################
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [E$_{ads}$]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,2.5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [${E_{ads}}^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
          label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,2.5])
  plt.legend()
  plt.show()

plot_history(history)
#######################################################################################################   
model = build_model()

######## The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.25, verbose=0, callbacks=[early_stop])
##                    validation_split = 0.25, verbose=0, callbacks=[early_stop, PrintDot()])
              
plot_history(history)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
rmse_test=np.sqrt(mse)
#    print ('%s' %n, '{}'.format(mse))
loss, mae, mse = model.evaluate(normed_train_data, train_labels, verbose=0)
rmse_train=np.sqrt(mse)

#########################################################################################################################
with open('ann-out.csv', mode='a') as out_file:
        out_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)      
        out_writer.writerow(['%s', '{}'.format(rmse_train), '{}'.format(rmse_test)])
        out_file.close()
        data_stat=pd.read_csv('ann-out.csv',  names = ["RMSE-Train", "RMSE-test"])  
from shutil import copyfile    
copyfile('ann-out.csv', 'ANN-cross-validation-out.csv')
data_stat.describe().to_csv("stat-results-ANN.csv")
try:
    os.remove("ann-out.csv")
except OSError:
    pass
########################################################################################################################

train_dataset = dataset.sample(frac=0.75,random_state=500)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('EXX+RPA')
test_labels = test_dataset.pop('EXX+RPA')

#print train_labels

normed_train_data = preprocessing.scale(train_dataset)
normed_test_data = preprocessing.scale(test_dataset)   
test_predictions = model.predict(normed_test_data).flatten()
train_predictions = model.predict(normed_train_data).flatten()
############################################################################################################

##############################################################################################################
plt.figure(figsize=(4,4),dpi=300)
plt.title('ANN-Tensorflow', fontsize=13)
plt.xlim((-3,5))
plt.xticks(np.linspace(-3,5,5,endpoint=True))
plt.ylim((-3,5))
plt.yticks(np.linspace(-3,5,5,endpoint=True))
plt.tick_params(axis='both', which='major', labelsize=13)
plt.scatter(train_labels, train_predictions, label='Training',facecolors='green',alpha=0.8, edgecolors='none',s=60,marker='^')
plt.scatter(test_labels, test_predictions, label='Testing',facecolors='blue',alpha=0.75, edgecolors='none',s=60, marker='o')
plt.xlabel("Calculated",fontsize=13)
plt.ylabel("ML predicted",fontsize=13)

#####fitting with a y=x line here (polynomial approach)
plt.plot((-3,5),(-3,5), 'r-',linestyle='dashed',label='y=x line')
plt.tight_layout()
plt.legend(loc=2, prop={'size': 13}, frameon=False)
#plt.text(1, -1, 'RMSE=0.19 eV \n R$^2$=0.96',fontsize=13)
plt.savefig ('Fig.eps')
plt.show()



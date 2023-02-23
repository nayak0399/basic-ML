#import_libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from numpy import arange,array,ones
from scipy import stats
from scipy.interpolate import *



#import_descriptors_files_here_in_cvs_format
dataset = pd.read_csv('functional.csv')
#dataset=dataset.drop('Enthalpy-of-Fusion(kJmol-1)',axis=1)
dataset.head()
#dataset.describe()
print (dataset.head())


###finding_correlation between attributes_using_pearson
dataset.corr(method='pearson')
correlation_matrix = dataset.corr().round(3)





##########ploting correlation matrix in temperature map
#yticklabels =('tick_min', 'tick_max', 'tick_step')
#xticklabels =('tick_min', 'tick_max', 'tick_step')
plt.figure(figsize=(8,8),dpi=200)
#xticks = ['G', 'R', 'AN', 'AM','P', 'EN', 'IE', '$\Delta_{fus}$H' ,'$\\rho$']
#cbar_ax = figure.add_axes([.905, .3, .05, .3])
#sns.heatmap(data=correlation_matrix, vmin=0.95, vmax=1,square=True, annot_kws={'size': 15},annot=True, fmt="g", cmap='YlGnBu')
#sns.set(font_scale=1.5)
#sns.heatmap(data=correlation_matrix, vmax=1,center=0,square=True, annot=False, fmt="g", cmap='PiYG',yticklabels=xticks,xticklabels=xticks,cbar_kws={"shrink": .725},annot_kws={'size': 20})

sns.heatmap(data=correlation_matrix, square=True, fmt='g',vmin=0.95, cmap='viridis', vmax=1, annot=True, cbar_kws={"shrink": .81},annot_kws={'size': 15},cbar=False)
#plt.xlabels = ['Frogs', 'Hogs', 'Bogs', 'Slogs','Frogs', 'Hogs', 'Bogs', 'Slogs','X']
#plt.ylabels = ['Frogs', 'Hogs', 'Bogs', 'Slogs','Frogs', 'Hogs', 'Bogs', 'Slogs','X']
#plt.xticks()
#plt.yticks()
#plt.figure(figsize=(10,10))
plt.savefig('correlation.eps', bbox_inches="tight")
plt.tight_layout()
plt.show()




##finding_correlations_with_target
pearson = dataset.corr(method='pearson')
#corr_with_target = pearson.ix[-1][:-1]  #[target_column][:lastcolumn]



#corr_with_target_dict = corr_with_target.to_dict()

#print("FEATURE \tCORRELATION")
#for attr in sorted(corr_with_target_dict.items(), key = lambda x: -abs(x[1])):
#    print("{0}: \t{1}".format(*attr))

#print ('----------------------------------------------------')
#print ('Correlations between Descriptors and Target')
    
#print corr_with_target[abs(corr_with_target).argsort()[::1]]





##########finding-correlation_between_attributes_pair

attrs = pearson.iloc[:-1,:-1] # all except target

#########only important correlations and not auto-correlations
#threshold
threshold = 0.5
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
    .unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
   for key in important_corrs])), columns=['Attribute Pair', 'Correlation'])
#sorted by absolute value

unique_important_corrs = unique_important_corrs.ix[
abs(unique_important_corrs['Correlation']).argsort()[::-1]]

print unique_important_corrs

################################################
###plot_data between descriptor with headers

#features = ['moment4d']
#for i, col in enumerate(features):
#   plt.subplot(1, len(features) , i+1)
#    x = dataset[col]
#    y = target
#    plt.scatter(x, y, marker='o')
#    plt.title(col)
#    plt.xlabel(col)
#plt.ylabel('moment')
#plt.show()






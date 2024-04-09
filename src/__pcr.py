import os
import sys
sys.path.append(os.getcwd())
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, r'.\Data-Analysis-With-Pima-Diabetes-Dataset')
import numpy as np
import pandas as pd
import matplotlib.pylab as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from sklearn.decomposition import PCA

sns.set_palette(palette='deep')
sns.set_style('whitegrid')


class PCR:
    def __init__(self):
        self.df = pd.read_csv(r'.\datasets\cleaned_dataset.csv')
        self.X = self.df.iloc[:,:-1]
        self.y = self.df.iloc[:,-1:]
        print(self.X)
        print(self.y)
    
    def __train_test_split(self):
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.25,random_state=21)

if __name__ == '__main__':
    pcr = PCR()
    pcr.run()
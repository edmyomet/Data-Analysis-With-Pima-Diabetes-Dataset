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

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


class PolyReg:
    def __init__(self):
        self.df = pd.read_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\datasets\cleaned_dataset.csv')
        self.feature_list = self.df.columns
        self.X = self.df.iloc[:,:-1]
        self.y = self.df.iloc[:,-1:]
        
    def __test_train_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.33,random_state=21)
        
    def __polyfit(self):
        self.poly_feat = PolynomialFeatures(degree=1)
        self.X_train_poly = self.poly_feat.fit_transform(self.X_train)
        self.X_test_poly = self.poly_feat.fit_transform(self.X_test)
    
    def __train(self):
        self.reg = LinearRegression()
        self.reg.fit(self.X_train_poly,self.y_train)
    
    def __test(self):
        self.y_pred = self.reg.predict(self.X_test_poly)
    
    def __score(self):
        self.mse = mean_squared_error(self.y_pred,self.y_test)
        self.mae = mean_absolute_error(self.y_test,self.y_pred)
        self.r2 = r2_score(self.y_test,self.y_pred)
        self.sm = np.array([[self.mse,self.mae,self.r2]])
        print(self.sm)
    def model(self):
        self.__test_train_split()
        self.__polyfit()
        self.__train()
        self.__test()
        self.__score()        
    

if __name__ == '__main__':
    pr = PolyReg()
    pr.model()
import numpy as np
import pandas as pd
import matplotlib.pylab as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class Diabetes:
    def __init__(self):
        self.df = pd.read_csv(r'C:\Users\shrut\OneDrive\Pictures\wallpapers\Documents\Projects\Data-Analysis-With-Pima-Diabetes-Dataset\datasets\diabetes.csv')
        self.feature_list = self.df.columns
        
    def __dataset_description(self):
        self.df.describe().to_csv(r'C:\Users\shrut\OneDrive\Pictures\wallpapers\Documents\Projects\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\dataset_description.csv')

    def __freq(self):
        for feature in self.feature_list:
            pd.DataFrame(self.df[feature].value_counts()).to_csv(r'C:\Users\shrut\OneDrive\Pictures\wallpapers\Documents\Projects\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\freq.csv',mode='a')
            
    def __mean(self):
        self.mean = self.df.mean(numeric_only=True,skipna=True)
        self.mean.to_csv(r'C:\Users\shrut\OneDrive\Pictures\wallpapers\Documents\Projects\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\mean.csv')
    def __median(self):
        self.median = self.df.median(numeric_only=True,skipna=True)
        self.median.to_csv(r'C:\Users\shrut\OneDrive\Pictures\wallpapers\Documents\Projects\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\median.csv')
        
    def analysis(self):
        self.__dataset_description()
        self.__mean()
        self.__median()
        #self.__freq()
        
        


if __name__ == '__main__':
    db = Diabetes()
    db.analysis()
    
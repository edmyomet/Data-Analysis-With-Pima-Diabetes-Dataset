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

    def analysis(self):
        self.__dataset_description()
        


if __name__ == '__main__':
    db = Diabetes()
    db.analysis()
    
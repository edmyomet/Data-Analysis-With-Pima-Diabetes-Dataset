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
        
    def __mode(self):
        self.mode = self.df.mode(numeric_only=True)
        self.mode.to_csv(r'C:\Users\shrut\OneDrive\Pictures\wallpapers\Documents\Projects\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\mode.csv')
        
    def __mean_median_deviation(self):
        temp = pd.DataFrame( (np.array(self.mean) - np.array(self.median)) / 100)
        for value in temp[:]:
            if value > 0.05 or value < -0.05:
                self.mm_dev = True
            else:
                self.mm_dev = False
                break
    
    def __fill_nulls(self):
        self.__mean_median_deviation()
        if self.mm_dev == True:
            self.df.fillna(self.mean,inplace=True)
        else:
            self.df.fillna(self.median, inplace=True)
    
    def __boxplot(self):
        self.hue = np.random.randint(100,size=8)
        fig,axes= plt.subplots(1,8,figsize=(25,4),squeeze=False)
        fig.suptitle('Box Plot For Outlier Detection')
        for i in range(8):
            sns.cubehelix_palette(as_cmap=True)
            sns.boxplot(self.df[self.feature_list[i]],width=0.2,ax=axes[0][i])
            axes[0][i].set_title(f'Box Plot for {self.feature_list[i]}')
        plt.savefig(r'C:\Users\shrut\OneDrive\Pictures\wallpapers\Documents\Projects\Data-Analysis-With-Pima-Diabetes-Dataset\output\img\boxplot.png')
    
    
    
    def __normal_distr(self):
        fig,axes = plt.subplots(1,8,figsize=(25,4),squeeze=False)
        fig.suptitle('Normal Distribution for the Diabetes Dataset')
        for i in range(8):
            sns.histplot(data=self.df,x=self.df[self.feature_list[i]],kde=True,ax=axes[0][i])
            axes[0][i].set_title(f'{self.feature_list[i]}')
        plt.savefig(r'C:\Users\shrut\OneDrive\Pictures\wallpapers\Documents\Projects\Data-Analysis-With-Pima-Diabetes-Dataset\output\img\histplot.png')
        
    def __skew(self):
        pass
    
    def __kurt(self):
        pass
            
        
    def analysis(self):
        self.__dataset_description()
        self.__mean()
        self.__median()
        #self.__freq()
        self.__mode()
        self.__fill_nulls()
        self.__boxplot()

        

if __name__ == '__main__':
    db = Diabetes()
    db.analysis()
    
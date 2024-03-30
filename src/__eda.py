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

sns.set_palette(palette='deep')
sns.set_style('whitegrid')

class Diabetes:
    def __init__(self):
        self.df = pd.read_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\datasets\diabetes.csv')
        self.feature_list = self.df.columns
        
    def __dataset_description(self):
        self.df.describe().to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\dataset_description.csv')

    def __freq(self):
        for feature in self.feature_list:
            pd.DataFrame(self.df[feature].value_counts()).to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\freq.csv',mode='a')
            
    def __mean(self):
        self.mean = self.df.mean(numeric_only=True,skipna=True)
        self.mean.to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\mean.csv')
    def __median(self):
        self.median = self.df.median(numeric_only=True,skipna=True)
        self.median.to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\median.csv')
        
    def __mode(self):
        self.mode = self.df.mode(numeric_only=True)
        self.mode.to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\mode.csv')
        
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
        fig,axes= plt.subplots(1,8,figsize=(25,4),squeeze=False)
        fig.suptitle('Box Plot For Outlier Detection')
        for i in range(8):
            sns.cubehelix_palette(as_cmap=True)
            sns.boxplot(self.df[self.feature_list[i]],width=0.2,ax=axes[0][i])
            axes[0][i].set_title(f'Box Plot for {self.feature_list[i]}')
        plt.savefig(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\img\boxplot.png')
    
    def __std(self):
        self.std = self.df.std(numeric_only=True)
        self.std.to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\std.csv')
        
    
    def __normal_distr(self):
        fig,axes = plt.subplots(1,8,figsize=(25,4),squeeze=False)
        fig.suptitle('Normal Distribution for the Diabetes Dataset')
        for i in range(8):
            sns.histplot(data=self.df,x=self.df[self.feature_list[i]],kde=True,ax=axes[0][i])
            axes[0][i].set_title(f'{self.feature_list[i]}')
        plt.savefig(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\img\histplot.png')
        
    def __skewness(self):
        self.skew = self.df.skew(numeric_only=True)
        self.skew.to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\skew.csv')
    
    def __kurtosis(self):    
        self.kurt = self.df.kurtosis(numeric_only=True)
        self.kurt.to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\kurtosis.csv')
    
    def __outlier_remove(self):
        for feature in self.feature_list:
            q1 = self.df[feature].quantile(0.25)
            q3 = self.df[feature].quantile(0.75)
            
            iqr = q3 - q1
            
            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr
            
            self.df_cleaned = self.df[(self.df[feature] < upper ) & (self.df[feature] > lower)]
            self.df = self.df_cleaned
                       
    def __cov(self):
        self.cov = self.df.cov(numeric_only=True)
        self.cov.to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\cov.csv')
    
    def __corr(self):
        self.corr = self.df.corr(numeric_only=True)
        self.corr.to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\results\corr.csv')
    
    def __scatter(self):
        fig,axes= plt.subplots(8,8,figsize=(30,50),squeeze=False,layout='constrained')
        for i in range(8):
            for j in range(8):
                x = self.df[self.feature_list[i]]
                y = self.df[self.feature_list[j]]
                label = self.df.Outcome
                m,b= np.polyfit(x,y,1)
                sns.scatterplot(x=x,y=y,hue=label,palette='crest',ax=axes[i][j])
                axes[i][j].plot(x,m*x+b,linewidth=2,color='black')
        plt.savefig(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\img\scatterplot.png')
        
    
    def __heatmap(self):
        fig,axes=plt.subplots(1,1,figsize=(10,8),squeeze=False)
        sns.heatmap(self.corr,center=0, annot=True,fmt='.2f',cmap=sns.cubehelix_palette(as_cmap=True))
        plt.savefig(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\output\img\heatmap.png')    
    
    def __drop_features(self):
        self.df.drop(columns=['SkinThickness'],inplace=True)
    
    def __MinMax_Scaler(self):
        self.scaler = MinMaxScaler()
        self.df_scaled = self.scaler.fit_transform(self.df)
        pd.DataFrame(self.df_scaled,columns=self.df.columns).to_csv(r'.\Data-Analysis-With-Pima-Diabetes-Dataset\datasets\cleaned_dataset.csv')       
    def analysis(self):
        self.__dataset_description()
        self.__mean()
        self.__median()
        #self.__freq()
        self.__mode()
        self.__fill_nulls()
        self.__boxplot()
        self.__skewness()
        self.__kurtosis()
        self.__normal_distr()
        self.__outlier_remove()
        self.__cov()
        self.__corr()
        self.__scatter()
        self.__heatmap()
        self.__drop_features()
        self.__MinMax_Scaler()

if __name__ == '__main__':
    db = Diabetes()
    db.analysis()
    
'''
Mini-Project 2 : Car Price Prediction using Multiple Linear Regression
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
import sklearn
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger=setup_logging('main')
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from training_log_code import training_data
from testing_log_code import testing_data
from saving_model import saving
class MINI_PROJECT_2:
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(self.path)
            logger.info(f'The shape of the data was:{self.df.shape}')
            self.df = self.df.drop(['Car ID'], axis=1)
            logger.info(f'After removing the carid column then the shape of the data was:{self.df.shape}')
            self.X = self.df.drop(['Price'], axis=1) #indepedent columns
            self.y = self.df['Price'] # dependent
            logger.info(f'The list of column names are:{self.X.columns}')
            logger.info(f'The datatype of columns is:{self.df.dtypes}')   # converting categorical columns into numerical
            categorical_columns = self.X.select_dtypes(include='object').columns
            logger.info(f'The categorical columns in X are:{categorical_columns}')
            for col in categorical_columns:
                unique_values = self.X[col].unique()

                for i in range(len(unique_values)):
                    self.X[col] = self.X[col].replace(unique_values[i], i)

            logger.info(f' After Converted categorical columns into numerical columns the updated columns  is:\n{self.X.head(10)}')
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,random_state=42)
            logger.info(f"Train Dataset size : {self.X_train.shape} => {self.y_train.shape}")
            logger.info(f"Test Dataset size : {self.X_test.shape} => {self.y_test.shape}")
        except Exception as e:
            er_ty,er_line,er_msg=sys.exc_info()
            logger.info(f'error in line no:{er_line.tb_lineno} due to error msg :{er_msg} and reason was:{er_ty}')
    def training_data(self):
        try:
            reg=training_data(self.X_train,self.y_train)
            self.reg=reg
        except Exception as e:
             er_line, er_msg ,er_ty= sys.exc_info()
             logger.info(f'error in line no: {er_line.tb_lineno} due to error msg :{er_msg} and reason was:{er_ty}')
    def testing_data(self):
         try:
             testing_data(self.reg, self.X_test, self.y_test)

         except Exception as e:
             er_line, er_msg,er_ty = sys.exc_info()
             logger.info(f'error in line no:{er_line.tb_lineno} due to error msg :{er_msg} and reason was:{er_ty}')
    def saving_model(self):
        try:
            saving(self.reg,self.X_train)
        except Exception as e:
            er_line, er_msg, er_ty = sys.exc_info()
            logger.info(f'error in line no:{er_line.tb_lineno} due to error msg :{er_msg} and reason was:{er_ty}')

if __name__ == "__main__":
    obj=MINI_PROJECT_2("car_price_prediction_.csv")
    obj.training_data()
    obj.testing_data()
    obj.saving_model()

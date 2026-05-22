import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import logging
import sklearn
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
from sklearn.linear_model import LinearRegression
logger=setup_logging('training_log_code')
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
def training_data(X_train,y_train):
    try:
        global reg
        logger.info(f'Training data Performance')
        reg = LinearRegression()
        reg.fit(X_train,y_train)
        y_train_predictions =reg.predict(X_train)
        logger.info(f'Train Accuracy:{r2_score(y_train,y_train_predictions)}')
        logger.info(f'Train loss:{root_mean_squared_error(y_train, y_train_predictions)}')
        return reg
    except Exception as e:
        er_line, er_msg, er_ty, = sys.exc_info()
        logger.info(f'error in line no:{er_line.tb_lineno} due to error msg :{er_msg} and reason was:{er_ty}')

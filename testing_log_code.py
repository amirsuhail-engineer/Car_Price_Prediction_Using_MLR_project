
import sys
import logging
from log_code import setup_logging
logger=setup_logging('testing_log_code')
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
def testing_data(reg,X_test,y_test):
    try:
        logger.info(f'Testing performance')
        y_test_predictions = reg.predict(X_test)
        logger.info(f'Test Accuracy:{r2_score(y_test, y_test_predictions)}')
        logger.info(f'Test loss:{root_mean_squared_error(y_test, y_test_predictions)}')

    except Exception as e:
        er_line, er_msg,er_ty,= sys.exc_info()
        logger.info(f'error in line no:{er_line.tb_lineno} due to error msg :{er_msg} and reason was:{er_ty}')

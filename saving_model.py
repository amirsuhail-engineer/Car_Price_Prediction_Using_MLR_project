import pickle
from log_code import setup_logging
logger=setup_logging('saving_model')
def saving(reg,X_train):
    try:
       with open('car_price.pkl','wb') as f:
           pickle.dump(reg,f)
       logger.info(f'saving the model')
       with open('car_price.pkl','rb') as t:
           m=pickle.load(t)
       logger.info(f'Now Testing the own data prediction:{m.predict(X_train.tail(1))}')
    except Exception as e:
        er_line, er_msg, er_ty = sys.exc_info()
        logger.info(f'error in line no:{er_line.tb_lineno} due to error msg :{er_msg} and reason was:{er_ty}')

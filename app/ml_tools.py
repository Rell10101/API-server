from dotenv import load_dotenv
import os
import pandas as pd
import pickle
from sklearn.metrics import r2_score
import json

load_dotenv()


model_filename = os.getenv("MODEL_FILE_NAME")
with open(model_filename, "rb") as f:
        lin_reg = pickle.load(f)


train_data = pd.read_csv('../models/train_data.csv')


def predict_Y(x1,x2,x3,x4,x5):
    y_pred = lin_reg.predict([[x1,x2,x3,x4,x5]])[0]
    return y_pred


y_pred = lin_reg.predict(train_data[['x1','x2','x3','x4', 'x5']])

# TODO: нужно считать его тут, или взять из другой лабы?
# R2 = r2_score(train_data['y'], y_pred)



with open('../models/model_info.json', 'r', encoding='utf-8') as f:
    model_info = json.load(f)

R2 = model_info['r2_score']




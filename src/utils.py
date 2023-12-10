import os
import sys
import numpy as np 
import pandas as pd
import pickle
import dill
from sklearn.metrics import r2_score, mean_absolute_error, get_scorer
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from tensorflow.keras.models import Model

def save_object(file_path, obj):
    try:
        if isinstance(obj, Model):
            # For Keras models
            obj.save(file_path)
        else:
            # For other objects
            with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        # You create a GridSearchCV object with your model here in models her, the parameter grid (param_grid), and the number of folds for cross-validation (cv=3).
        # You fit the GridSearchCV object to your training data (X_train, y_train).
        # You extract the best parameters found by GridSearchCV.
        # You update your model with the best parameters using set_params.
        # You fit the updated model to your training data.
        # Finally, you can use the updated model for making predictions.

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para, cv=3)

            gs.fit(X_train, y_train) # Train model

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)

            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)


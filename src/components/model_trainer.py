import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass
# Modelling

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

from src.utils import save_object,evaluate_models
from src.logger import logging
from src.exception import CustomException
import numpy as np


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            # split the features and target
            logging.info("split training and test arry")
            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            
            models={
                'Random Forest':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'XGBoost':XGBRegressor(),
                'CatBoost':CatBoostRegressor(),
                'AdaBoost':AdaBoostRegressor(),
                'Linear Regression':LinearRegression(),
                'Ridge':Ridge(),
                'Lasso':Lasso(),
                'SVR':SVR(),
                'KNN':KNeighborsRegressor() 
            }
            
            model_report:dict = evaluate_models(x_train, y_train, x_test, y_test, models)
            
            best_model_score=max(sorted(model_report.values()))
            
            
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("no model found")
            
            logging.info(f"best model is {best_model_name}")
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test, predicted)
            logging.info(f"r2 score is {r2}")
            logging.info(f"model training completed")
            return r2
        except Exception as e:
            raise CustomException(e, sys)

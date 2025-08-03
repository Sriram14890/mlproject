import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Train and Test Input.")
            X_train, y_train, X_test, y_test=(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models={
                "Linear Regression" : LinearRegression(),
                "Support Vector Regressor" : SVR(),
                "K Nearest Neighbors Regressor" : KNeighborsRegressor(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "CatBoost Regressor" : CatBoostRegressor(verbose=False),
                "XGB Regressor" : XGBRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                             models=models)
            best_model_score=max(model_report.values())
            best_model_name=max(model_report, key=model_report.get)
            best_model=models[best_model_name]

            """if best_model_score<0.6:
                raise CustomException("No Best Model Found.")"""
            logging.info("Best Model found on training and testing datasets.")

            save_object(
                file_path=self.model_trainer.trained_model_file_path, obj=best_model
                )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
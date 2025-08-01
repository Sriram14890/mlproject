import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This Function is responsible for Data Transformation 
        '''    
        try:
            num_cols=["writing score", "reading score"]
            cat_cols=[
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]


            num_pipeline=Pipeline(steps=[
                ("Imputer", SimpleImputer(strategy="median")),
                ("Scaler", StandardScaler())
            ])

            cat_pipeline=Pipeline([
                ("Imputer", SimpleImputer(strategy="most_frequent")),
                ("One Hot Encoder", OneHotEncoder()),
                
            
            ])
            logging.info(f"Categorical Columns: {cat_cols}")

            logging.info(f"Numerical Columns: {num_cols}")

            preprocessor=ColumnTransformer([
                ("Num Pipeline", num_pipeline, num_cols),
                ("Cat Pipeline", cat_pipeline, cat_cols)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initialise_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read Train and Test data completed.")

            logging.info("Obtaining Preprocessor Info.")

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name="math score"

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test dataframes.")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved Preprocessing Object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
            
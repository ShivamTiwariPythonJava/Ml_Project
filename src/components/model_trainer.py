import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pk1")
    logging.info(" final result pkl file will be stored in artifact")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        #Model trainer class will generate pkl file at last which need to be stored in artifacts so this instance is needed so pkl file can be stored at path
        # mnetioned in ModelTrainerConfig path attribute it is like connection 
        logging.info("DataTransformation instances can access the file path defined in DataTransformationConfig")
    

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            X_train, y_train, X_test, y_test = (
                # train_array and test_array we got from data_transformation file as return 
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # we will train with multiple moldels to choose best one, this is way to use multiple models at once
            models = {

                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor":XGBRegressor(),
                # in catboost verbose false means during training logs will not be genrated means huge data like current iteration, training loss, evaluation metrics 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),}
            
            
            model_report:dict = evaluate_models(X_train = X_train, y_train=y_train,X_test=X_test,y_test = y_test, models=models)


            ## To get best model name from dict

            best_model_score = max(sorted(model_report.values()))
            print("Best_Model_Score: ", best_model_score)

            ## To get best model name from dict 

            best_model_name = list(model_report.keys())[

                list(model_report.values()).index(best_model_score)

            ]
            print("Best_Model_Name:",best_model_name)
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("best model found on both training and testing datset")

            # Using save_object to create pkl file check utils.py for detail
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square





            

        except Exception as e:
            raise CustomException(e,sys)
    



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
            
            param={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    # this is bagging means we will use combination of multiple models here let's say uf we are using descision tree in random forest
                    # n_estimators show us with how many treewe try each 8,16 like that we are increasing number of tree 
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            
            model_report:dict = evaluate_models(X_train = X_train, y_train=y_train,X_test=X_test,y_test = y_test, models=models, param=param)


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
    



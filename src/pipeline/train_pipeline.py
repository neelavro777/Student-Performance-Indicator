import sys
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        data_ingestion_obj = DataIngestion()
        self.train_data, self.test_data = data_ingestion_obj.initiate_data_ingestion()
        
        data_transformation_obj = DataTransformation()
        self.train_data_arr, self.test_data_arr,_ = data_transformation_obj.initiate_data_transformation(self.train_data, self.test_data)
        
        self.X_test = self.test_data_arr[:,:-1]
        self.y_test = self.test_data_arr[:,-1]

    def train(self):

        try:
            model_trainer = ModelTrainer()
            self.best_model = model_trainer.initiate_model_trainer(self.train_data_arr, self.test_data_arr)

        except Exception as e:
            raise CustomException(e, sys)

    def test(self):

        try:
            predictions = self.best_model.predict(self.X_test)
            r2_square = r2_score(self.y_test, predictions)
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)     

if __name__=="__main__":
    training_obj = TrainPipeline()

    logging.info(
            f"Initiating Training"
            )

    training_obj.train()

    logging.info(
            f"Initiating Testing"
        )

    r2_square = training_obj.test()

    logging.info(
            f"R2 Square value of Model on test set is {r2_square}"
        )        

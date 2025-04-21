import os 
import pandas as pd
import numpy as np
from src.logger import get_logger
from config.paths_config import *
from src.custom_exception import CustomException
from utils.common_functions import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self,train_path,test_path,config_path,processed_dir):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
    def preprocess_data(self,df):
        try:
            logger.info("starting data processing step")

            logger.info("Dropping the columns")

            df.drop(columns=['Unnamed: 0','Booking_ID'],inplace =True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("apply label encoding")
            label_encoder = LabelEncoder()
            mapping={}
            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mapping[col] = {label:code for label,code in zip(label_encoder.classes_ , label_encoder.transform(label_encoder.classes_))}
            logger.info("label mappings:")
            for columns,mapping in mapping.items():
                logger.info(f"{col}:{mapping}")
            
            logger.info("skewness handling")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x:x.skew())
            for column in skewness [skewness>skew_threshold].index:
                df[column]= np.log1p(df[column])
            return df
        
        except Exception as e: 
            logger.error(f"error during preprocessing{e}")
            raise CustomException("error while preprocess data",e)
        
    def balanced_data(self,df):
        try:
            logger.info("hanlding imbalanced data")
            X = df.drop('booking_status', axis=1)
            y = df['booking_status']
            smote =SMOTE(random_state=42)
            X_resampled , y_resampled = smote.fit_resample(X,y)
            balanced_df = pd.DataFrame(X_resampled , columns=X.columns)
            balanced_df["booking_status"] = y_resampled

            logger.info("data balanced successfully")
            return balanced_df
        except Exception as e: 
            logger.error(f"error during balancing data{e}")
            raise CustomException("error while balancing  data",e)
        
    def select_features(self,df):
        try:
            logger.info("starting feature selction process")
            X = df.drop(columns='booking_status')
            y = df["booking_status"]
            model =  RandomForestClassifier(random_state=42)
            model.fit(X,y)
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature':X.columns,
                                                  'importance':feature_importance
                                                  })
            top_features_importance_df = feature_importance_df.sort_values(by="importance" , ascending=False)

            num_feature_selection = self.config["data_processing"]["no_of_features"]

            top_10_features = top_features_importance_df["feature"].head(num_feature_selection).values

            logger.info(f"feature selected{top_10_features}")

            top_10_df = df[top_10_features.tolist() + ["booking_status"]]

            logger.info("feature selection completed success ")
            return top_10_df

        except Exception as e: 
            logger.error(f"error feature selection{e}")
            raise CustomException("error while feature selection",e)
        
    def save_data(self,df,file_path):
        try:
            logger.info("saving data into processed folder")
            df.to_csv(file_path,index=False)
            logger.info("data saved suceesgully{file_path}")

        except Exception as e: 
            logger.error(f"error saving data{e}")
            raise CustomException("error while saving data",e)
        
    def process(self):
        try:
            logger.info("loading data from RAW directory")
            self.train_df = load_data(self.train_path)
            self.test_df = load_data(self.test_path)

            logger.info("Preprocessing data")
            self.train_df = self.preprocess_data(self.train_df)
            self.test_df = self.preprocess_data(self.test_df)

            logger.info("Balancing data")
            self.train_df = self.balanced_data(self.train_df)
            self.test_df = self.balanced_data(self.test_df)

            logger.info("Selecting features")
            self.train_df = self.select_features(self.train_df)
            self.test_df = self.test_df[self.train_df.columns]

            logger.info("Saving processed data")
            self.save_data(self.train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(self.test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing successfully")
        except Exception as e:
            logger.error(f"error during preprocessing pipeline: {e}")
            raise CustomException("error during preprocessing pipeline", e)
        
if __name__=="__main__":
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,CONFIG_PATH,PROCESSED_DIR,)
    processor.process()

        

        


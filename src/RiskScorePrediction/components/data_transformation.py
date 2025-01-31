import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from RiskScorePrediction import logger
from RiskScorePrediction.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

        self.numerical_columns = [
            'Age', 'AnnualIncome', 'CreditScore', 'Experience', 'LoanAmount',
            'LoanDuration', 'NumberOfDependents', 'MonthlyDebtPayments',
            'CreditCardUtilizationRate', 'NumberOfOpenCreditLines',
            'NumberOfCreditInquiries', 'DebtToIncomeRatio', 'BankruptcyHistory',
            'PreviousLoanDefaults', 'PaymentHistory', 'LengthOfCreditHistory',
            'SavingsAccountBalance', 'CheckingAccountBalance', 'TotalAssets',
            'TotalLiabilities', 'MonthlyIncome', 'UtilityBillsPaymentHistory',
            'JobTenure', 'NetWorth', 'BaseInterestRate', 'InterestRate',
            'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'LoanApproved'
        ]

        self.categorical_columns = [
            'EmploymentStatus', 'EducationLevel',
            'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose'
        ]

    def get_data_transformation(self):
        try:
            
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

           
            logger.info(f"Categorical columns: {self.categorical_columns}")
            logger.info(f"Numerical columns: {self.numerical_columns}")

           
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, self.numerical_columns),
                ("cat_pipeline", cat_pipeline, self.categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            logger.error(f"Error in get_data_transformation: {str(e)}")
            raise e

    def initiate_data_transformation(self):
        try:
           
            train_df = pd.read_csv(self.config.train_data_dir)
            test_df = pd.read_csv(self.config.test_data_dir)

            logger.info("Successfully loaded train and test datasets.")

            
            preprocessing_obj = self.get_data_transformation()

            target_column_name = "RiskScore"

           
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            
            one_hot_feature_names = preprocessing_obj.named_transformers_["cat_pipeline"] \
                .named_steps["one_hot_encoder"].get_feature_names_out(self.categorical_columns)

            
            final_column_names = self.numerical_columns + list(one_hot_feature_names)

            logger.info(f"Transformed training data shape: {input_feature_train_arr.shape}")
            logger.info(f"Transformed test data shape: {input_feature_test_arr.shape}")

            
            train_arr = pd.DataFrame(
                np.c_[input_feature_train_arr, np.array(target_feature_train_df)],
                columns=final_column_names + [target_column_name]
            )
            test_arr = pd.DataFrame(
                np.c_[input_feature_test_arr, np.array(target_feature_test_df)],
                columns=final_column_names + [target_column_name]
            )

            
            train_arr.to_csv(self.config.transformed_train_data_dir, index=False)
            test_arr.to_csv(self.config.transformed_test_data_dir, index=False)

            logger.info("Data transformation completed successfully.")
            logger.info(f"Size of training data: {train_arr.shape}")
            logger.info(f"Size of test data: {test_arr.shape}")

            save_object(
                file_path=self.config.preprocessor_file_dir,
                obj=preprocessing_obj

            )

        except Exception as e:
            logger.error(f"Error in initiate_data_transformation: {str(e)}")
            raise e

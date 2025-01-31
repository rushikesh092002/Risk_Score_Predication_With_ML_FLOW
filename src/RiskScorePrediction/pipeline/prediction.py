import sys
import os
from RiskScorePrediction import logger 
from RiskScorePrediction.utils.common import load_object

import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Updated model and preprocessor paths
            model_path = os.path.join("artifacts", "model_trainer", "model.joblib")
            preprocessor_path = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
            print("before Loading")

            # Load the model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Apply preprocessor transformation
            data_scaled = preprocessor.transform(features)

            # Predict using the loaded model
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise e

class CustomData:
    def __init__(self,
                 Age: int,
                 AnnualIncome: int, 
                 CreditScore: int, 
                 Experience: str,
                 LoanAmount: int,
                 LoanDuration: int, 
                 NumberOfDependents: int, 
                 MonthlyDebtPayments: int,
                 CreditCardUtilizationRate: int, 
                 NumberOfOpenCreditLines: int,
                 NumberOfCreditInquiries: int, 
                 DebtToIncomeRatio: int,
                 BankruptcyHistory: int,
                 PreviousLoanDefaults: int, 
                 PaymentHistory: int,
                 LengthOfCreditHistory: int,
                 SavingsAccountBalance: int, 
                 CheckingAccountBalance: int, 
                 TotalAssets: int,
                 TotalLiabilities: int, 
                 MonthlyIncome: int, 
                 UtilityBillsPaymentHistory: int,
                 JobTenure: int, 
                 NetWorth: int, 
                 BaseInterestRate: int, 
                 InterestRate: int,
                 MonthlyLoanPayment: int, 
                 TotalDebtToIncomeRatio: int, 
                 LoanApproved: int,
                 EmploymentStatus: str, 
                 EducationLevel: str,
                 MaritalStatus: str, 
                 HomeOwnershipStatus: str, 
                 LoanPurpose: str):
        
        # Initialize the input data
        self.Age = Age
        self.AnnualIncome = AnnualIncome
        self.CreditScore = CreditScore
        self.Experience = Experience
        self.LoanAmount = LoanAmount
        self.LoanDuration = LoanDuration
        self.NumberOfDependents = NumberOfDependents
        self.MonthlyDebtPayments = MonthlyDebtPayments
        self.CreditCardUtilizationRate = CreditCardUtilizationRate
        self.NumberOfOpenCreditLines = NumberOfOpenCreditLines
        self.NumberOfCreditInquiries = NumberOfCreditInquiries
        self.DebtToIncomeRatio = DebtToIncomeRatio 
        self.BankruptcyHistory = BankruptcyHistory
        self.PreviousLoanDefaults = PreviousLoanDefaults 
        self.PaymentHistory = PaymentHistory
        self.LengthOfCreditHistory = LengthOfCreditHistory
        self.SavingsAccountBalance = SavingsAccountBalance
        self.CheckingAccountBalance = CheckingAccountBalance 
        self.TotalAssets = TotalAssets
        self.TotalLiabilities = TotalLiabilities 
        self.MonthlyIncome = MonthlyIncome
        self.UtilityBillsPaymentHistory = UtilityBillsPaymentHistory
        self.JobTenure = JobTenure 
        self.NetWorth = NetWorth 
        self.BaseInterestRate = BaseInterestRate
        self.InterestRate = InterestRate
        self.MonthlyLoanPayment = MonthlyLoanPayment
        self.TotalDebtToIncomeRatio = TotalDebtToIncomeRatio 
        self.LoanApproved = LoanApproved
        self.EmploymentStatus = EmploymentStatus 
        self.EducationLevel = EducationLevel
        self.MaritalStatus = MaritalStatus
        self.HomeOwnershipStatus = HomeOwnershipStatus
        self.LoanPurpose = LoanPurpose
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Age': [self.Age],
                'AnnualIncome': [self.AnnualIncome],
                'CreditScore': [self.CreditScore],
                'Experience': [self.Experience],
                'LoanAmount': [self.LoanAmount],
                'LoanDuration': [self.LoanDuration],
                'NumberOfDependents': [self.NumberOfDependents],
                'MonthlyDebtPayments': [self.MonthlyDebtPayments],                  
                'CreditCardUtilizationRate': [self.CreditCardUtilizationRate],
                'NumberOfOpenCreditLines': [self.NumberOfOpenCreditLines],
                'NumberOfCreditInquiries': [self.NumberOfCreditInquiries],
                'DebtToIncomeRatio': [self.DebtToIncomeRatio],
                'BankruptcyHistory': [self.BankruptcyHistory],
                'PreviousLoanDefaults': [self.PreviousLoanDefaults],
                'PaymentHistory': [self.PaymentHistory],
                'LengthOfCreditHistory': [self.LengthOfCreditHistory],
                'SavingsAccountBalance': [self.SavingsAccountBalance],
                'CheckingAccountBalance': [self.CheckingAccountBalance],
                'TotalAssets': [self.TotalAssets],
                'TotalLiabilities': [self.TotalLiabilities],
                'MonthlyIncome': [self.MonthlyIncome],
                'UtilityBillsPaymentHistory': [self.UtilityBillsPaymentHistory],
                'JobTenure': [self.JobTenure],
                'NetWorth': [self.NetWorth],
                'BaseInterestRate': [self.BaseInterestRate],
                'InterestRate': [self.InterestRate],
                'MonthlyLoanPayment': [self.MonthlyLoanPayment],
                'TotalDebtToIncomeRatio': [self.TotalDebtToIncomeRatio],
                'LoanApproved': [self.LoanApproved],
                'EmploymentStatus': [self.EmploymentStatus],
                'EducationLevel': [self.EducationLevel],
                'MaritalStatus': [self.MaritalStatus],
                'HomeOwnershipStatus': [self.HomeOwnershipStatus],
                'LoanPurpose': [self.LoanPurpose]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise e

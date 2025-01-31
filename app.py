from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from RiskScorePrediction.pipeline.prediction import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(


            Age=float(request.form.get('Age')), 
            AnnualIncome=float(request.form.get('AnnualIncome')), 
            CreditScore=float(request.form.get('CreditScore')), 
            Experience=float(request.form.get('Experience')),
            LoanAmount=float(request.form.get('LoanAmount')),
            LoanDuration=float(request.form.get('LoanDuration')),
            NumberOfDependents=float(request.form.get('NumberOfDependents')),
            MonthlyDebtPayments=float(request.form.get('MonthlyDebtPayments')),
            CreditCardUtilizationRate=float(request.form.get('CreditCardUtilizationRate')),
            NumberOfOpenCreditLines=float(request.form.get('NumberOfOpenCreditLines')),
            NumberOfCreditInquiries=float(request.form.get('NumberOfCreditInquiries')),
            DebtToIncomeRatio=float(request.form.get('DebtToIncomeRatio')),
            BankruptcyHistory=float(request.form.get('BankruptcyHistory')),
            PreviousLoanDefaults=float(request.form.get('PreviousLoanDefaults')),
            PaymentHistory=float(request.form.get('PaymentHistory')),
            LengthOfCreditHistory=float(request.form.get('LengthOfCreditHistory')),
            SavingsAccountBalance=float(request.form.get('SavingsAccountBalance')),
            CheckingAccountBalance=float(request.form.get('CheckingAccountBalance')),
            TotalAssets=float(request.form.get('TotalAssets')),
            TotalLiabilities=float(request.form.get('TotalLiabilities')),
            MonthlyIncome=float(request.form.get('MonthlyIncome')),
            UtilityBillsPaymentHistory=float(request.form.get('UtilityBillsPaymentHistory')),
            JobTenure=float(request.form.get('JobTenure')),
            NetWorth=float(request.form.get('NetWorth')),
            BaseInterestRate=float(request.form.get('BaseInterestRate')),
            InterestRate=float(request.form.get('InterestRate')),
            MonthlyLoanPayment=float(request.form.get('MonthlyLoanPayment')),
            TotalDebtToIncomeRatio=float(request.form.get('TotalDebtToIncomeRatio')),
            LoanApproved=float(request.form.get('LoanApproved')),
            EmploymentStatus=request.form.get('EmploymentStatus'),
            EducationLevel=request.form.get('EducationLevel'),
            MaritalStatus=request.form.get('MaritalStatus'),
            HomeOwnershipStatus=request.form.get('HomeOwnershipStatus'),
            LoanPurpose=request.form.get('LoanPurpose'),



        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
    
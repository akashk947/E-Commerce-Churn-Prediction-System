import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import gradio as gr

# Define the ChurnPredictor class
class ChurnPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.medians = {}
        self.model = XGBClassifier(gamma=0, learning_rate=0.1, max_depth=7, n_estimators=200, random_state=42)
        self._initialize_encoders_and_medians()
        self._train_dummy_model()

    def _initialize_encoders_and_medians(self):
        # Hardcoded unique values for categorical features based on the notebook snippets
        categorical_features = {
            'PreferredLoginDevice': ['Mobile Phone', 'Phone', 'Computer'],
            'PreferredPaymentMode': ['Debit Card', 'UPI', 'CC', 'Cash on Delivery', 'E wallet', 'COD', 'Credit Card'],
            'Gender': ['Female', 'Male'],
            'PreferedOrderCat': ['Laptop & Accessory', 'Mobile', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'],
            'MaritalStatus': ['Single', 'Divorced', 'Married']
        }

        for col, unique_values in categorical_features.items():
            le = LabelEncoder()
            le.fit(unique_values)
            self.label_encoders[col] = le

        # Approximate medians for numerical features based on the notebook output
        # In a real scenario, these would be calculated from the training data.
        self.medians = {
            'Tenure': 9.0,
            'WarehouseToHome': 16.0,
            'HourSpendOnApp': 3.0,
            'OrderAmountHikeFromlastYear': 15.0,
            'CouponUsed': 1.0,
            'OrderCount': 3.0,
            'DaySinceLastOrder': 3.0
        }

        self.feature_order = [
            'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
            'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
            'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
            'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
            'DaySinceLastOrder', 'CashbackAmount'
        ]

    def _train_dummy_model(self):
        # Create a dummy DataFrame to train the model, mimicking the notebook's data structure and preprocessing.
        # This is crucial because the model is trained within the notebook, not saved.
        # In a real application, you would load a pre-trained model.
        data = {
            'Churn': np.random.randint(0, 2, 100), # Dummy target
            'Tenure': np.random.rand(100) * 20,
            'PreferredLoginDevice': np.random.choice(self.label_encoders['PreferredLoginDevice'].classes_, 100),
            'CityTier': np.random.randint(1, 4, 100),
            'WarehouseToHome': np.random.rand(100) * 50,
            'PreferredPaymentMode': np.random.choice(self.label_encoders['PreferredPaymentMode'].classes_, 100),
            'Gender': np.random.choice(self.label_encoders['Gender'].classes_, 100),
            'HourSpendOnApp': np.random.rand(100) * 5,
            'NumberOfDeviceRegistered': np.random.randint(1, 6, 100),
            'PreferedOrderCat': np.random.choice(self.label_encoders['PreferedOrderCat'].classes_, 100),
            'SatisfactionScore': np.random.randint(1, 6, 100),
            'MaritalStatus': np.random.choice(self.label_encoders['MaritalStatus'].classes_, 100),
            'NumberOfAddress': np.random.randint(1, 15, 100),
            'Complain': np.random.randint(0, 2, 100),
            'OrderAmountHikeFromlastYear': np.random.rand(100) * 30,
            'CouponUsed': np.random.randint(0, 5, 100),
            'OrderCount': np.random.randint(1, 10, 100),
            'DaySinceLastOrder': np.random.rand(100) * 10,
            'CashbackAmount': np.random.rand(100) * 200
        }
        df_dummy = pd.DataFrame(data)

        # Apply the same preprocessing steps as in the notebook
        for col, median_val in self.medians.items():
            df_dummy[col].fillna(median_val, inplace=True)
        for col, le in self.label_encoders.items():
            df_dummy[col] = le.transform(df_dummy[col])

        X_dummy = df_dummy[self.feature_order]
        y_dummy = df_dummy['Churn']
        self.model.fit(X_dummy, y_dummy)

    def predict_churn(self,
                      Tenure, PreferredLoginDevice, CityTier, WarehouseToHome,
                      PreferredPaymentMode, Gender, HourSpendOnApp, NumberOfDeviceRegistered,
                      PreferedOrderCat, SatisfactionScore, MaritalStatus, NumberOfAddress,
                      Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount,
                      DaySinceLastOrder, CashbackAmount):

        input_data = {
            'Tenure': Tenure,
            'PreferredLoginDevice': PreferredLoginDevice,
            'CityTier': CityTier,
            'WarehouseToHome': WarehouseToHome,
            'PreferredPaymentMode': PreferredPaymentMode,
            'Gender': Gender,
            'HourSpendOnApp': HourSpendOnApp,
            'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
            'PreferedOrderCat': PreferedOrderCat,
            'SatisfactionScore': SatisfactionScore,
            'MaritalStatus': MaritalStatus,
            'NumberOfAddress': NumberOfAddress,
            'Complain': Complain,
            'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
            'CouponUsed': CouponUsed,
            'OrderCount': OrderCount,
            'DaySinceLastOrder': DaySinceLastOrder,
            'CashbackAmount': CashbackAmount
        }

        # Create a DataFrame from input data
        df_input = pd.DataFrame([input_data])

        # Apply preprocessing
        for col, median_val in self.medians.items():
            df_input[col].fillna(median_val, inplace=True)
        for col, le in self.label_encoders.items():
            df_input[col] = le.transform(df_input[col])

        # Ensure correct feature order for prediction
        df_input = df_input[self.feature_order]

        # Make prediction
        prediction_proba = self.model.predict_proba(df_input)[0][1] # Probability of churn
        prediction_class = "Will Churn" if prediction_proba >= 0.5 else "Will Not Churn"

        return prediction_class, f"{prediction_proba*100:.2f}%"

# Initialize the predictor
churn_predictor = ChurnPredictor()

# Define Gradio Interface with min values for numerical inputs
iface = gr.Interface(
    fn=churn_predictor.predict_churn,
    inputs=[
        gr.Number(label="Tenure (Months)", minimum=0),
        gr.Dropdown(label="Preferred Login Device", choices=['Mobile Phone', 'Phone', 'Computer']),
        gr.Number(label="City Tier (1, 2, or 3)", minimum=1), # City tier usually starts from 1
        gr.Number(label="Warehouse to Home Distance (Km)", minimum=0),
        gr.Dropdown(label="Preferred Payment Mode", choices=['Debit Card', 'UPI', 'CC', 'Cash on Delivery', 'E wallet', 'COD', 'Credit Card']),
        gr.Dropdown(label="Gender", choices=['Female', 'Male']),
        gr.Number(label="Hour Spend on App", minimum=0),
        gr.Number(label="Number of Devices Registered", minimum=1), # At least one device is expected
        gr.Dropdown(label="Preferred Order Category", choices=['Laptop & Accessory', 'Mobile', 'Mobile Phone', 'Others', 'Fashion', 'Grocery']),
        gr.Number(label="Satisfaction Score (1-5)", minimum=1, maximum=5), # Satisfaction score is typically within a range
        gr.Dropdown(label="Marital Status", choices=['Single', 'Divorced', 'Married']),
        gr.Number(label="Number of Addresses", minimum=1), # At least one address is expected
        gr.Number(label="Complain (1 if complained, 0 otherwise)", minimum=0, maximum=1),
        gr.Number(label="Order Amount Hike From Last Year (%)", minimum=0),
        gr.Number(label="Coupon Used (Number)", minimum=0),
        gr.Number(label="Order Count (Number)", minimum=0),
        gr.Number(label="Day Since Last Order", minimum=0),
        gr.Number(label="Cashback Amount", minimum=0)
    ],
    outputs=[
        gr.Textbox(label="Churn Prediction"),
        gr.Textbox(label="Churn Probability")
    ],
    title="E-commerce Customer Churn Prediction",
    description="Enter customer details to predict whether they will churn."
)

iface.launch()

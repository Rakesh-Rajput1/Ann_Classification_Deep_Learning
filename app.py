import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

# Load the pipeline mode for preprocessing
with open('./model/model_pipeline.pkl',"rb") as f:
    load_pipeline=pickle.load(f)



# Load the Ann model for prediction
model=tf.keras.models.load_model('./model/Annmodel.h5')



# streamlit app
st.title('Customer churn Prediction')



geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


input_data = {
    'CreditScore': credit_score,
    'Geography':geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}
input_df = pd.DataFrame([input_data])

# Transform the input data
y_transform=load_pipeline.transform(input_df)

# predict data
y_pred=model.predict(y_transform)

churn=y_pred[0][0]>5

if churn:
    st.write('Customer is likely to churn.')
else:
    st.write('Customer is not likely to churn.')
st.write(f'Churn probability: {y_pred[0][0]:.2f}')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load data
data = pd.read_csv(r'D:\Credit Card Fraud Detection Using Streamlit\credit.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("This model detects whether a credit card transaction is legitimate or fraudulent.")

# Display an example input
example_input = ', '.join([str(round(val, 2)) for val in X.iloc[0, :5]])  # Use first row as an example
st.write(f"Example input (first 5 features): {example_input} ...")

# Create input fields for user to enter feature values
input_df = st.text_input("Input all features separated by commas:")
submit = st.button("Submit")

if submit:
    try:
        # Replace tabs and spaces with commas, then split
        input_df = input_df.replace('\t', ',').replace(' ', ',')
        input_df_lst = input_df.split(',')

        # Validate number of features
        if len(input_df_lst) != X.shape[1]:
            st.error(f"Invalid input! Expected {X.shape[1]} features, but got {len(input_df_lst)}.")
        else:
            # Convert to float
            features = np.array(input_df_lst, dtype=np.float64)
            features_scaled = scaler.transform(features.reshape(1, -1))  # Scale input
            prediction = model.predict(features_scaled)

            # Display result
            if prediction[0] == 0:
                st.success("Legitimate transaction")
            else:
                st.error("Fraudulent transaction")
    except ValueError:
        st.error("Invalid input. Ensure all features are numeric and separated by commas.")

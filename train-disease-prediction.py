# Importing libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib

# Reading the training data
df = pd.read_csv('Training.csv')

# Preprocessing the data
df.dropna(axis=1, inplace=True)
encoder = LabelEncoder()
df["prognosis"] = encoder.fit_transform(df["prognosis"])
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing Random Forest Model
random_forest_model = RandomForestClassifier(random_state=18)

# Creating a pipeline for Random Forest
random_forest_pipeline = Pipeline([("classifier", random_forest_model)])

# Training the Random Forest model on the training data
random_forest_pipeline.fit(X_train, y_train)

# Saving the model and related data to a file
model_data = {
    'model': random_forest_pipeline,
    'X_columns': X.columns.values,
    'encoder': encoder
}
joblib.dump(model_data, 'random_forest_model_data.joblib')

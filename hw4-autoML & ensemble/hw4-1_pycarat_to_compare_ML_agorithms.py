# Import required libraries
import pandas as pd
from pycaret.classification import *

# Step 1: Load Titanic dataset
def load_data():
    # Load dataset from seaborn or local file
    data = sns.load_dataset('titanic')
    return data

# Step 2: Preprocess the Titanic dataset
def preprocess_data(data):
    data = data.copy()

    # Handle missing values
    data['age'].fillna(data['age'].median(), inplace=True)
    data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)
    data['deck'].fillna('Unknown', inplace=True)

    # Encode categorical variables
    data = pd.get_dummies(data, columns=['sex', 'embarked', 'class', 'who', 'deck', 'embark_town', 'alive', 'alone'], drop_first=True)

    # Drop unnecessary columns
    data.drop(['name', 'ticket', 'cabin'], axis=1, inplace=True)

    return data

# Step 3: Initialize and train models using PyCaret
def train_models(data):
    # Setup PyCaret for classification
    clf = setup(
        data=data,
        target='survived',
        silent=True,
        session_id=123,
        normalize=True
    )

    # Compare multiple models and select the best one
    best_model = compare_models()

    return best_model

# Step 4: Tune the best model
def tune_best_model(model):
    tuned_model = tune_model(model)
    return tuned_model

# Step 5: Evaluate the tuned model
def evaluate_model_performance(model):
    # Evaluate model using PyCaret's visualization tools
    evaluate_model(model)

# Step 6: Apply ensemble methods
def apply_ensemble(model):
    bagged_model = ensemble_model(model, method='Bagging')
    boosted_model = ensemble_model(model, method='Boosting')

    return bagged_model, boosted_model

# Main Function
def main():
    # Load and preprocess the dataset
    data = load_data()
    preprocessed_data = preprocess_data(data)

    # Train models and get the best one
    best_model = train_models(preprocessed_data)

    # Tune the best model
    tuned_model = tune_best_model(best_model)

    # Evaluate the tuned model
    evaluate_model_performance(tuned_model)

    # Apply ensemble methods
    bagged_model, boosted_model = apply_ensemble(tuned_model)

    print("Model training and evaluation completed.")

if __name__ == "__main__":
    main()

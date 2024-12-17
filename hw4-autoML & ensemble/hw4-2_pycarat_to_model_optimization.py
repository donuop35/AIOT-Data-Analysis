# Import required libraries
import pandas as pd
import seaborn as sns
from pycaret.classification import *
import optuna

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

    # Convert 'deck' to a string type before filling
    data['deck'] = data['deck'].astype(str)
    data['deck'].fillna('Unknown', inplace=True)

    # Ensure target column 'survived' is numeric
    data['survived'] = data['survived'].astype(int)

    # Encode categorical variables
    data = pd.get_dummies(data, columns=['sex', 'embarked', 'class', 'who', 'deck', 'embark_town', 'alive', 'alone'], drop_first=True)

    # Drop unnecessary columns if they exist
    columns_to_drop = ['name', 'ticket', 'cabin']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    return data

# Step 3: PyCaret setup
def setup_environment(data):
    clf = setup(
        data=data,
        target='survived',
        normalize=True,
        feature_selection=True, # Disable polynomial features to avoid feature mismatch
        session_id=123,
        verbose=False
    )
    return clf

# Step 4: Model selection using compare_models
def model_selection():
    # Compare 16 models and return the best one
    best_model = compare_models()
    return best_model

# Step 5: Hyperparameter optimization with Optuna
def hyperparameter_optimization(data):
    def objective(trial):
        try:
            clf = setup(
                data=data,
                target='survived',
                normalize=True,
                session_id=123,
                verbose=False
            )
            # Suggest model and tune it
            model_name = trial.suggest_categorical("model", ['lr', 'rf', 'svm', 'xgboost'])
            model = create_model(model_name)
            tuned_model = tune_model(model, optimize='Accuracy')

            # Pull metrics and ensure accuracy exists
            metrics = pull()
            if 'Accuracy' in metrics.columns:
                return metrics['Accuracy'].iloc[0]
            elif 'accuracy' in metrics.columns:  # Handle lowercase
                return metrics['accuracy'].iloc[0]
            else:
                print("Accuracy metric not found. Returning 0.")
                return 0
        except Exception as e:
            print(f"Error during trial: {e}")
            return 0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print(f"Best hyperparameters: {study.best_params}")

# Step 6: Train and finalize the best model
def train_and_finalize_model(best_model):
    # Tune and finalize the model
    tuned_model = tune_model(best_model, optimize='Accuracy')
    final_model = finalize_model(tuned_model)
    return final_model

# Main Function
def main():
    # Load and preprocess the dataset
    data = load_data()
    preprocessed_data = preprocess_data(data)

    # Setup environment
    setup_environment(preprocessed_data)

    # Model selection
    best_model = model_selection()
    print(f"Best model selected: {best_model}")

    # Hyperparameter optimization using Optuna
    hyperparameter_optimization(preprocessed_data)

    # Train and finalize the best model
    final_model = train_and_finalize_model(best_model)

    # Save the final model
    save_model(final_model, 'best_titanic_model')

    print("Model training, tuning, and saving completed.")

if __name__ == "__main__":
    main()

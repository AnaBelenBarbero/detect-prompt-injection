import pandas as pd
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train(best_model=RandomForestClassifier, best_params={'random_state': 1993}):
    df = pd.read_csv("data/processed/processed_employee_data.csv")

    df_model = df[(df['EmployeeStatus'] != 'Future Start') & (df['EmployeeStatus'] != 'Leave of Absence')].copy()

    numeric_features = ['Age', 'Current Employee Rating', 'Engagement Score', 
                       'Satisfaction Score', 'Work-Life Balance Score']

    categorical_features = ['GenderCode', 'RaceDesc', 'MaritalDesc', 'State',
                           'JobFunctionDescription', 'EmployeeClassificationType',
                           'BusinessUnit', 'DepartmentType', 'Division']

    calibrated_forest = CalibratedClassifierCV(
       estimator=best_model(**best_params)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            #('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist', min_frequency=0.1, max_categories=10), 
             categorical_features)
        ])

    # Create pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', calibrated_forest)
    ])

    model = rf_pipeline.fit(df_model.drop(columns=['EmployeeStatus']), df_model['EmployeeStatus'])

    return model

if __name__ == "__main__":
    model = train(best_model=RandomForestClassifier, best_params={'random_state': 1993})

    file_path = "model.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {file_path}")
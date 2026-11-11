
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_data(
    data_path="data/Telco-Customer-Churn.csv"
):
    # 1. Load the dataset
    data = pd.read_csv(data_path)

    # 2. Convert TotalCharges to numeric
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].str.strip(),
        errors="coerce"
    ).fillna(0)

    # 3. Remove customer ID
    data = data.drop(columns=["customerID"])

    # 4. Encode the target variable
    data["Churn"] = data["Churn"].str.strip().map({
        "No": 0,
        "Yes": 1
    })

    # 5. Separate features and target
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    # 6. Create the training and test split
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 7. Split training data into model-training
    #    and simulated production batches
    x_model_train, x_production, y_model_train, y_production = (
        train_test_split(
            x_train,
            y_train,
            test_size=0.25,
            random_state=42,
            stratify=y_train
        )
    )

    # 8. Return all datasets for the next modules
    return {
        "x_model_train": x_model_train,
        "y_model_train": y_model_train,
        "x_production": x_production,
        "y_production": y_production,
        "x_test": x_test,
        "y_test": y_test
    }
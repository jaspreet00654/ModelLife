
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(x_model_train):
    """
    Build the preprocessing transformer using
    the model-training dataset's feature columns.
    """

    # Numerical columns that need scaling
    scaling_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # Remaining numerical columns
    numeric_columns = [
        column
        for column in x_model_train.select_dtypes(
            include=["int64", "float64"]
        ).columns
        if column not in scaling_columns
    ]

    # Categorical columns
    categorical_columns = x_model_train.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scaled_numeric",
                StandardScaler(),
                scaling_columns
            ),
            (
                "numeric",
                "passthrough",
                numeric_columns
            ),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            )
        ]
    )

    return preprocessor
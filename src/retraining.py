from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_candidate_model(preprocessor, x_new, y_new):
    """Train a candidate model using newly labelled data."""

    candidate_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    candidate_model.fit(x_new, y_new)

    return candidate_model
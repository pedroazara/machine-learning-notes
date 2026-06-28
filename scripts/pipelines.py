from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


def obesity_risk_pipeline(data_path, test_size=0.2, model_type="ova"):

    data = pd.read_csv(data_path)

    X = data.drop(columns="NObeyesdad")
    y = data["NObeyesdad"].astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )

    numerical_features = X.select_dtypes(include=["float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features)
        ]
    )

    if model_type == "ova":
        classifier = LogisticRegression(
            multi_class="ovr",
            max_iter=1000
        )

    elif model_type == "ovo":
        classifier = OneVsOneClassifier(
            LogisticRegression(max_iter=1000)
        )

    else:
        raise ValueError("Escolha 'ova' ou 'ovo'.")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline
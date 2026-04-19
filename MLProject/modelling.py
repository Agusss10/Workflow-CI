import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

#set tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Latihan Credit Scoring")

# load data hasil preprocessing
df = pd.read_csv("preprocessing/credit_risk_preprocessing/data_processed.csv")

# split data
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training
with mlflow.start_run():
    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)
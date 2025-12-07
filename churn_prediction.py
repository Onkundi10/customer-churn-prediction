"""
churn_prediction.py
-------------------

This script trains a simple logistic regression model to predict customer
churn based on a synthetic dataset of subscription features.  It reads the
dataset from ``data/synthetic_churn.csv``, preprocesses the features,
performs a train/test split, trains a model, evaluates its accuracy and
plots a confusion matrix.  The confusion matrix is saved to
``output/churn_confusion_matrix.png``.

The synthetic dataset is inspired by the IBM Telco Customer Churn dataset,
which includes columns such as tenure, MonthlyCharges and TotalCharges
【595890463988344†L0-L3】.  In our dataset we use tenure, monthly charges,
contract type and payment method as features and a randomly generated
``Churn`` label.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the synthetic churn dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame containing the dataset.
    """
    return pd.read_csv(csv_path)


def preprocess_features(df: pd.DataFrame):
    """Separate features and target, and scale numerical features.

    Args:
        df: DataFrame with columns 'tenure', 'MonthlyCharges',
            'ContractType', 'PaymentMethod' and 'Churn'.

    Returns:
        X_train, X_test, y_train, y_test ready for training.
    """
    X = df[['tenure', 'MonthlyCharges', 'ContractType', 'PaymentMethod']]
    y = df['Churn']
    # Scale numerical features for logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


def train_model(X_train, y_train) -> LogisticRegression:
    """Train a logistic regression model.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Trained LogisticRegression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test, y_test):
    """Evaluate the model and return accuracy and confusion matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm


def plot_confusion_matrix(cm, output_path: str) -> None:
    """Plot and save the confusion matrix.

    Args:
        cm: Confusion matrix array.
        output_path: File path for the saved figure.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['No Churn', 'Churn'])
    plt.yticks(tick_marks, ['No Churn', 'Churn'])
    plt.xlabel('Predicted')
  """
churn_prediction.py
-------------------

This script trains a simple logistic regression model to predict customer
churn based on a synthetic dataset of subscription features.  It reads the
dataset from ``data/synthetic_churn.csv``, preprocesses the features,
performs a train/test split, trains a model, evaluates its accuracy and
plots a confusion matrix.  The confusion matrix is saved to
``output/churn_confusion_matrix.png``.

The synthetic dataset is inspired by the IBM Telco Customer Churn dataset,
which includes columns such as tenure, MonthlyCharges and TotalCharges
【595890463988344†L0-L3】.  In our dataset we use tenure, monthly charges,
contract type and payment method as features and a randomly generated
``Churn`` label.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the synthetic churn dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame containing the dataset.
    """
    return pd.read_csv(csv_path)


def preprocess_features(df: pd.DataFrame):
    """Separate features and target, and scale numerical features.

    Args:
        df: DataFrame with columns 'tenure', 'MonthlyCharges',
            'ContractType', 'PaymentMethod' and 'Churn'.

    Returns:
        X_train, X_test, y_train, y_test ready for training.
    """
    X = df[['tenure', 'MonthlyCharges', 'ContractType', 'PaymentMethod']]
    y = df['Churn']
    # Scale numerical features for logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


def train_model(X_train, y_train) -> LogisticRegression:
    """Train a logistic regression model.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Trained LogisticRegression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test, y_test):
    """Evaluate the model and return accuracy and confusion matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm


def plot_confusion_matrix(cm, output_path: str) -> None:
    """Plot and save the confusion matrix.

    Args:
        cm: Confusion matrix array.
        output_path: File path for the saved figure.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['No Churn', 'Churn'])
    plt.yticks(tick_marks, ['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # Loop over data dimensions and create text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """Main entry point for churn prediction."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'synthetic_churn.csv')
    output_path = os.path.join(current_dir, 'output', 'churn_confusion_matrix.png')

    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = preprocess_features(df)
    model = train_model(X_train, y_train)
    acc, cm = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {acc:.3f}")
    print('Confusion Matrix:')
    print(cm)
    plot_confusion_matrix(cm, output_path)
    print(f'Confusion matrix saved to {output_path}')


if __name__ == '__main__':
    main()  plt.ylabel('Actual')
    # Loop over data dimensions and create text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """Main entry point for churn prediction."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'synthetic_churn.csv')
    output_path = os.path.join(current_dir, 'output', 'churn_confusion_matrix.png')

    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = preprocess_features(df)
    model = train_model(X_train, y_train)
    acc, cm = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {acc:.3f}")
    print('Confusion Matrix:')
    print(cm)
    plot_confusion_matrix(cm, output_path)
    print(f'Confusion matrix saved to {output_path}')


if __name__ == '__main__':
    main()

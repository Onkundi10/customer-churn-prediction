# Customer Churn Prediction

This project implements a basic workflow for predicting **customer churn** in
subscription‑based businesses.  Churn refers to the phenomenon where a
customer ends their subscription and stops paying for a service.  The project
uses a synthetic dataset inspired by the Telco Customer Churn dataset, which
contains information such as customerID, tenure, monthly charges and total
charges【595890463988344†L0-L3】.  In the real Telco dataset, each row
represents a customer with a boolean `Churn` column indicating whether they
cancelled service in the last month【595890463988344†L0-L3】.  Our synthetic
dataset uses similar numerical features (tenure, monthly charges, contract
type, payment method) and a randomly generated churn label for demonstration
purposes.

## Project features

* **Synthetic data generation:** A dataset of 200 customers with features
  including tenure (months), monthly charges, contract type (0 for
  month‑to‑month, 1 for longer term), payment method (0 or 1) and a churn
  label.
* **Preprocessing:** Encodes categorical features and splits the data into
  training and testing sets.
* **Model training:** Trains a logistic regression model using
  scikit‑learn to predict the churn label from the input features.
* **Evaluation:** Computes accuracy and plots a confusion matrix showing
  predicted vs actual churn values.  The confusion matrix is saved to
  `output/churn_confusion_matrix.png`.

## Getting started

1. **Install dependencies.**  You need `pandas`, `matplotlib` and
   `scikit‑learn`.  You can install them with:
# Customer Churn Prediction

This project implements a basic workflow for predicting **customer churn** in
subscription‑based businesses.  Churn refers to the phenomenon where a
customer ends their subscription and stops paying for a service.  The project
uses a synthetic dataset inspired by the Telco Customer Churn dataset, which
contains information such as customerID, tenure, monthly charges and total
charges【595890463988344†L0-L3】.  In the real Telco dataset, each row
represents a customer with a boolean `Churn` column indicating whether they
cancelled service in the last month【595890463988344†L0-L3】.  Our synthetic
dataset uses similar numerical features (tenure, monthly charges, contract
type, payment method) and a randomly generated churn label for demonstration
purposes.

## Project features

* **Synthetic data generation:** A dataset of 200 customers with features
  including tenure (months), monthly charges, contract type (0 for
  month‑to‑month, 1 for longer term), payment method (0 or 1) and a churn
  label.
* **Preprocessing:** Encodes categorical features and splits the data into
  training and testing sets.
* **Model training:** Trains a logistic regression model using
  scikit‑learn to predict the churn label from the input features.
* **Evaluation:** Computes accuracy and plots a confusion matrix showing
  predicted vs actual churn values.  The confusion matrix is saved to
  `output/churn_confusion_matrix.png`.

## Getting started

1. **Install dependencies.**  You need `pandas`, `matplotlib` and
   `scikit‑learn`.  You can install them with:

   ```bash
   pip install pandas matplotlib scikit-learn
   ```

2. **Run the analysis script:**

   ```bash
   python churn_prediction.py
   ```

   This will load the synthetic dataset from `data/synthetic_churn.csv`,
   train the model, print the accuracy to the console and generate a
   confusion matrix image saved to `output/churn_confusion_matrix.png`.

## Files and directories

| Path | Description |
| --- | --- |
| `data/synthetic_churn.csv` | Synthetic dataset of 200 customers with features and churn label. |
| `churn_prediction.py` | Python script for training and evaluating a churn prediction model. |
| `output/churn_confusion_matrix.png` | Output image of the confusion matrix showing model performance. |
| `README.md` | Project documentation and usage instructions. |

## License

This project is released under the MIT License.  The synthetic dataset is
inspired by publicly available fields in the Telco Customer Churn data
source【595890463988344†L0-L3】.

   ```bash
   pip install pandas matplotlib scikit-learn
   ```

2. **Run the analysis script:**

   ```bash
   python churn_prediction.py
   ```

   This will load the synthetic dataset from `data/synthetic_churn.csv`,
   train the model, print the accuracy to the console and generate a
   confusion matrix image saved to `output/churn_confusion_matrix.png`.

## Files and directories

| Path | Description |
| --- | --- |
| `data/synthetic_churn.csv` | Synthetic dataset of 200 customers with features and churn label. |
| `churn_prediction.py` | Python script for training and evaluating a churn prediction model. |
| `output/churn_confusion_matrix.png` | Output image of the confusion matrix showing model performance. |
| `README.md` | Project documentation and usage instructions. |

## License

This project is released under the MIT License.  The synthetic dataset is
inspired by publicly available fields in the Telco Customer Churn data
source【595890463988344†L0-L3】.

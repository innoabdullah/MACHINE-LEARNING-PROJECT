# End-to-End Medical Triage Assistant using Machine Learning

## Project Overview
This project is developed as part of **LAB #14 – Complex Computing Activity**.  
The aim of this project is to design and implement an end-to-end Machine Learning system that assists in **medical triage** by classifying patients into different risk levels based on medical data.

The system uses multiple machine learning models and compares their performance to identify the most suitable model for healthcare decision support.



## Objectives
- To apply multiple Machine Learning concepts learned in previous labs
- To preprocess and analyze medical data
- To train and evaluate different classification models
- To provide interpretable results for medical decision-making
- To understand ethical considerations in healthcare ML systems

## Dataset
- **Dataset Used:** Breast Cancer Wisconsin Dataset  
- **Source:** Scikit-learn / Kaggle  
- **Type:** Tabular medical data  
- **Target Classes:**  
  - Low Risk  
  - Medium Risk  
  - High Risk  



## Machine Learning Models Used
The following models were implemented and compared:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- Naïve Bayes  
- K-Nearest Neighbors (KNN)  



## Integrated Lab Concepts
This project integrates concepts from the following labs:

- Lab 05: Random Forest Classifier  
- Lab 06: Support Vector Machine  
- Lab 08: KNN Classifier  
- Lab 09: Naïve Bayes Classifier  
- Lab 10: K-Means Clustering (data insight / anomaly observation)  
- Lab 12: Real-time decision support concept  
- Lab 14: Complex system integration  



## Project Structure
Medical-Triage-ML/
│
├── data/
│ └── dataset.csv
│
├── notebooks/
│ ├── preprocessing.ipynb
│ ├── model_training.ipynb
│ ├── evaluation.ipynb
│ └── explainability.ipynb
│
├── results/
│ ├── confusion_matrix.png
│ └── roc_curve.png
│
├── README.md
└── requirements.txt

yaml
Copy code

## Evaluation Metrics
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

Special focus was given to **Recall**, as false negatives can be critical in medical applications.



## Explainability
Feature importance from the Random Forest model was used to understand which medical features contributed most to the prediction.  
This helps improve trust and transparency in the system.

## Ethical Considerations
- The model is intended to **assist**, not replace, medical professionals
- Predictions should not be used without expert validation
- Dataset bias and misclassification risks are acknowledged



## How to Run
1. Clone the repository
2. Install dependencies:
pip install -r requirements.txt

yaml
Copy code
3. Run notebooks in sequence from the `notebooks` folder


## Author
**Name:**RAJA ABDULLAH 
**Program:** Bachelor of Engineering / AI  
**Course:** Machine Learning Lab  


# MFM_bopf
Maternal Fetal Morbidity study using Python and machine and deep learning to identify some potential markers to decrease the incidence of mortality.
This is repository is a store of the work of Mike Bopf, in assocication with Laritza Rodiquez at the National Library of Medicine.
## Data Overview
This study used data extracted from the Consortium for Safe Labor (CSL) dataset created by the Eunice Kennedy Shriver National Institute of Child Health and Human Development (NICHD). It includes antepartum, intrapartum, and postpartum medical histories of 224,438 women from 12 hospitals in the United States.
## Outcomes
Separate models were constructed to predict two target outcomes. The first and primary outcome included a composite and included all those that received transfusion of any blood products or had a postpartum hemorrhage defined by documented blood loss of 1,000 mL or more during or after delivery, and the second outcome was all patients that received transfusion of any blood products.
## Machine Learning algorithms
The goal of the project was to compare different machine and deep learning algorithms against each other. We applied the following algorithms using a 30/70 test/train split:
- Logistic Regression
- Support Vector Machines
- Multi-Layer Perceptron
- Random Forest
- Gradient Boosting

We also ran two deep learning algorithms:
- TensorFlow Imbalanced example
- Learned Embedding

Hyperparameter tuning for all algorithms was done using a customized grid search technique. 

## Performance Statistics
Due to the imbalanced nature of this dataset (4-8% positive cases, depending on the outcome target), we used a number of primary statistics for comparison:
- MCC: (Matthews Correlation Coefficient
- ROC-AUC: Receiver Operating Characteristic - Area Under the Curve
- PR-AUC: Precision/Recall - Area Under the Curve
- F2: Modified F-score skewed towards Recall
- Raw Confusion Matrix values

## Software Dependencies
Some of the MFM_mbopf notebooks and code depends on some convenience routines
found in this repo: [mwb_common](https://github.com/mbopf/mwb_common)

The way I've handled this is to make sure the code in mwb_common is included
in the PYTHONPATH environment variable.

## Software Versions
Vast majority of the code (both Python and Jupyter Notebooks), was run using Python 3.6 although later on version 3.9 was used. I don't believe any 3.9-specific code was used, 
however. These versions should work for most of the code:

| Package | Version |
|---------|---------|
| Python  | 3.6 |
| numpy | 1.21 |
| pandas  | 1.2 |
| scikit-learn | 0.24 |
| TensorFlow | 2.4 |
| keras | 2.4 |
| imbalanced-learn | 0.9.0 |
| jupyter | 1.0.0 |
| Matplotlib | 3.5 |

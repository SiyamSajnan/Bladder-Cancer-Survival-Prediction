# Bladder Cancer Survival Prediction with Machine Learning

## Project Description
This project explores machine learning and deep learning models to predict survival outcomes in bladder cancer patients. The study compares various machine learning models and ensemble methods, assessing their effectiveness in survival prediction based on patient characteristics. The models evaluated include Random Forest, SVM, AdaBoost, K-Nearest Neighbors, XGBoost, Artificial Neural Networks (ANN), and ensemble techniques using hard and soft voting classifiers.

## Objectives
- **Primary Objective**: To determine the accuracy and reliability of machine learning models in predicting survival for bladder cancer patients.
- **Comparative Analysis**: Evaluate model performance across various metrics, including accuracy, precision, recall, and F1-score, to identify the best performing model(s) for bladder cancer survival prediction.

## Dataset and Preprocessing
- **Source**: The dataset was sourced from the cBioPortal, specifically the Bladder Cancer (MSK Cell Report) dataset, containing 1659 samples.
- **Features**: The dataset includes clinical, genetic, and physiological features, such as age, cancer type, ethnicity, mutation count, MSI score, treatment history, and overall survival status.
- **Preprocessing**: Steps included dropping irrelevant or sparse columns, handling missing values, encoding categorical data, and downsampling to address class imbalance.

## Models Employed
- **Random Forest**: An ensemble learning model based on multiple decision trees.
- **Support Vector Machine (SVM)**: Utilizes hyperplanes for classification tasks, with different kernel options.
- **AdaBoost**: An ensemble method that combines multiple weak classifiers.
- **K-Nearest Neighbors (KNN)**: Classifies samples based on proximity to other data points.
- **XGBoost**: A gradient boosting method optimized for speed and performance.
- **Artificial Neural Network (ANN)**: A neural network model with layers mimicking human brain structures.
- **Voting Classifier**: Ensemble approach using hard and soft voting of the best-performing models.

## Methodology
- **Hyperparameter Tuning**: Employed GridSearchCV for parameter optimization across models.
- **Ensemble Techniques**: Created voting classifiers (both hard and soft) to leverage model strengths.
- **Evaluation Metrics**: Used accuracy, precision, recall, and F1-score to measure performance across both regular and downsampled datasets.

## Results
- **Best Model**: Hard voting classifier performed best on the undersampled dataset, achieving an accuracy of 90%.
- **AdaBoost**: The highest-performing individual model, with an accuracy of 89% on the downsampled dataset.
- **Overall Findings**: Ensemble methods provided minor improvements over individual models. Undersampling enhanced model performance by reducing bias in the minority class, especially in metrics such as recall and F1-score.

## Conclusion
The study concludes that ensemble methods, particularly the hard voting classifier, yield the best results for bladder cancer survival prediction. However, AdaBoost also showed strong performance among individual models. This project highlights the potential of machine learning models in assisting clinical prognosis for bladder cancer patients.

## Future Directions
Future research could explore deep learning models, different upsampling techniques like SMOTE, and other advanced methods to improve prediction accuracy and handle class imbalance more effectively.

## References
1. SEER Cancer Stat Facts: Bladder Cancer, National Cancer Institute.
2. Tsai et al., "Machine learning in prediction of bladder cancer on clinical laboratory data," Diagnostics, 2022.
3. Hasnain et al., "Machine learning models for predicting post-cystectomy recurrence and survival in bladder cancer patients," PLoS ONE, 2019.


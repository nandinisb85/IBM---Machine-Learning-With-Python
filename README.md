# IBM---Machine-Learning-With-Python
### Overview

This project involves training and evaluating different machine learning models to predict rain in Australia using weather data. The models tested include Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Support Vector Machine (SVM).

### Key Steps

1. **Importing Libraries**: 
   - Various Python libraries are imported, including `pandas`, `numpy`, `scikit-learn` (for model training and evaluation), and others.

2. **Data Downloading**: 
   - The weather dataset is downloaded from a specified URL using the `requests` library.

3. **Model Training**:
   - Several models are trained on the dataset:
     - **Logistic Regression (LR)**
     - **K-Nearest Neighbors (KNN)**
     - **Decision Tree**
     - **Support Vector Machine (SVM)**

4. **Model Evaluation**:
   - The trained models are evaluated using several metrics:
     - **Accuracy**: Measures the percentage of correct predictions.
     - **Jaccard Score**: A metric for assessing the similarity between predicted and actual labels.
     - **F1 Score**: The harmonic mean of precision and recall.
     - **Log Loss**: Used to evaluate the performance of classification models where the prediction is a probability.

5. **Model Performance**:
   - The performance of each model is summarized in a table, showing the accuracy, Jaccard score, F1 score, and Log Loss.

### Results

- The models' accuracy ranges from about 61% to 72%.
- The SVM and Logistic Regression models have the highest accuracy (72.2%), but both models show poor Jaccard and F1 scores.
- The Decision Tree model has the highest Jaccard and F1 scores but lower accuracy.

### Conclusion

The notebook suggests that while accuracy is a crucial metric, other metrics like F1 score and Jaccard index are also important, especially in imbalanced datasets like rain prediction, where correctly predicting the minority class (rain) is critical.

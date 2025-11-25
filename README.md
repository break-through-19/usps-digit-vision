# Handwritten Digit Classification

This project implements and compares three different machine learning algorithms for classifying handwritten digits using the USPS dataset.

## 1. Algorithms Used

The following algorithms were implemented and evaluated:

*   **Decision Tree Classifier**: A tree-structured classifier, implemented using `sklearn.tree.DecisionTreeClassifier`.
*   **K-Nearest Neighbors (KNN)**: Implemented using `sklearn.neighbors.KNeighborsClassifier`, supporting both uniform and distance-based weighting.
*   **Support Vector Machine (SVM)**: A powerful classifier implemented using `sklearn.svm.SVC`.

## 2. Hyperparameter Tuning

Various hyperparameters were tuned for each model to optimize performance:

*   **Decision Tree**:
    *   `max_leaf_nodes`: Tested with `None` (unlimited) and restricted values (e.g., 10) to observe the effect of tree complexity.
    *   `min_samples_split`: Varied between 2 and 10 to control overfitting.
    *   `min_samples_leaf`: Varied between 1 and 5.
    *   `criterion`: Compared 'gini' impurity and 'entropy' (information gain).

*   **K-Nearest Neighbors (KNN)**:
    *   `k` (Number of Neighbors): Tested values of 1, 3, 5, and 7.
    *   `weights`: Compared 'uniform' (majority vote) and 'distance' (inverse distance weighting) strategies.

*   **Support Vector Machine (SVM)**:
    *   `kernel`: Compared 'linear' and 'poly' (polynomial) kernels.
    *   `degree`: For the polynomial kernel, degrees 2 and 3 were evaluated.
    *   `decision_function_shape`: Used 'ovr' (one-vs-rest) for multiclass classification.

## 3. Preprocessing Steps

The data preprocessing pipeline involved the following steps:

1.  **Data Loading**: The USPS dataset was loaded from `USPS_all.mat`.
2.  **Data Splitting**: The dataset was split into a training set (first 7291 samples) and a test set (remaining 2007 samples).
3.  **Normalization**: The feature data (`X`) was normalized using Z-score normalization (`StandardScaler`). This ensures that all features contribute equally to the distance calculations (crucial for KNN and SVM) and helps with convergence.

## 4. High-Level Comparison of Algorithms

Based on the evaluation results:

*   **Support Vector Machine (SVM)**: Generally provided robust performance. The comparison between linear and polynomial kernels helps determine if the data is linearly separable or requires mapping to a higher-dimensional space.
*   **K-Nearest Neighbors (KNN)**: The custom implementation demonstrated the effectiveness of instance-based learning. Lower `k` values typically capture local structures well but can be sensitive to noise, while higher `k` values provide smoother decision boundaries. Distance weighting often improves performance by giving more influence to closer neighbors.
*   **Decision Tree**: While interpretable, Decision Trees can be prone to overfitting. Restricting the tree size (e.g., via `max_leaf_nodes`) was shown to impact accuracy, highlighting the trade-off between model complexity and generalization. The visualization of the tree structure provides insights into the most discriminative pixels.

The project also includes automated insights that identify:
*   The worst-performing class for each model.
*   The most frequent misclassifications (confusion matrix analysis).
*   The overall hardest and easiest digits to classify across all models.

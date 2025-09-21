# Codealpha_Diseases_Prediction_form_Medical_data:

### 1. Data Loading and Exploration
The notebook begins by importing necessary libraries like NumPy, Pandas, and Matplotlib. It then loads a CSV file named `Dataset for heart diseases.csv` into a Pandas DataFrame. The `.info()` method reveals that the dataset contains 297 entries and 14 columns, with all columns having a non-null `int64` or `float64` data type. The `.describe()` method is used to provide a statistical summary of the dataset, including the mean, standard deviation, minimum, and maximum values for each column.

### 2. Data Visualization and Preprocessing
To understand the data better, the notebook generates a few visualizations:
- A **correlation matrix** is displayed as a heatmap, showing the pairwise correlations between all 14 columns.
- Histograms are plotted for each column, providing a visual representation of the distribution of values.
- A bar chart shows the count of each class in the **`condition`** column, with `0` representing no heart disease and `1` representing a heart condition. The plot shows that the classes are relatively balanced.
- **One-hot encoding** is applied to categorical columns (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) using `pd.get_dummies()`. This converts these categorical features into a format suitable for machine learning algorithms.
- **Feature scaling** is performed on the continuous numerical columns (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) using `StandardScaler`. This process standardizes the features by removing the mean and scaling to unit variance, which is important for distance-based algorithms like KNN and SVM.

### 3. Model Training and Evaluation
The dataset is split into a training set and a testing set, with 33% of the data allocated for testing. The notebook then explores four different classification models:

- **K-Nearest Neighbors (KNN):**
  - The model is trained and evaluated for `k` values ranging from 1 to 20.
  - A line plot visualizes the model's accuracy scores for each `k` value.
  - The output indicates that the highest accuracy score is **78.79%**, achieved with 8 neighbors.

- **Support Vector Classifier (SVC):**
  - The model is tested using four different kernel functions: **'linear'**, **'poly'**, **'rbf'**, and **'sigmoid'**.
  - A bar chart displays the accuracy scores for each kernel.
  - The model achieves its best score of **85.86%** with the 'linear' kernel.

- **Decision Tree Classifier:**
  - The model is trained by varying the `max_features` parameter from 1 up to the total number of features.
  - A line plot shows the model's score for each `max_features` value.
  - The best score for this model is **72.73%**, which is achieved with `max_features` values of 2, 4, and 18.

- **Random Forest Classifier:**
  - The model is trained and evaluated using different numbers of estimators (`n_estimators`): 10, 100, 200, 500, and 1000.
  - A bar chart displays the accuracy scores for each number of estimators.
  - The highest score of **84.85%** is achieved with 100 or 500 estimators.

### 4. User Input and Prediction
The final part of the notebook creates a command-line interface to allow a user to input values for all the features in the dataset. The user's input is then converted into a DataFrame and preprocessed using the same `StandardScaler` that was fit on the training data. Finally, the preprocessed user data is passed to each of the trained models to make a prediction, and the predictions from KNN, SVC, Decision Tree, and Random Forest are printed.

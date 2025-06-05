# AIML_Internship_Task7
# Breast Cancer Classification using SVM

This task demonstrates binary classification on the Breast Cancer dataset from Kaggle using Support Vector Machines (SVM) with both **linear** and **RBF kernels**

## Preprocessing

1. Loaded dataset using `pandas`
2. Dropped unnecessary columns like `id`
3. Encoded target variable: `M` → 1, `B` → 0
4. Performed **standardization** using `StandardScaler`
5. Optionally applied **PCA** to reduce to 2D for visualization

## Model: Support Vector Machine (SVM)

I trained two models:
- `SVC(kernel='linear')`
- `SVC(kernel='rbf')`

### Hyperparameter Tuning (Manual)

I manually iterated over:
- `C`: [0.1, 1, 10, 100]
- `gamma`: [1, 0.1, 0.01, 0.001]

For each combination, we trained the model and calculated accuracy on the test set to find the best hyperparameters.

## Visualization

- **Decision boundary** was visualized using:
  - `plt.contourf` to shade prediction regions
  - `plt.scatter` to show data points colored by class
  - `ListedColormap(['blue', 'red'])` to map:
    - Class 0 → Blue
    - Class 1 → Red
- **Legend** added using `plt.legend()` to label each class

## Cross-Validation

Performed manual cross-validation using `cross_val_score`

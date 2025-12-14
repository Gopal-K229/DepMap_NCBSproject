#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc

# ---------------------------------------------------------
# 1. Data Loading and Splitting
# ---------------------------------------------------------
df = pd.read_csv("processed_df_with_all_target_genes.csv", index_col=0)

# Filter for Skin lineage
df = df[df['lineage_1_x'] == "Breast"]
#%%
# Define Identifiers and Target
# In R, you used update_role to set these as IDs. In Python, we typically just separate them.
target_col = 'BRCA1'
id_cols = ['depmap_id', 'cell_line_display_name', 'lineage_1_x', 'PARP1','BRCA2','TP53', 'BRIP1']

# Separate Features (X) and Target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify feature columns (removing IDs from the training set)
# We handle the ID drop in the column selection or manually here
X_model = X.drop(columns=id_cols, errors='ignore')

# Set Seed and Split (prop = 3/4)
# random_state=222 matches set.seed(222)
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, train_size=0.75, random_state=222
)
#%%
# ---------------------------------------------------------
# 2. Recipe / Pipeline Construction
# ---------------------------------------------------------

# Select numerical and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns

# Create transformers (equivalent to step_impute, step_dummy, step_center, step_scale)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # step_impute_median
    ('scaler', StandardScaler())                    # step_center + step_scale
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # step_impute_mode
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # step_dummy
])

# Combine into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    verbose_feature_names_out=False # Keeps original names if possible
)

# Define Model: Elastic Net
# R: mixture = 0.5 -> Python: l1_ratio = 0.5
# R: penalty -> Python: alpha
lasso_model = ElasticNet(l1_ratio=0.5, max_iter=10000, random_state=345)

# Create full workflow (Pipeline)
# Unlike R, we typically don't add step_zv (zero variance) explicitly, 
# as ElasticNet handles colinearity well, but StandardScaler handles scaling.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', lasso_model)])

# ---------------------------------------------------------
# 3. Hyperparameter Tuning (Grid Search)
# ---------------------------------------------------------

# Define Grid
# R: penalty(range = c(-3, -0.5), levels = 50) -> Log scale
# np.logspace takes base 10 by default. -3 to -0.5
alphas = np.logspace(-10, 0, 50)

param_grid = {
    'classifier__alpha': alphas
}

# Cross Validation (v = 5)
cv_folds = KFold(n_splits=5, shuffle=True, random_state=345)

grid_search = GridSearchCV(
    clf, 
    param_grid, 
    cv=cv_folds, 
    scoring='neg_root_mean_squared_error', # Sklearn maximizes score, so neg_rmse
    return_train_score=True,
    n_jobs=-1
)

print("Tuning hyperparameters...")
grid_search.fit(X_train, y_train)

# Get best results
best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_['classifier__alpha']
print(f"Best Penalty (Alpha): {best_alpha}")

# ---------------------------------------------------------
# 4. Extract Coefficients (VIP Plot Prep)
# ---------------------------------------------------------

# Access the model and feature names from the pipeline
final_elastic_net = best_model.named_steps['classifier']

# Get feature names after one-hot encoding
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()

# Create a DataFrame of coefficients
coef_df = pd.DataFrame({
    'term': feature_names,
    'estimate': final_elastic_net.coef_
})

# Filter: Remove intercept (not in coef_ anyway) and zero coefficients
important_genes = coef_df[coef_df['estimate'] != 0].copy()
important_genes['abs_estimate'] = important_genes['estimate'].abs()
important_genes = important_genes.sort_values(by='abs_estimate', ascending=False)

print("\nTop 10 Genes:")
print(important_genes.head(10)[['term', 'estimate']])

# Save to CSV (Optional)
# important_genes.drop(columns='abs_estimate').to_csv("significant_genes_elastic_net_py.csv", index=False)

# ---------------------------------------------------------
# 5. Evaluation and Predictions
# ---------------------------------------------------------

# Predict on Test Set
y_pred = best_model.predict(X_test)

# Collect Metrics
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
r2_val = r2_score(y_test, y_pred)

print(f"RMSE: {rmse_val}")
print(f"R-Squared: {r2_val}")

# Create Prediction DataFrame
test_predictions = pd.DataFrame({
    'BRCA2': y_test.values,
    '.pred': y_pred
})

# ---------------------------------------------------------
# 6. Visualizations
# ---------------------------------------------------------

# Set generic plot style to match R theme_minimal/bw roughly
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# --- Plot A: Tuning Results (RMSE vs Penalty) ---
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['mean_test_rmse'] = -results_df['mean_test_score'] # Convert back to positive RMSE

plt.figure(figsize=(10, 6))
plt.plot(results_df['param_classifier__alpha'], results_df['mean_test_rmse'], marker='o')
plt.xscale('log')
plt.axvline(x=best_alpha, color='red', linestyle='--', label=f'Best Alpha: {best_alpha:.4f}')
plt.xlabel("Regularization Penalty (Log Scale)")
plt.ylabel("RMSE")
plt.title("Hyperparameter Tuning: Finding the Optimal Penalty\n(The red line marks the minimized RMSE)")
plt.legend()
plt.show()

# --- Plot B: Feature Importance (Top 20 Genes) ---
top_genes = important_genes.head(20).copy()
top_genes['Color'] = np.where(top_genes['estimate'] > 0, 'steelblue', 'firebrick')

plt.figure(figsize=(10, 8))
# Reorder for plotting (create a range)
y_pos = np.arange(len(top_genes))
plt.barh(y_pos, top_genes['estimate'], color=top_genes['Color'])
plt.yticks(y_pos, top_genes['term'])
plt.gca().invert_yaxis() # Put largest at top
plt.xlabel("Coefficient (Importance)")
plt.title("Top 20 Genetic Predictors of BRCA2 Dependency\nPositive = Co-dependency | Negative = Compensatory/Resistance")
plt.grid(axis='y')
# Save plot
plt.tight_layout()
plt.savefig("feature_importance_plot_py.png", dpi=300)
plt.show()


# --- Plot C: Predicted vs Actual ---
plt.figure(figsize=(8, 6))
plt.scatter(test_predictions['BRCA2'], test_predictions['.pred'], 
            alpha=0.6, s=50, color="#2c3e50")
# Add diagonal line
limits = [min(min(test_predictions['BRCA2']), min(test_predictions['.pred'])),
          max(max(test_predictions['BRCA2']), max(test_predictions['.pred']))]
plt.plot(limits, limits, ls="--", c="firebrick", lw=2)

# Annotation for R2
plt.text(x=max(test_predictions['BRCA2']), y=min(test_predictions['.pred']), 
         s=f"R² = {r2_val:.2f}", 
         ha='right', va='bottom', fontsize=14, fontweight='bold')

plt.xlabel("Actual BRCA2 Dependency Score (CRISPR)")
plt.ylabel("Predicted Score")
plt.title("Model Performance: Predicted vs. Actual Dependency\nElastic Net Model (Mixture = 0.5)")
plt.show()


# --- Plot D: ROC Curve (Classification Logic) ---
# Truth: Is the cell line ACTUALLY dependent? (< -0.5)
# In R code: factor(ifelse(brca1 < -0.5, "Dependent", "Not_Dependent"))
# Note: In ROC, "Dependent" (the event) usually corresponds to 1.
# Here we define Dependent as 1, Not_Dependent as 0.

binary_truth = (test_predictions['BRCA2'] < -0.5).astype(int) 

# For Regression to Classification ROC, we use the predicted continuous value.
# Since lower BRCA1 score = higher dependency, we need to invert the prediction 
# or flip the logic for the ROC calculation to make sense (Dependent = Positive Class).
# Standard ROC expects higher values = Positive class. 
# Since Dependent is NEGATIVE score, we negate the prediction so lower becomes higher.
fpr, tpr, thresholds = roc_curve(binary_truth, -test_predictions['.pred'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve: Distinguishing Dependent Cell Lines\nAbility to classify 'Essential' vs 'Non-Essential' lines")
plt.legend(loc="lower right")
plt.show()


# --- Plot E: Residual Plot ---
residuals = test_predictions['BRCA2'] - test_predictions['.pred']

plt.figure(figsize=(8, 6))
plt.scatter(test_predictions['.pred'], residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Value")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot: Where are the errors?")
plt.show()
# %%

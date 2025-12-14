#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df_master = pd.read_csv("processed_df_with_all_target_genes.csv", index_col=0)
df_master=df_master[df_master["lineage_1_x"]=="Breast"]

#%%
# ==========================================
# PART 2: THE ANALYSIS (STEP 3)
# Goal: Find which features correlate with PARP1_Dependency
# ==========================================

# 1. Define Target
target_col = 'BRCA1'

# 2. Define Features (Exclude non-numeric metadata and the target itself)
# We select only numeric columns to avoid errors with IDs or Strings
numeric_df = df_master.select_dtypes(include=[np.number])
features = [col for col in numeric_df.columns if col != target_col and col != 'Unnamed: 0']

results = []

print(f"Running Univariate Analysis for Target: {target_col}...")

for feature in features:
    # Get data vectors
    # IMPORTANT: Drop NA values for this specific pair, otherwise pearsonr fails
    temp_df = df_master[[feature, target_col]].dropna()
    
    if len(temp_df) < 2: # Skip if not enough data points
        continue
        
    x = temp_df[feature]
    y = temp_df[target_col]
    
    # Calculate Correlation (Pearson)
    corr, p_val = stats.pearsonr(x, y)
    
    # Store result
    results.append({
        'Feature': feature,
        'Correlation': corr,
        'P_Value': p_val,
        'Abs_Corr': abs(corr)
    })

# Convert to DataFrame and Sort
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='P_Value', ascending=True)

# Display Top 5 Predictors
print("\nTop 5 Strongest Predictors:")
print(df_results[['Feature', 'Correlation', 'P_Value']].head(5))

# ==========================================
# PART 3: VISUALIZATION
# ==========================================
plt.figure(figsize=(14, 6))

# --- PLOT A: Volcano Plot (Correlation vs Significance) ---
plt.subplot(1, 2, 1)
df_results['log_pval'] = -np.log10(df_results['P_Value'])

# Scatter all points gray
sns.scatterplot(data=df_results, x='Correlation', y='log_pval', color='grey', alpha=0.5)

# Highlight Top 5 Significant features in Red
top_features = df_results.head(5)
sns.scatterplot(data=top_features, x='Correlation', y='log_pval', color='red', s=100)

# Add labels to the top hits (adjusting position slightly to avoid overlap)
for i, row in top_features.iterrows():
    plt.text(row['Correlation'], row['log_pval'], row['Feature'], fontsize=9, fontweight='bold')

plt.axvline(0, linestyle='--', color='black', alpha=0.3)
plt.title(f'Volcano Plot of Predictors for {target_col}')
plt.xlabel('Correlation Coefficient (r)')
plt.ylabel('-log10(P-Value)')

# --- PLOT B: Scatter/Boxplot of the #1 Best Predictor ---
plt.subplot(1, 2, 2)
best_feature = df_results.iloc[0]['Feature']
corr_val = df_results.iloc[0]['Correlation']
pval_val = df_results.iloc[0]['P_Value']

# Clean data again for plotting the best feature
plot_data = df_master[[best_feature, target_col]].dropna()

# Check if the feature name suggests it is categorical (Binary Mutation)
# Based on your columns: "Damaging Mutations XPA", etc.
if "Mutations" in best_feature:
    sns.boxplot(data=plot_data, x=best_feature, y=target_col, palette="Set2")
    sns.stripplot(data=plot_data, x=best_feature, y=target_col, color='black', alpha=0.3, jitter=True)
    plt.title(f"Top Hit: {best_feature}\n(Discrete vs Continuous)")
else:
    sns.regplot(data=plot_data, x=best_feature, y=target_col, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title(f"Top Hit: {best_feature}\nr = {corr_val:.2f}, p = {pval_val:.2e}")

plt.ylabel(f'{target_col} Score')
plt.xlabel(best_feature)
plt.tight_layout()
plt.show()
# %%

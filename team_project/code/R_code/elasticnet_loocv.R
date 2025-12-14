# 1. Load Libraries
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(janitor)
library(ggplot2)
library(tidymodels)
library(corrr)
library(glmnet)
library(vip)
library(doParallel) # For parallel processing

# 2. Setup Parallel Processing (CPU optimization)
# This uses all available CPU cores to speed up the tuning loop.
all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

# Prefer tidymodels handling of conflicts
tidymodels_prefer()

# 3. Load and Filter Data
# Ensure "tulika_dataframe.csv" is in your working directory
df <- read.csv("tulika_dataframe.csv")

# Filtering for Breast lineage (adjust column name if needed)
if("lineage_1" %in% names(df)) {
  df <- df %>% filter(lineage_1 == "Breast")
} else {
  message("Warning: 'lineage_1' column not found. Skipping filter.")
}

set.seed(222)

# 4. Split Data
# With N=60, a 3/4 split gives ~45 training and ~15 testing samples
df_split <- initial_split(df, prop = 3/4, strata = brca1)
df_train <- training(df_split)
df_test  <- testing(df_split)

# 5. Cross-Validation Setup
# Replaced LOOCV with Repeated 5-Fold CV. 
# This splits data into 5 chunks, repeats the process 10 times.
# It is stable and supported by tune_grid.
set.seed(345)
cv_folds <- vfold_cv(df_train, v = 5, repeats = 2)

# 6. Recipe (Data Preprocessing)
lasso_rec <- 
  recipe(brca1 ~ ., data = df_train) %>%
  # Set ID columns so they aren't used as predictors but are kept in data
  update_role(depmap_id, cell_line_display_name, lineage_1, new_role = "ID") %>%
  # Handle missing values
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # Convert categories to dummy variables
  step_dummy(all_nominal_predictors()) %>%
  # Remove variables with zero variance (all same value)
  step_zv(all_predictors()) %>%
  # Normalize numeric predictors (Crucial for Elastic Net)
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors())

# 7. Model Specification: Elastic Net
# We tune both 'penalty' (lambda) and 'mixture' (alpha)
# mixture = 1 is Lasso, mixture = 0 is Ridge. We let the model decide.
elastic_net_model <- 
  linear_reg(
    penalty = tune(), 
    mixture = tune()
  ) %>% 
  set_engine("glmnet")

# 8. Workflow
enet_wf <- 
  workflow() %>%
  add_model(elastic_net_model) %>%
  add_recipe(lasso_rec)

# 9. Tuning Grid
# Create a grid of parameters to test
# levels = c(50, 5) means 50 penalty values and 5 mixture values
enet_grid <- grid_regular(penalty(), mixture(), levels = c(50, 5))

print("Starting model tuning... this may take a moment.")

# 10. Run Tuning (This runs in parallel)
enet_tuning_results <- 
  enet_wf %>% 
  tune_grid(cv_folds,
            grid = enet_grid,
            control = control_grid(save_pred = TRUE))

# 11. Select Best Model
# We pick the hyperparameters that gave the lowest RMSE (Root Mean Squared Error)
best_params <- enet_tuning_results %>%
  select_best(metric = "rmse")

print("Best Parameters found:")
print(best_params)

# 12. Finalize Workflow
final_enet_wf <- 
  enet_wf %>%
  finalize_workflow(best_params)

# 13. Final Fit on Training + Evaluation on Test
# last_fit() fits the final model on the FULL training set 
# and evaluates it on the hold-out test set automatically.
final_fit <- final_enet_wf %>%
  last_fit(df_split)

# 14. Output Results
# Collect metrics (RMSE, R2) on the test set
test_metrics <- collect_metrics(final_fit)
print("Test Set Metrics:")
print(test_metrics)

# Variable Importance Plot (VIP)
# Shows which genes/features were most important for the prediction
vip_plot <- final_fit %>%
  extract_workflow() %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20)

print(vip_plot)

# Stop the parallel cluster to free up resources
stopCluster(cl)


#######################################################################################
########################################################################################

# Extract the underlying model object
final_model_obj <- extract_fit_parsnip(final_fit)

# Extract coefficients (genes) that were NOT removed (non-zero)
# This creates a clean table of Gene Name and its Weight (estimate)
important_genes <- tidy(final_model_obj) %>%
  filter(term != "(Intercept)") %>%  # Remove the intercept
  filter(estimate != 0) %>%          # Keep only non-zero coefficients
  arrange(desc(abs(estimate)))       # Sort by magnitude of importance

# View the top 10 genes in console
print(head(important_genes, 10))

# STORE: Save this list to a CSV file for your report/paper
write_csv(important_genes, "significant_genes_elastic_net.csv")



# Extract metrics (RMSE and R-squared)
final_metrics <- collect_metrics(final_fit)

# STORE: Save metrics to a CSV
write_csv(final_metrics, "model_performance_metrics.csv")


# Extract predictions from the test set
test_predictions <- collect_predictions(final_fit)

# STORE: Save predictions to CSV
# This allows you to plot "Predicted vs Actual" in Excel or R later
write_csv(test_predictions, "test_set_predictions.csv")


# STORE: Save the Variable Importance Plot
# You can adjust width/height (in inches)
ggsave("feature_importance_plot.png", plot = vip_plot, width = 8, height = 6, dpi = 300)


# Plot Predicted vs Observed
ggplot(test_predictions, aes(x = brca1, y = .pred)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(lty = 2, color = "red") + # Perfect prediction line
  labs(
    title = "Elastic Net: Predicted vs Actual BRCA1 Scores",
    x = "Actual BRCA1 Score",
    y = "Predicted Score"
  ) +
  theme_minimal()
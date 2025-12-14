
library(readr); library(dplyr); library(tidyr); library(stringr)
library(janitor); library(ggplot2); library(patchwork) # patchwork optional for arranging plots
library(tidymodels); library(corrr); library(glmnet)
library(vip)
tidymodels_prefer()

df<-read.csv("tulika_dataframe.csv")
df <- df[df$lineage_1 == "Breast", ]
set.seed(222)

df_split <- initial_split(df, prop = 3/4, strata = brca1)

df_train <- training(df_split)
df_test  <- testing(df_split)

set.seed(345)
cv_folds <- vfold_cv(df_train, v = 5)


lasso_rec <-
  recipe(brca1 ~ ., data = df_train) %>%
  update_role(depmap_id, cell_line_display_name, lineage_1, new_role = "ID") %>%
  step_impute_median(all_numeric_predictors()) %>%   # Fix numeric NAs
  step_impute_mode(all_nominal_predictors()) %>%      # Fix categorical NAs
  step_dummy(all_nominal_predictors()) %>%            # Convert factors to dummies
  step_zv(all_predictors()) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors())


lasso_model <-
  linear_reg(
    penalty = tune(),
    mixture = 0.5,
  ) %>%
  set_engine("glmnet")
lasso_wf <-
  workflow() %>%
  add_model(lasso_model) %>%
  add_recipe(lasso_rec)


penalty_grid <- grid_regular(
  penalty(range = c(-3, -0.5)), 
 levels = 50
)



#penalty_grid <- grid_regular(penalty(), levels = 50)
print(penalty_grid, n = 50)
plot(penalty_grid$penalty, main = "Penalty Values to Test", 
     ylab = "Penalty", xlab = "Index")

lasso_tuning_results <- 
  lasso_wf %>% 
  tune_grid(cv_folds,
            grid = penalty_grid,
            control = control_grid(save_pred = TRUE))
lasso_tuning_results
autoplot(lasso_tuning_results) +
  labs(title = "Did we find the sweet spot?")
collect_metrics
tuning_metrics <- lasso_tuning_results %>% 
  collect_metrics()

tuning_metrics

best_penalty <- lasso_tuning_results %>%
  select_best(metric = "rmse")

final_lasso_wf <-
  lasso_wf %>%
  finalize_workflow(best_penalty)

final_fit <- final_lasso_wf %>%
  last_fit(df_split)


vip_plot <-final_fit %>%
  extract_workflow() %>%
  extract_fit_parsnip() %>%
  vip(num_features = 30)

###########################################################################################
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
#write_csv(important_genes, "significant_genes_elastic_net.csv")



# Extract metrics (RMSE and R-squared)
final_metrics <- collect_metrics(final_fit)

# STORE: Save metrics to a CSV
#write_csv(final_metrics, "model_performance_metrics.csv")


# Extract predictions from the test set
test_predictions <- collect_predictions(final_fit)

# STORE: Save predictions to CSV
# This allows you to plot "Predicted vs Actual" in Excel or R later
#write_csv(test_predictions, "test_set_predictions.csv")


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





##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################


# Calculate correlation for the label
cor_val <- cor(test_predictions$brca1, test_predictions$.pred)
r2_val <- cor_val^2

ggplot(test_predictions, aes(x = brca1, y = .pred)) +
  geom_point(alpha = 0.6, size = 3, color = "#2c3e50") +  # Professional Dark Blue
  geom_abline(lty = 2, color = "firebrick", size = 1) +   # Red diagonal line
  annotate("text", x = max(test_predictions$brca1), y = min(test_predictions$.pred), 
           label = paste0("R² = ", round(r2_val, 2)), 
           hjust = 1, vjust = 0, size = 6, fontface = "bold") +
  labs(
    title = "Model Performance: Predicted vs. Actual Dependency",
    subtitle = "Elastic Net Model (Mixture = 0.5)",
    x = "Actual BRCA1 Dependency Score (CRISPR)",
    y = "Predicted Score"
  ) +
  theme_minimal(base_size = 14) # Larger text for slides

# Take top 20 genes
top_genes <- head(important_genes, 20)

ggplot(top_genes, aes(x = reorder(term, estimate), y = estimate, fill = estimate > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = c("firebrick", "steelblue")) + # Red for Neg, Blue for Pos
  labs(
    title = "Top 20 Genetic Predictors of BRCA1 Dependency",
    subtitle = "Positive = Co-dependency | Negative = Compensatory/Resistance",
    x = "Gene",
    y = "Coefficient (Importance)"
  ) +
  theme_light(base_size = 14) +
  theme(panel.grid.major.y = element_blank()) # Clean look

autoplot(lasso_tuning_results) +
  geom_vline(xintercept = select_best(lasso_tuning_results, metric = "rmse")$penalty, 
             lty = 2, color = "red") + # Mark the chosen spot
  labs(
    title = "Hyperparameter Tuning: Finding the Optimal Penalty",
    subtitle = "The red line marks the minimized Root Mean Squared Error (RMSE)",
    x = "Regularization Penalty (Log Scale)"
  ) +
  theme_bw(base_size = 14)


library(yardstick)


library(yardstick)

# Create a binary classification just for plotting
roc_data <- test_predictions %>%
  mutate(
    # Truth: Is the cell line ACTUALLY dependent? (< -1)
    truth_class = factor(ifelse(brca1 < -0.5, "Dependent", "Not_Dependent"), 
                         levels = c("Dependent", "Not_Dependent"))
  )

# Plot ROC
roc_data %>%
  roc_curve(truth = truth_class, .pred) %>%
  autoplot() +
  labs(
    title = "ROC Curve: Distinguishing Dependent Cell Lines",
    subtitle = "Ability to classify 'Essential' vs 'Non-Essential' lines",
    x = "False Positive Rate", 
    y = "True Positive Rate"
  ) +
  theme_minimal(base_size = 14)

ggplot(test_predictions, aes(x = .pred, y = brca1 - .pred)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, lty = 2, color = "red") +
  labs(
    title = "Residual Plot: Where are the errors?",
    x = "Predicted Value",
    y = "Residuals (Actual - Predicted)"
  ) +
  theme_minimal(base_size = 14)


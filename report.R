library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(pROC)

hotels <- read.csv("https://www.louisaslett.com/Courses/MISCADA/hotels.csv")

str(hotels)

summary(hotels)

colSums(is.na(hotels))

sapply(hotels, class)

hotels[hotels == "NULL"] <- NA  
hotels$children[is.na(hotels$children)] <- 0 

hotels_clean <- hotels %>%
  select(-c(reservation_status_date, company, agent))

hotels_clean$is_canceled <- as.factor(hotels_clean$is_canceled)

categorical_vars <- c("hotel", "meal", "market_segment", "distribution_channel",
                      "reserved_room_type", "assigned_room_type", "deposit_type",
                      "customer_type", "reservation_status")

hotels_clean[categorical_vars] <- lapply(hotels_clean[categorical_vars], as.factor)

hotels_clean <- hotels_clean %>%
  mutate(across(where(is.integer), as.numeric)) %>% 
  mutate(across(where(is.logical), as.factor))

print("Check for missing values：")
print(colSums(is.na(hotels_clean)))

# Save cleaned data
write.csv(hotels_clean, "hotels_cleaned.csv", row.names = FALSE)

print("Data cleaning complete, saved as hotels_cleaned.csv")

library(ggplot2)

# Visualize order cancellations
ggsave("cancellation_distribution.png",
       ggplot(hotels_clean, aes(x = is_canceled)) +
         geom_bar(fill = "blue") +
         labs(title = "Booking Cancellation Distribution", 
              x = "Canceled (0=Not Canceled, 1=Canceled)", 
              y = "Count") +
         theme_minimal(), 
       width = 6, height = 4, dpi = 300)

# Visualize relationship between lead_time and cancellation rate
ggsave("lead_time_vs_cancellation.png",
       ggplot(hotels_clean, aes(x = lead_time, fill = is_canceled)) +
         geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
         labs(title = "Lead Time vs Cancellation Rate", x = "Lead Time (days)", y = "Count") +
         theme_minimal(), 
       width = 6, height = 4, dpi = 300)

# Visualize relationship between Market Segment and cancellation rate
ggsave("market_segment_vs_cancellation.png",
       ggplot(hotels_clean, aes(x = market_segment, fill = is_canceled)) +
         geom_bar(position = "fill") +
         labs(title = "Cancellation Rate by Market Segment", x = "Market Segment", y = "Proportion of Cancellations") +
         theme_minimal() +
         theme(axis.text.x = element_text(angle = 45, hjust = 1)), 
       width = 6, height = 4, dpi = 300)

# Visualize relationship between Customer Type and cancellation rate
ggsave("customer_type_vs_cancellation.png",
       ggplot(hotels_clean, aes(x = customer_type, fill = is_canceled)) +
         geom_bar(position = "fill") +
         labs(title = "Cancellation Rate by Customer Type", x = "Customer Type", y = "Proportion of Cancellations") +
         theme_minimal(), 
       width = 6, height = 4, dpi = 300)

# Visualize relationship between Deposit Type and cancellation rate
ggsave("deposit_type_vs_cancellation.png",
       ggplot(hotels_clean, aes(x = deposit_type, fill = is_canceled)) +
         geom_bar(position = "fill") +
         labs(title = "Cancellation Rate by Deposit Type", x = "Deposit Type", y = "Proportion of Cancellations") +
         theme_minimal(), 
       width = 6, height = 4, dpi = 300)

# Check correlation between variables
library(corrplot)
numeric_vars <- hotels_clean %>%
  select(where(is.numeric))

correlation_matrix <- cor(numeric_vars)
png("correlation_matrix.png", width = 800, height = 600)
corrplot(correlation_matrix, method = "circle")
dev.off()

print("Data visualization complete, all images saved。")

hotels_cleaned <- read.csv("hotels_cleaned.csv")

hotels_cleaned$is_canceled <- as.factor(hotels_cleaned$is_canceled)


# Split 70% train / 30% test
set.seed(42)
trainIndex <- createDataPartition(hotels_cleaned$is_canceled, p = 0.7, list = FALSE)
train_data <- hotels_cleaned[trainIndex, ]
test_data <- hotels_cleaned[-trainIndex, ]

fill_mode <- function(x) {
  x[is.na(x)] <- names(which.max(table(x, useNA = "no")))
  return(x)
}



train_data$country <- fill_mode(train_data$country)
test_data$country <- fill_mode(test_data$country)

factor_vars <- names(Filter(is.factor, train_data))

for (var in factor_vars) {
  test_data[[var]] <- factor(test_data[[var]], levels = levels(train_data[[var]]))
}


# Handle data imbalance
train_data <- downSample(x = train_data[, -which(names(train_data) == "is_canceled")], 
                         y = train_data$is_canceled)
colnames(train_data)[ncol(train_data)] <- "is_canceled"

# Train random forest model
set.seed(123)
rf_model <- randomForest(is_canceled ~ ., data = train_data, 
                         ntree = 100, 
                         mtry = sqrt(ncol(train_data) - 1),  
                         maxnodes = 10, 
                         importance = TRUE)

# Predict test set
rf_pred <- predict(rf_model, test_data, type = "response")

# Compute confusion matrix & accuracy
conf_matrix <- table(test_data$is_canceled, rf_pred)
print(conf_matrix)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Random Forest Model accuracy：", round(accuracy, 4), "\n")


dummies <- dummyVars(" ~ .", data = hotels_cleaned, fullRank = TRUE)

train_matrix <- predict(dummies, newdata = train_data)
test_matrix <- predict(dummies, newdata = test_data)

train_cols <- colnames(train_matrix)
test_cols <- colnames(test_matrix)

# Find and fill missing columns in test_matrix with 0
missing_cols <- setdiff(train_cols, test_cols)
for (col in missing_cols) {
  test_matrix <- cbind(test_matrix, rep(0, nrow(test_matrix)))
  colnames(test_matrix)[ncol(test_matrix)] <- col
}

# Remove extra columns from test_matrix
extra_cols <- setdiff(test_cols, train_cols)
if (length(extra_cols) > 0) {
  test_matrix <- test_matrix[, !(colnames(test_matrix) %in% extra_cols)]
}

# Ensure consistent column order
test_matrix <- test_matrix[, train_cols]

# Convert data to xgb.DMatrix
train_matrix <- as.matrix(train_matrix)
test_matrix <- as.matrix(test_matrix)

#  delet characteristics of data breaches
train_matrix <- train_matrix[, !(colnames(train_matrix) %in% "is_canceled.1")]
test_matrix <- test_matrix[, !(colnames(test_matrix) %in% "is_canceled.1")]

# delet reservation_status 
train_matrix <- train_matrix[, !(colnames(train_matrix) %in% c("reservation_statusCheck-Out", "reservation_statusNo-Show"))]
test_matrix <- test_matrix[, !(colnames(test_matrix) %in% c("reservation_statusCheck-Out", "reservation_statusNo-Show"))]


dtrain <- xgb.DMatrix(data = train_matrix, label = as.numeric(train_data$is_canceled) - 1)
dtest <- xgb.DMatrix(data = test_matrix, label = as.numeric(test_data$is_canceled) - 1)

# Train XGBoost model
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgboost(
  data = dtrain,
  params = xgb_params,
  nrounds = 100,
  verbose = 1
)

xgb_pred_prob <- predict(xgb_model, dtest)  
xgb_pred_class <- ifelse(xgb_pred_prob > 0.5, 1, 0) 
xgb_pred_class <- factor(xgb_pred_class, levels = c(0, 1))

conf_matrix_xgb <- table(test_data$is_canceled, xgb_pred_class)
accuracy_xgb <- sum(diag(conf_matrix_xgb)) / sum(conf_matrix_xgb)

print(conf_matrix_xgb)
cat("GBoost model accuracy:", round(accuracy_xgb, 4), "\n")

conf_matrix_xgb <- confusionMatrix(as.factor(xgb_pred_class), as.factor(test_data$is_canceled))

print(conf_matrix_xgb)

roc_rf <- roc(as.numeric(test_data$is_canceled) - 1, as.numeric(rf_pred))
roc_xgb <- roc(as.numeric(test_data$is_canceled) - 1, xgb_pred_prob)


roc_rf_df <- data.frame(TPR = rev(roc_rf$sensitivities), 
                        FPR = rev(1 - roc_rf$specificities), 
                        Model = "Random Forest")

roc_xgb_df <- data.frame(TPR = rev(roc_xgb$sensitivities), 
                         FPR = rev(1 - roc_xgb$specificities), 
                         Model = "XGBoost")

roc_data <- rbind(roc_rf_df, roc_xgb_df)

# ROC
roc_plot <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.2) +  
  geom_abline(linetype = "dashed", color = "gray") +
  theme_minimal() +
  labs(title = "ROC Comparison", x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  scale_color_manual(values = c("blue", "red"))  

print(roc_plot)

ggsave("roc_comparison.png", plot = roc_plot, width = 6, height = 4, dpi = 300)


auc_rf <- auc(roc_rf)
auc_xgb <- auc(roc_xgb)

cat("Random Forest AUC Score:", round(auc_rf, 4), "\n")
cat("XGBoost AUC Score:", round(auc_xgb, 4), "\n")

ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1) +
  geom_abline(linetype = "dashed") +
  theme_minimal() +
  labs(title = "ROC Comparison", x = "1 - Specificity", y = "Sensitivity") +
  scale_color_manual(values = c("blue", "red"))

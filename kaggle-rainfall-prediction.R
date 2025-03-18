# Rainfall prediction - Kaggle
# Andrex Ibiza, MBA
# 2025-03-16

# # Load libraries
# library(caret) # For data partitioning and evaluation
# library(dplyr) # For data manipulation
# library(ggplot2) # For visualization
# library(ggthemes) # For ggplot2 themes
# library(pROC) # For ROC analysis
# library(randomForest) # For Random Forest model
# library(xgboost) # For XGBoost model
# 
# # Ensure reproducibility
# set.seed(666)

# # Load data
# train <- read.csv("data/train.csv")
# test <- read.csv("data/test.csv")

# head(train)
# head(test)

# All of the data are continuous numeric values (except the prediction), 
# so we can move quickly into the model building process.

###                 MODEL 1: LOGISTIC REGRESSION                 ###

# dim(train)

# # `id` and `day` columns are not useful for the model, so we can remove them.
# data_model <- train[, c("pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
#                        "humidity", "cloud", "sunshine", "winddirection", "windspeed", "rainfall")]
# 
# # Split `train` data into training and testing sets (80/20 split)
# trainIndex <- createDataPartition(data_model$rainfall, p = 0.8, list = FALSE)
# trainData <- data_model[trainIndex, ]
# testData <- data_model[-trainIndex, ]

# # Fit logistic regression model using `glm` function
# lr_model <- glm(
#   rainfall ~ ., 
#   data = trainData, 
#   family = "binomial")
# 
# # Generate predictions on the test set
# # We convert probabilities to binary predictions with a cutoff of 0.5
# testData$pred <- ifelse(predict(lr_model, newdata = testData, type = "response") > 0.5, 1, 0)
# 
# # Evaluate model accuracy with a confusion matrix and ROC curve
# confusionMatrix(table(testData$rainfall, testData$pred))
# roc_obj <- roc(testData$rainfall, testData$pred)
# 
# # Visualize model performance: ROC curve and AUC
# ggroc <- ggplot(data = data.frame(
#   fpr = 1 - roc_obj$specificities,
#   tpr = roc_obj$sensitivities),
#   aes(x = fpr, y = tpr)) +
#   geom_line(color = "blue", size = 1) +
#   geom_abline(linetype = "dashed", color = "red") +
#   labs(title = "ROC Curve",
#        x = "False Positive Rate",
#        y = "True Positive Rate") +
#   theme_wsj()
# 
# print(ggroc)
# 
# auc(roc_obj)
# # Area under the curve: 0.779
# 
# 
# ###                 MODEL 2: RANDOM FOREST                 ###
# # Random Forest is a popular ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy.
# 
# # Convert the target variable to a factor so that Random Forest performs classification.
# trainData$rainfall <- as.factor(trainData$rainfall)
# testData$rainfall <- as.factor(testData$rainfall)
# 
# # Fit Random Forest model using `randomForest` function
# rf_model <- randomForest(
#   rainfall ~ ., 
#   data = trainData, 
#   ntree = 500)
# 
# # Generate predictions on the test set
# testData$pred_rf <- predict(rf_model, newdata = testData)
# 
# # To perform ROC analysis, we need predicted probabilities.
# # Assume that the positive class (rain occurred) is labeled as "1".
# # Check levels(testData$rainfall) to confirm your label order.
# testData$pred_prob_rf <- predict(rf_model, newdata = testData, type = "prob")[, "1"]
# 
# # Evaluate model accuracy with a confusion matrix.
# cm <- confusionMatrix(
#   reference = testData$rainfall, 
#   data = testData$pred_rf
# )
# print(cm)
# 
# # Evaluate model performance with the ROC curve and AUC.
# # Note: roc expects a numeric response; if your response levels are "0" and "1",
# # we need to convert them appropriately.
# actual_numeric <- as.numeric(as.character(testData$rainfall))
# roc_obj_rf <- roc(actual_numeric, testData$pred_prob_rf)
# 
# # Visualize model performance: ROC curve and AUC.
# roc_data <- data.frame(
#   fpr = 1 - roc_obj_rf$specificities,
#   tpr = roc_obj_rf$sensitivities
# )
# 
# ggroc_rf <- ggplot(data = roc_data, aes(x = fpr, y = tpr)) +
#   geom_line(color = "blue", size = 1) +
#   geom_abline(linetype = "dashed", color = "red") +
#   labs(title = "ROC Curve for Random Forest Model",
#        x = "False Positive Rate",
#        y = "True Positive Rate") +
#   theme_wsj()
# 
# print(ggroc_rf)
# 
# # Print the AUC value.
# auc_value <- auc(roc_obj_rf)
# cat("AUC:", auc_value, "\n")
# 
# #                         MODEL 3. XGBOOST
# # Recode the target variable so that the positive class is "Yes" and the negative is "No"
# # Here we assume that original rainfall values are "0" and "1".
# trainData$rainfall <- factor(trainData$rainfall, levels = c("0", "1"), labels = c("No", "Yes"))
# testData$rainfall  <- factor(testData$rainfall,  levels = c("0", "1"), labels = c("No", "Yes"))
# 
# # Set up cross-validation controls
# fitControl <- trainControl(
#   method = "cv",                # k-fold cross-validation
#   number = 5,                   # number of folds
#   verboseIter = TRUE,           # output training progress
#   classProbs = TRUE,            # needed for twoClassSummary
#   summaryFunction = twoClassSummary,
#   allowParallel = TRUE
# )
# 
# # Define a grid of hyperparameters to try.
# # Adjust grid values as needed.
# xgbGrid <- expand.grid(
#   nrounds = c(50, 100),         # number of boosting rounds
#   max_depth = c(3, 6),          # maximum depth of trees
#   eta = c(0.01, 0.1),           # learning rate
#   gamma = c(0, 1),              # minimum loss reduction
#   colsample_bytree = c(0.6, 0.8), # subsample ratio of columns
#   min_child_weight = c(1, 3),   # minimum sum of instance weight
#   subsample = c(0.6, 0.8)        # subsample ratio of the training instances
# )
# 
# # Train the xgboost model using caret.
# # Make sure that your training data contains only predictors and the target.
# # Adjust the formula if necessary. In this case, all columns except "rainfall" are used as predictors.
# xgb_tuned <- train(
#   rainfall ~ ., 
#   data = trainData,
#   method = "xgbTree",
#   trControl = fitControl,
#   tuneGrid = xgbGrid,
#   metric = "ROC"  # optimize based on ROC AUC
# )
# 
# # Display the best tuned hyperparameters
# print(xgb_tuned$bestTune)
# 
# # Generate predicted probabilities on the test set using the tuned model.
# # The probabilities for the positive class ("Yes") are extracted.
# pred_prob <- predict(xgb_tuned, newdata = testData, type = "prob")[, "Yes"]
# 
# # Convert probabilities to class predictions with a threshold of 0.5.
# pred_class <- ifelse(pred_prob > 0.5, "Yes", "No")
# pred_class <- factor(pred_class, levels = c("No", "Yes"))
# 
# # Evaluate the model with a confusion matrix.
# cm <- confusionMatrix(
#   reference = testData$rainfall, 
#   data = pred_class
# )
# print(cm)
# 
# # For ROC analysis, convert the actual responses to a binary numeric vector.
# # Here, "Yes" = 1 and "No" = 0.
# actual_numeric <- ifelse(testData$rainfall == "Yes", 1, 0)
# roc_obj <- roc(actual_numeric, pred_prob)
# 
# # Visualize the ROC curve.
# roc_data <- data.frame(
#   fpr = 1 - roc_obj$specificities,
#   tpr = roc_obj$sensitivities
# )
# 
# ggroc_xgb <- ggplot(data = roc_data, aes(x = fpr, y = tpr)) +
#   geom_line(color = "blue", size = 1) +
#   geom_abline(linetype = "dashed", color = "red") +
#   labs(title = "ROC Curve for Tuned xgboost Model",
#        x = "False Positive Rate",
#        y = "True Positive Rate") +
#   theme_wsj()
# 
# print(ggroc_xgb)
# 
# # Print the AUC value.
# auc_value <- auc(roc_obj)
# cat("AUC:", auc_value, "\n")
# 
# # AUC: 0.871096


# #                         MODEL 4.1 NEURAL NETWORK
# 
# library(nnet)
# 
# # Recode the target variable so the positive class is "Yes" and the negative is "No"
# # Assume that original rainfall values are "0" and "1".
# trainData$rainfall <- factor(trainData$rainfall, levels = c("0", "1"), labels = c("No", "Yes"))
# testData$rainfall  <- factor(testData$rainfall,  levels = c("0", "1"), labels = c("No", "Yes"))
# 
# library(naniar)
# n_miss(trainData$rainfall)
# n_miss(testData$rainfall)
# 
# # Set up cross-validation controls
# fitControl <- trainControl(
#   method = "cv",                # k-fold cross-validation
#   number = 5,                   # number of folds
#   verboseIter = FALSE,          # do not output detailed progress
#   classProbs = TRUE,            # needed for twoClassSummary to compute ROC
#   summaryFunction = twoClassSummary,
#   allowParallel = TRUE
# )
# 
# # Define a grid of hyperparameters for neural network tuning.
# # 'size' is the number of hidden units, 'decay' is the regularization parameter.
# nnGrid <- expand.grid(
#   size = c(5, 10, 15),     # number of hidden units
#   decay = c(0.0, 0.01, 0.1) # regularization parameter to avoid overfitting
# )
# 
# # Train the neural network model using caret's train() function.
# # The preProcess argument imputes missing values using the median before training.
# nn_tuned <- train(
#   rainfall ~ ., 
#   data = trainData,
#   method = "nnet",
#   trControl = fitControl,
#   tuneGrid = nnGrid,
#   metric = "ROC",           # optimize based on ROC AUC
#   trace = FALSE,            # disable printing of training iterations
#   maxit = 200               # maximum iterations (adjust as necessary)
#   )
# 
# # Display the best tuned hyperparameters
# print(nn_tuned$bestTune)
# 
# # Generate predicted probabilities on the test set using the tuned model.
# # Here, we also preProcess the testData so that missing values are handled in the same way.
# pred_prob_nn <- predict(nn_tuned, newdata = testData, type = "prob")[, "Yes"]
# 
# # Convert predicted probabilities to class predictions using 0.5 as the threshold.
# pred_class_nn <- ifelse(pred_prob_nn > 0.5, "Yes", "No")
# pred_class_nn <- factor(pred_class_nn, levels = c("No", "Yes"))
# 
# # Evaluate the model with a confusion matrix.
# cm_nn <- confusionMatrix(
#   reference = testData$rainfall, 
#   data = pred_class_nn
# )
# print(cm_nn)
# 
# # ROC analysis using numeric values directly.
# roc_obj_nn <- roc(testData$rainfall, pred_prob_nn)
# 
# # Visualize the ROC curve.
# roc_data <- data.frame(
#   fpr = 1 - roc_obj_nn$specificities,
#   tpr = roc_obj_nn$sensitivities
# )
# 
# ggroc_nn <- ggplot(data = roc_data, aes(x = fpr, y = tpr)) +
#   geom_line(color = "blue", size = 1) +
#   geom_abline(linetype = "dashed", color = "red") +
#   labs(title = "ROC Curve for Tuned Neural Network",
#        x = "False Positive Rate",
#        y = "True Positive Rate") +
#   theme_clean()
# 
# print(ggroc_nn)
# 
# # Print the AUC value.
# auc_value <- auc(roc_obj_nn)
# cat("AUC:", auc_value, "\n")
# 
# # any_na(test)
# # miss_case_summary(test)
# 
# # Impute missing values in the test dataframe using median imputation
# pp <- preProcess(test, method = "medianImpute")
# test_imputed <- predict(pp, test)
# 
# # Generate predicted probabilities on the original test dataframe using the tuned model.
# pred_prob_test <- predict(nn_tuned, newdata = test_imputed, type = "prob")[, "Yes"]
# 
# # Convert predicted probabilities to class predictions using 0.5 as the threshold.
# test$rainfall <- ifelse(pred_prob_test > 0.5, 1, 0)
# 
# # Create a new dataframe for submission with the updated test dataframe.
# submission <- test %>% 
#   select(id, rainfall)
# 
# # Write the submission dataframe to a CSV file.
# # write.csv(submission, "submission.csv", row.names = FALSE)
# 
# # Print the first few rows of the submission dataframe to verify.
# head(submission)

# #                         MODEL 4.2 NEURAL NETWORK
# 
# # expanded hyperparameter tuning grid
# library(nnet)
# library(naniar)
# 
# # Ensure reproducibility
# set.seed(666)
# 
# # Load data
# train <- read.csv("data/train.csv")
# test <- read.csv("data/test.csv")
# 
# # `id` and `day` columns are not useful for the model, so we can remove them.
# data_model <- train[, c("pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
#                         "humidity", "cloud", "sunshine", "winddirection", "windspeed", "rainfall")]
# 
# # Split `train` data into training and testing sets (80/20 split)
# trainIndex <- createDataPartition(data_model$rainfall, p = 0.8, list = FALSE)
# trainData <- data_model[trainIndex, ]
# testData <- data_model[-trainIndex, ]
# 
# # Recode the target variable so the positive class is "Yes" and the negative is "No"
# # Assume that original rainfall values are "0" and "1".
# trainData$rainfall <- factor(trainData$rainfall, levels = c("0", "1"), labels = c("No", "Yes"))
# testData$rainfall  <- factor(testData$rainfall,  levels = c("0", "1"), labels = c("No", "Yes"))
# 
# n_miss(trainData$rainfall)
# n_miss(testData$rainfall)
# 
# # Set up cross-validation controls
# fitControl <- trainControl(
#   method = "cv",                # k-fold cross-validation
#   number = 5,                   # number of folds
#   verboseIter = FALSE,          # do not output detailed progress
#   classProbs = TRUE,            # needed for twoClassSummary to compute ROC
#   summaryFunction = twoClassSummary,
#   allowParallel = TRUE
# )
# 
# # Define a grid of hyperparameters for neural network tuning.
# # 'size' is the number of hidden units, 'decay' is the regularization parameter.
# nnGrid <- expand.grid(
#   size = c(5, 10, 15, 20, 25),         # number of hidden units
#   decay = c(0, 0.001, 0.01, 0.1, 1)     # regularization parameter to avoid overfitting
# )
# 
# # Train the neural network model using caret's train() function.
# # The preProcess argument imputes missing values using the median before training.
# nn_tuned <- train(
#   rainfall ~ ., 
#   data = trainData,
#   method = "nnet",
#   trControl = fitControl,   # best tuned size = 10, decay = 1
#   tuneGrid = nnGrid,
#   metric = "ROC",           # optimize based on ROC AUC
#   trace = FALSE,            # disable printing of training iterations
#   maxit = 200               # maximum iterations (adjust as necessary)
# )
# 
# # Display the best tuned hyperparameters
# print(nn_tuned$bestTune)
# 
# # Generate predicted probabilities on the test set using the tuned model.
# # Here, we also preProcess the testData so that missing values are handled in the same way.
# pred_prob_nn <- predict(nn_tuned, newdata = testData, type = "prob")[, "Yes"]
# 
# # Convert predicted probabilities to class predictions using 0.5 as the threshold.
# pred_class_nn <- ifelse(pred_prob_nn > 0.5, "Yes", "No")
# pred_class_nn <- factor(pred_class_nn, levels = c("No", "Yes"))
# 
# # Evaluate the model with a confusion matrix.
# cm_nn <- confusionMatrix(
#   reference = testData$rainfall, 
#   data = pred_class_nn
# )
# print(cm_nn)
# 
# # ROC analysis using numeric values directly.
# roc_obj_nn <- roc(testData$rainfall, pred_prob_nn)
# 
# # Visualize the ROC curve.
# roc_data <- data.frame(
#   fpr = 1 - roc_obj_nn$specificities,
#   tpr = roc_obj_nn$sensitivities
# )
# 
# ggroc_nn <- ggplot(data = roc_data, aes(x = fpr, y = tpr)) +
#   geom_line(color = "blue", size = 1) +
#   geom_abline(linetype = "dashed", color = "red") +
#   labs(title = "ROC Curve for Tuned Neural Network",
#        x = "False Positive Rate",
#        y = "True Positive Rate") +
#   theme_clean()
# 
# print(ggroc_nn)
# 
# # Print the AUC value.
# auc_value <- auc(roc_obj_nn)
# cat("AUC:", auc_value, "\n")
# 
# # any_na(test)
# # miss_case_summary(test)
# 
# # Impute missing values in the test dataframe using median imputation
# pp <- preProcess(test, method = "medianImpute")
# test_imputed <- predict(pp, test)
# 
# # Generate predicted probabilities on the original test dataframe using the tuned model.
# pred_prob_test <- predict(nn_tuned, newdata = test_imputed, type = "prob")[, "Yes"]
# 
# # Convert predicted probabilities to class predictions using 0.5 as the threshold.
# test$rainfall <- ifelse(pred_prob_test > 0.5, 1, 0)
# 
# # Create a new dataframe for submission with the updated test dataframe.
# submission <- test %>% 
#   select(id, rainfall)
# 
# # Write the submission dataframe to a CSV file.
# write.csv(submission, "submission.csv", row.names = FALSE)
# 
# # Print the first few rows of the submission dataframe to verify.
# head(submission)

# #                         MODEL 5. NEW FEATURES + NNET
# 
# # Load libraries
# library(caret) # For data partitioning and evaluation
# library(dplyr) # For data manipulation
# library(ggplot2) # For visualization
# library(ggthemes) # For ggplot2 themes
# library(naniar) # For missing data visualization
# library(nnet) # For neural network model
# library(pROC) # For ROC analysis
# library(randomForest) # For Random Forest model
# library(xgboost) # For XGBoost model
# 
# # Ensure reproducibility
# set.seed(666)
# 
# # Load data
# train <- read.csv("data/train.csv")
# test <- read.csv("data/test.csv")
# 
# # ---------- New Feature Engineering --------------
# # Create new features based on the existing columns.
# # Assuming train and test data frames are already loaded
# 
# # 1. Temperature Range (maxtemp - mintemp)
# train$temp_range <- train$maxtemp - train$mintemp
# test$temp_range <- test$maxtemp - test$mintemp
# 
# # 2. Dewpoint Gap (temparature - dewpoint)
# train$temp_dew_diff <- train$temparature - train$dewpoint
# test$temp_dew_diff <- test$temparature - test$dewpoint
# 
# # 3. Wind Vector Components
# # Convert wind direction from degrees to radians and calculate x and y components
# train$wind_rad <- train$winddirection * pi / 180
# train$wind_x <- train$windspeed * cos(train$wind_rad)
# train$wind_y <- train$windspeed * sin(train$wind_rad)
# 
# test$wind_rad <- test$winddirection * pi / 180
# test$wind_x <- test$windspeed * cos(test$wind_rad)
# test$wind_y <- test$windspeed * sin(test$wind_rad)
# 
# # 4. Pressure Change (difference from previous observation; first observation set to NA)
# train$pressure_change <- c(NA, diff(train$pressure))
# test$pressure_change <- c(NA, diff(test$pressure))
# # Optionally replace NA with 0:
# train$pressure_change[is.na(train$pressure_change)] <- 0
# test$pressure_change[is.na(test$pressure_change)] <- 0
# 
# # 5. Sunshine Features
# # 5a. Log-transformed sunshine (adding 1 to avoid log(0))
# train$sunshine_log <- log(train$sunshine + 1)
# test$sunshine_log <- log(test$sunshine + 1)
# 
# # 5b. Sunshine ratio compared to cloud+sunshine (safeguard division by zero)
# train$sun_frac <- with(train, ifelse((cloud + sunshine) > 0, sunshine / (cloud + sunshine), 0))
# test$sun_frac <- with(test, ifelse((cloud + sunshine) > 0, sunshine / (cloud + sunshine), 0))
# 
# # Print a summary of the new features to confirm
# summary(train[, c("temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")])
# summary(test[, c("temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")])
# 
# 
# # `id` and `day` columns are not useful for the model, so we can remove them.
# data_model <- train[, c("pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
#                         "humidity", "cloud", "sunshine", "winddirection", "windspeed", "rainfall",
#                         "temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")]
# 
# # Split `train` data into training and testing sets (80/20 split)
# trainIndex <- createDataPartition(data_model$rainfall, p = 0.8, list = FALSE)
# trainData <- data_model[trainIndex, ]
# testData <- data_model[-trainIndex, ]
# 
# # Recode the target variable so the positive class is "Yes" and the negative is "No"
# # Assume that original rainfall values are "0" and "1".
# trainData$rainfall <- factor(trainData$rainfall, levels = c("0", "1"), labels = c("No", "Yes"))
# testData$rainfall  <- factor(testData$rainfall,  levels = c("0", "1"), labels = c("No", "Yes"))
# 
# n_miss(trainData$rainfall)
# n_miss(testData$rainfall)
# 
# # Set up cross-validation controls
# fitControl <- trainControl(
#   method = "cv",                # k-fold cross-validation
#   number = 5,                   # number of folds
#   verboseIter = FALSE,          # do not output detailed progress
#   classProbs = TRUE,            # needed for twoClassSummary to compute ROC
#   summaryFunction = twoClassSummary,
#   allowParallel = TRUE
# )
# 
# # Define a grid of hyperparameters for neural network tuning.
# # 'size' is the number of hidden units, 'decay' is the regularization parameter.
# nnGrid <- expand.grid(
#   size = c(5, 10, 15, 20, 25),         # number of hidden units
#   decay = c(0, 0.001, 0.01, 0.1, 1)     # regularization parameter to avoid overfitting
# )
# 
# # Train the neural network model using caret's train() function.
# # The preProcess argument imputes missing values using the median before training.
# nn_tuned <- train(
#   rainfall ~ .,
#   data = trainData,
#   method = "nnet",
#   trControl = fitControl,   # best tuned size = 10, decay = 1
#   tuneGrid = nnGrid,
#   metric = "ROC",           # optimize based on ROC AUC
#   trace = FALSE,            # disable printing of training iterations
#   maxit = 200               # maximum iterations (adjust as necessary)
# )
# 
# # Display the best tuned hyperparameters
# print(nn_tuned$bestTune)
# 
# # Generate predicted probabilities on the test set using the tuned model.
# # Here, we also preProcess the testData so that missing values are handled in the same way.
# pred_prob_nn <- predict(nn_tuned, newdata = testData, type = "prob")[, "Yes"]
# 
# # Convert predicted probabilities to class predictions using 0.5 as the threshold.
# pred_class_nn <- ifelse(pred_prob_nn > 0.5, "Yes", "No")
# pred_class_nn <- factor(pred_class_nn, levels = c("No", "Yes"))
# 
# # Evaluate the model with a confusion matrix.
# cm_nn <- confusionMatrix(
#   reference = testData$rainfall,
#   data = pred_class_nn
# )
# print(cm_nn)
# 
# # ROC analysis using numeric values directly.
# roc_obj_nn <- roc(testData$rainfall, pred_prob_nn)
# 
# # Visualize the ROC curve.
# roc_data <- data.frame(
#   fpr = 1 - roc_obj_nn$specificities,
#   tpr = roc_obj_nn$sensitivities
# )
# 
# ggroc_nn <- ggplot(data = roc_data, aes(x = fpr, y = tpr)) +
#   geom_line(color = "blue", size = 1) +
#   geom_abline(linetype = "dashed", color = "red") +
#   labs(title = "ROC Curve for Tuned Neural Network",
#        x = "False Positive Rate",
#        y = "True Positive Rate") +
#   theme_clean()
# 
# print(ggroc_nn)
# 
# # Print the AUC value.
# auc_value <- auc(roc_obj_nn)
# cat("AUC:", auc_value, "\n")
# 
# # any_na(test)
# # miss_case_summary(test)
# 
# # Impute missing values in the test dataframe using median imputation
# pp <- preProcess(test, method = "medianImpute")
# test_imputed <- predict(pp, test)
# 
# # Generate predicted probabilities on the original test dataframe using the tuned model.
# pred_prob_test <- predict(nn_tuned, newdata = test_imputed, type = "prob")[, "Yes"]
# 
# # Convert predicted probabilities to class predictions using 0.5 as the threshold.
# test$rainfall <- ifelse(pred_prob_test > 0.5, 1, 0)
# 
# # Create a new dataframe for submission with the updated test dataframe.
# submission <- test %>%
#   select(id, rainfall)
# 
# # Write the submission dataframe to a CSV file.
# write.csv(submission, "submission.csv", row.names = FALSE)
# 
# # Print the first few rows of the submission dataframe to verify.
# head(submission)

# -------- Try new features with XGBoost ------------
# # Load libraries
# library(caret)         # For data partitioning and evaluation
# library(dplyr)         # For data manipulation
# library(ggplot2)       # For visualization
# library(ggthemes)      # For ggplot2 themes
# library(naniar)        # For missing data visualization
# library(nnet)          # For neural network model
# library(pROC)          # For ROC analysis
# library(randomForest)  # For Random Forest model
# library(xgboost)       # For XGBoost model
# 
# # Ensure reproducibility
# set.seed(666)
# 
# # Load data
# train <- read.csv("data/train.csv")
# test <- read.csv("data/test.csv")
# 
# # ---------- New Feature Engineering --------------
# # 1. Temperature Range (maxtemp - mintemp)
# train$temp_range <- train$maxtemp - train$mintemp
# test$temp_range <- test$maxtemp - test$mintemp
# 
# # 2. Dewpoint Gap (temparature - dewpoint)
# train$temp_dew_diff <- train$temparature - train$dewpoint
# test$temp_dew_diff <- test$temparature - test$dewpoint
# 
# # 3. Wind Vector Components
# train$wind_rad <- train$winddirection * pi / 180
# train$wind_x <- train$windspeed * cos(train$wind_rad)
# train$wind_y <- train$windspeed * sin(train$wind_rad)
# 
# test$wind_rad <- test$winddirection * pi / 180
# test$wind_x <- test$windspeed * cos(test$wind_rad)
# test$wind_y <- test$windspeed * sin(test$wind_rad)
# 
# # 4. Pressure Change (difference from previous observation; first observation set to NA)
# train$pressure_change <- c(NA, diff(train$pressure))
# test$pressure_change <- c(NA, diff(test$pressure))
# train$pressure_change[is.na(train$pressure_change)] <- 0
# test$pressure_change[is.na(test$pressure_change)] <- 0
# 
# # 5. Sunshine Features
# train$sunshine_log <- log(train$sunshine + 1)   # Log-transformed sunshine (adding 1 avoids log(0))
# test$sunshine_log <- log(test$sunshine + 1)
# 
# train$sun_frac <- with(train, ifelse((cloud + sunshine) > 0, sunshine / (cloud + sunshine), 0))
# test$sun_frac <- with(test, ifelse((cloud + sunshine) > 0, sunshine / (cloud + sunshine), 0))
# 
# # Print a summary of the new features to confirm
# summary(train[, c("temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")])
# summary(test[, c("temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")])
# 
# # Remove 'id' and 'day' columns (if present) and create the modeling dataframe.
# data_model <- train[, c("pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
#                         "humidity", "cloud", "sunshine", "winddirection", "windspeed", "rainfall",
#                         "temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")]
# 
# # Split data_model into training and testing sets (80/20 split)
# trainIndex <- createDataPartition(data_model$rainfall, p = 0.8, list = FALSE)
# trainData <- data_model[trainIndex, ]
# testData <- data_model[-trainIndex, ]
# 
# # Recode the target variable so that "Yes" indicates rainfall and "No" no rainfall.
# trainData$rainfall <- factor(trainData$rainfall, levels = c("0", "1"), labels = c("No", "Yes"))
# testData$rainfall  <- factor(testData$rainfall, levels = c("0", "1"), labels = c("No", "Yes"))
# 
# n_miss(trainData$rainfall)
# n_miss(testData$rainfall)
# 
# # ---------------- Principal Component Analysis (PCA) ----------------
# # Apply PCA on the predictor variables in the training data (excluding the target 'rainfall')
# # It's important to scale/center the data for PCA
# 
# # Select only numeric predictor columns (exclude the target)
# predictor_vars <- trainData[, !names(trainData) %in% c("rainfall")]
# 
# # Perform PCA with centering and scaling
# pca_res <- prcomp(predictor_vars, center = TRUE, scale. = TRUE)
# 
# # Create a dataframe that includes the first two principal components and the target
# pca_df <- data.frame(PC1 = pca_res$x[,1],
#                      PC2 = pca_res$x[,2],
#                      rainfall = trainData$rainfall)
# 
# # Visualize the first two principal components colored by rainfall (target)
# ggplot(pca_df, aes(x = PC1, y = PC2, color = rainfall)) +
#   geom_point(alpha = 0.7, size = 2) +
#   labs(title = "PCA: PC1 vs PC2",
#        x = "Principal Component 1",
#        y = "Principal Component 2") +
#   theme_minimal()
# 
# # Create a scree plot to show the proportion of variance explained by each principal component
# var_explained <- (pca_res$sdev)^2 / sum((pca_res$sdev)^2)
# var_df <- data.frame(PC = 1:length(var_explained), Variance = var_explained)
# 
# ggplot(var_df, aes(x = PC, y = Variance)) +
#   geom_bar(stat = "identity", fill = "steelblue") +
#   labs(title = "Scree Plot",
#        x = "Principal Component",
#        y = "Proportion of Variance Explained") +
#   theme_minimal()
# 
# # ---------------- Set up Cross-Validation Controls ----------------
# fitControl <- trainControl(
#   method = "cv",                # k-fold cross-validation
#   number = 5,                   # 5-fold cross-validation
#   verboseIter = TRUE,           # output training progress
#   classProbs = TRUE,            # required for twoClassSummary
#   summaryFunction = twoClassSummary,
#   allowParallel = TRUE
# )
# 
# # ---------------- Define XGBoost Hyperparameter Grid ----------------
# xgbGrid <- expand.grid(
#   nrounds = c(50, 100, 150),             # number of boosting rounds
#   max_depth = c(3, 6, 9),                # maximum depth of trees
#   eta = c(0.01, 0.05, 0.1),              # learning rate
#   gamma = c(0, 0.5, 1),                  # minimum loss reduction required to make a further partition
#   colsample_bytree = c(0.6, 0.8, 1.0),   # subsample ratio of columns when constructing each tree
#   min_child_weight = c(1, 3, 5),         # minimum sum of instance weight needed in a child
#   subsample = c(0.6, 0.8, 1.0)           # subsample ratio of the training instances
# )
# 
# # ---------------- Train the XGBoost Model ----------------
# xgb_tuned <- train(
#   rainfall ~ .,
#   data = trainData,
#   method = "xgbTree",
#   trControl = fitControl,
#   tuneGrid = xgbGrid,
#   metric = "ROC"  # Optimize based on ROC AUC
# )
# 
# # Display the best tuned hyperparameters
# print(xgb_tuned$bestTune)
# 
# # Generate predicted probabilities on the test set using the tuned model.
# pred_prob <- predict(xgb_tuned, newdata = testData, type = "prob")[, "Yes"]
# 
# # Convert probabilities to class predictions with a threshold of 0.5.
# pred_class <- ifelse(pred_prob > 0.5, "Yes", "No")
# pred_class <- factor(pred_class, levels = c("No", "Yes"))
# 
# # Evaluate the model with a confusion matrix.
# cm <- confusionMatrix(
#   reference = testData$rainfall,
#   data = pred_class
# )
# print(cm)
# 
# # ROC Analysis
# # Convert actual responses into a binary numeric vector ("Yes" = 1, "No" = 0).
# actual_numeric <- ifelse(testData$rainfall == "Yes", 1, 0)
# roc_obj <- roc(actual_numeric, pred_prob)
# 
# roc_data <- data.frame(
#   fpr = 1 - roc_obj$specificities,
#   tpr = roc_obj$sensitivities
# )
# 
# ggroc_xgb <- ggplot(roc_data, aes(x = fpr, y = tpr)) +
#   geom_line(color = "blue", size = 1) +
#   geom_abline(linetype = "dashed", color = "red") +
#   labs(title = "ROC Curve for Tuned XGBoost Model",
#        x = "False Positive Rate",
#        y = "True Positive Rate") +
#   theme_wsj()
# 
# print(ggroc_xgb)
# 
# auc_value <- auc(roc_obj)
# cat("AUC:", auc_value, "\n")
# 
# # ---------------- Predict on Final Test Data and Create Submission ---------------
# pp <- preProcess(test, method = "medianImpute")
# test_imputed <- predict(pp, test)
# 
# pred_prob_test <- predict(xgb_tuned, newdata = test_imputed, type = "prob")[, "Yes"]
# test$rainfall <- ifelse(pred_prob_test > 0.5, 1, 0)
# 
# submission <- test %>% select(id, rainfall)
# write.csv(submission, "submission.csv", row.names = FALSE)
# head(submission)

# ----------------- TUNED
# nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
# 263     100         6 0.01     0              0.6                5       0.6

# # Load libraries
# library(caret)         # For data partitioning and evaluation
# library(dplyr)         # For data manipulation
# library(ggplot2)       # For visualization
# library(ggthemes)      # For ggplot2 themes
# library(naniar)        # For missing data visualization
# library(nnet)          # For neural network model
# library(pROC)          # For ROC analysis
# library(randomForest)  # For Random Forest model
# library(xgboost)       # For XGBoost model
# 
# # Ensure reproducibility
# set.seed(666)
# 
# # Load data
# train <- read.csv("data/train.csv")
# test <- read.csv("data/test.csv")
# 
# # ---------- New Feature Engineering --------------
# # 1. Temperature Range (maxtemp - mintemp)
# train$temp_range <- train$maxtemp - train$mintemp
# test$temp_range <- test$maxtemp - test$mintemp
# 
# # 2. Dewpoint Gap (temparature - dewpoint)
# train$temp_dew_diff <- train$temparature - train$dewpoint
# test$temp_dew_diff <- test$temparature - test$dewpoint
# 
# # 3. Wind Vector Components
# train$wind_rad <- train$winddirection * pi / 180
# train$wind_x <- train$windspeed * cos(train$wind_rad)
# train$wind_y <- train$windspeed * sin(train$wind_rad)
# 
# test$wind_rad <- test$winddirection * pi / 180
# test$wind_x <- test$windspeed * cos(test$wind_rad)
# test$wind_y <- test$windspeed * sin(test$wind_rad)
# 
# # 4. Pressure Change (difference from previous observation; first observation set to NA)
# train$pressure_change <- c(NA, diff(train$pressure))
# test$pressure_change <- c(NA, diff(test$pressure))
# train$pressure_change[is.na(train$pressure_change)] <- 0
# test$pressure_change[is.na(test$pressure_change)] <- 0
# 
# # 5. Sunshine Features
# train$sunshine_log <- log(train$sunshine + 1)   # Log-transformed sunshine (adding 1 avoids log(0))
# test$sunshine_log <- log(test$sunshine + 1)
# 
# train$sun_frac <- with(train, ifelse((cloud + sunshine) > 0, sunshine / (cloud + sunshine), 0))
# test$sun_frac <- with(test, ifelse((cloud + sunshine) > 0, sunshine / (cloud + sunshine), 0))
# 
# # Print a summary of the new features to confirm
# summary(train[, c("temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")])
# summary(test[, c("temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")])
# 
# # Remove 'id' and 'day' columns (if present) and create the modeling dataframe.
# data_model <- train[, c("pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
#                         "humidity", "cloud", "sunshine", "winddirection", "windspeed", "rainfall",
#                         "temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")]
# 
# # Split data_model into training and testing sets (80/20 split)
# trainIndex <- createDataPartition(data_model$rainfall, p = 0.8, list = FALSE)
# trainData <- data_model[trainIndex, ]
# testData <- data_model[-trainIndex, ]
# 
# # Recode the target variable so that "Yes" indicates rainfall and "No" no rainfall.
# trainData$rainfall <- factor(trainData$rainfall, levels = c("0", "1"), labels = c("No", "Yes"))
# testData$rainfall  <- factor(testData$rainfall, levels = c("0", "1"), labels = c("No", "Yes"))
# 
# n_miss(trainData$rainfall)
# n_miss(testData$rainfall)
# 
# # ---------------- Principal Component Analysis (PCA) ----------------
# # Apply PCA on the predictor variables in the training data (excluding the target 'rainfall')
# # It's important to scale/center the data for PCA
# 
# # Select only numeric predictor columns (exclude the target)
# predictor_vars <- trainData[, !names(trainData) %in% c("rainfall")]
# 
# # Perform PCA with centering and scaling
# pca_res <- prcomp(predictor_vars, center = TRUE, scale. = TRUE)
# 
# # Create a dataframe that includes the first two principal components and the target
# pca_df <- data.frame(PC1 = pca_res$x[,1],
#                      PC2 = pca_res$x[,2],
#                      rainfall = trainData$rainfall)
# 
# # Visualize the first two principal components colored by rainfall (target)
# ggplot(pca_df, aes(x = PC1, y = PC2, color = rainfall)) +
#   geom_point(alpha = 0.7, size = 2) +
#   labs(title = "PCA: PC1 vs PC2",
#        x = "Principal Component 1",
#        y = "Principal Component 2") +
#   theme_minimal()
# 
# # Create a scree plot to show the proportion of variance explained by each principal component
# var_explained <- (pca_res$sdev)^2 / sum((pca_res$sdev)^2)
# var_df <- data.frame(PC = 1:length(var_explained), Variance = var_explained)
# 
# ggplot(var_df, aes(x = PC, y = Variance)) +
#   geom_bar(stat = "identity", fill = "steelblue") +
#   labs(title = "Scree Plot",
#        x = "Principal Component",
#        y = "Proportion of Variance Explained") +
#   theme_minimal()
# 
# # ---------------- Set up Cross-Validation Controls ----------------
# fitControl <- trainControl(
#   method = "cv",                # k-fold cross-validation
#   number = 5,                   # 5-fold cross-validation
#   verboseIter = TRUE,           # output training progress
#   classProbs = TRUE,            # required for twoClassSummary
#   summaryFunction = twoClassSummary,
#   allowParallel = TRUE
# )
# 
# # ---------------- Define XGBoost Hyperparameter Grid ----------------
# xgbGrid <- expand.grid(
#   nrounds = 100,             # number of boosting rounds
#   max_depth = 6,                # maximum depth of trees
#   eta = 0.01,              # learning rate
#   gamma = 0,                  # minimum loss reduction required to make a further partition
#   colsample_bytree = 0.6,   # subsample ratio of columns when constructing each tree
#   min_child_weight = 3,         # minimum sum of instance weight needed in a child
#   subsample = 0.6           # subsample ratio of the training instances
# )
# 
# # ---------------- Train the XGBoost Model ----------------
# xgb_tuned <- train(
#   rainfall ~ .,
#   data = trainData,
#   method = "xgbTree",
#   trControl = fitControl,
#   tuneGrid = xgbGrid,
#   metric = "ROC"  # Optimize based on ROC AUC
# )
# 
# # Display the best tuned hyperparameters
# # print(xgb_tuned$bestTune)
# 
# # Generate predicted probabilities on the test set using the tuned model.
# pred_prob <- predict(xgb_tuned, newdata = testData, type = "prob")[, "Yes"]
# 
# # Convert probabilities to class predictions with a threshold of 0.5.
# pred_class <- ifelse(pred_prob > 0.5, "Yes", "No")
# pred_class <- factor(pred_class, levels = c("No", "Yes"))
# 
# # Evaluate the model with a confusion matrix.
# cm <- confusionMatrix(
#   reference = testData$rainfall,
#   data = pred_class
# )
# print(cm)
# 
# # ROC Analysis
# # Convert actual responses into a binary numeric vector ("Yes" = 1, "No" = 0).
# actual_numeric <- ifelse(testData$rainfall == "Yes", 1, 0)
# roc_obj <- roc(actual_numeric, pred_prob)
# 
# roc_data <- data.frame(
#   fpr = 1 - roc_obj$specificities,
#   tpr = roc_obj$sensitivities
# )
# 
# ggroc_xgb <- ggplot(roc_data, aes(x = fpr, y = tpr)) +
#   geom_line(color = "blue", size = 1) +
#   geom_abline(linetype = "dashed", color = "red") +
#   labs(title = "ROC Curve for Tuned XGBoost Model",
#        x = "False Positive Rate",
#        y = "True Positive Rate") +
#   theme_wsj()
# 
# print(ggroc_xgb)
# 
# auc_value <- auc(roc_obj)
# cat("AUC:", auc_value, "\n")
# 
# # ---------------- Predict on Final Test Data and Create Submission ---------------
# pp <- preProcess(test, method = "medianImpute")
# test_imputed <- predict(pp, test)
# 
# pred_prob_test <- predict(xgb_tuned, newdata = test_imputed, type = "prob")[, "Yes"]
# test$rainfall <- ifelse(pred_prob_test > 0.5, 1, 0)
# 
# submission <- test %>% select(id, rainfall)
# write.csv(submission, "submission.csv", row.names = FALSE)
# head(submission)

# Model 4.2 Neural Network - expanded hyperparameter tuning grid
# changed from 5 to 10 fold cross-validation
# Load libraries
library(caret)         # For data partitioning and evaluation
library(dplyr)         # For data manipulation
library(ggplot2)       # For visualization
library(ggthemes)      # For ggplot2 themes
library(naniar)        # For missing data visualization
library(nnet)          # For neural network model
library(pROC)          # For ROC analysis
library(randomForest)  # For Random Forest model
library(xgboost)       # For XGBoost model

# Ensure reproducibility
set.seed(666)

# Load data
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

# ---------- New Feature Engineering --------------
# 1. Temperature Range (maxtemp - mintemp)
train$temp_range <- train$maxtemp - train$mintemp
test$temp_range <- test$maxtemp - test$mintemp

# 2. Dewpoint Gap (temparature - dewpoint)
train$temp_dew_diff <- train$temparature - train$dewpoint
test$temp_dew_diff <- test$temparature - test$dewpoint

# 3. Wind Vector Components
train$wind_rad <- train$winddirection * pi / 180
train$wind_x <- train$windspeed * cos(train$wind_rad)
train$wind_y <- train$windspeed * sin(train$wind_rad)

test$wind_rad <- test$winddirection * pi / 180
test$wind_x <- test$windspeed * cos(test$wind_rad)
test$wind_y <- test$windspeed * sin(test$wind_rad)

# 4. Pressure Change (difference from previous observation; first observation set to NA)
train$pressure_change <- c(NA, diff(train$pressure))
test$pressure_change <- c(NA, diff(test$pressure))
train$pressure_change[is.na(train$pressure_change)] <- 0
test$pressure_change[is.na(test$pressure_change)] <- 0

# 5. Sunshine Features
train$sunshine_log <- log(train$sunshine + 1)   # Log-transformed sunshine (adding 1 avoids log(0))
test$sunshine_log <- log(test$sunshine + 1)

train$sun_frac <- with(train, ifelse((cloud + sunshine) > 0, sunshine / (cloud + sunshine), 0))
test$sun_frac <- with(test, ifelse((cloud + sunshine) > 0, sunshine / (cloud + sunshine), 0))

# Print a summary of the new features to confirm
summary(train[, c("temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")])
summary(test[, c("temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")])

# Remove 'id' and 'day' columns (if present) and create the modeling dataframe.
data_model <- train[, c("pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
                        "humidity", "cloud", "sunshine", "winddirection", "windspeed", "rainfall",
                        "temp_range", "temp_dew_diff", "wind_x", "wind_y", "pressure_change", "sunshine_log", "sun_frac")]

# Split data_model into training and testing sets (80/20 split)
trainIndex <- createDataPartition(data_model$rainfall, p = 0.8, list = FALSE)
trainData <- data_model[trainIndex, ]
testData <- data_model[-trainIndex, ]

# Recode the target variable so that "Yes" indicates rainfall and "No" no rainfall.
trainData$rainfall <- factor(trainData$rainfall, levels = c("0", "1"), labels = c("No", "Yes"))
testData$rainfall  <- factor(testData$rainfall, levels = c("0", "1"), labels = c("No", "Yes"))
n_miss(trainData$rainfall)
n_miss(testData$rainfall)

# Set up cross-validation controls
fitControl <- trainControl(
  method = "cv",                # k-fold cross-validation
  number = 5,                   # number of folds
  verboseIter = FALSE,          # do not output detailed progress
  classProbs = TRUE,            # needed for twoClassSummary to compute ROC
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# Define a grid of hyperparameters for neural network tuning.
# 'size' is the number of hidden units, 'decay' is the regularization parameter.
nnGrid <- expand.grid(
  size = c(5, 10, 15, 20, 25),         # number of hidden units
  decay = c(0, 0.001, 0.01, 0.1, 1)     # regularization parameter to avoid overfitting
)

# Train the neural network model using caret's train() function.
# The preProcess argument imputes missing values using the median before training.
nn_tuned <- train(
  rainfall ~ .,
  data = trainData,
  method = "nnet",
  trControl = fitControl,
  tuneGrid = nnGrid,
  metric = "ROC",           # optimize based on ROC AUC
  trace = FALSE,            # disable printing of training iterations
  maxit = 200               # maximum iterations (adjust as necessary)
)

# Display the best tuned hyperparameters
print(nn_tuned$bestTune)

# Generate predicted probabilities on the test set using the tuned model.
# Here, we also preProcess the testData so that missing values are handled in the same way.
pred_prob_nn <- predict(nn_tuned, newdata = testData, type = "prob")[, "Yes"]

# Convert predicted probabilities to class predictions using 0.5 as the threshold.
pred_class_nn <- ifelse(pred_prob_nn > 0.5, "Yes", "No")
pred_class_nn <- factor(pred_class_nn, levels = c("No", "Yes"))

# Evaluate the model with a confusion matrix.
cm_nn <- confusionMatrix(
  reference = testData$rainfall,
  data = pred_class_nn
)
print(cm_nn)

# ROC analysis using numeric values directly.
roc_obj_nn <- roc(testData$rainfall, pred_prob_nn)

# Visualize the ROC curve.
roc_data <- data.frame(
  fpr = 1 - roc_obj_nn$specificities,
  tpr = roc_obj_nn$sensitivities
)

ggroc_nn <- ggplot(data = roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", size = 1) +
  geom_abline(linetype = "dashed", color = "red") +
  labs(title = "ROC Curve for Tuned Neural Network",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_clean()

print(ggroc_nn)

# Print the AUC value.
auc_value <- auc(roc_obj_nn)
cat("AUC:", auc_value, "\n")

# any_na(test)
# miss_case_summary(test)

# Impute missing values in the test dataframe using median imputation
pp <- preProcess(test, method = "medianImpute")
test_imputed <- predict(pp, test)

# Generate predicted probabilities on the original test dataframe using the tuned model.
pred_prob_test <- predict(nn_tuned, newdata = test_imputed, type = "prob")[, "Yes"]

# Convert predicted probabilities to class predictions using 0.5 as the threshold.
test$rainfall <- ifelse(pred_prob_test > 0.5, 1, 0)

# Create a new dataframe for submission with the updated test dataframe.
submission <- test %>%
  select(id, rainfall)

# Write the submission dataframe to a CSV file.
write.csv(submission, "submission.csv", row.names = FALSE)

# Print the first few rows of the submission dataframe to verify.
head(submission)

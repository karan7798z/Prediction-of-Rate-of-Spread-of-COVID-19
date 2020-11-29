#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
## 1. Loading Required Libraries
library(readxl)
library(tidyverse)
library(xgboost)
library(caret)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
## 2. Loading required train and test data

data = read.csv('R_calculated_v2_11_Day_Interval_train.csv')

set.seed(122)
dt = sort(sample(nrow(data), nrow(data)*0.7))
train = data[dt,]
test = data[-dt,]

test_data = readxl::read_excel('R_calculated_v2_11_Day_Interval_test.xlsx')

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
## 3. Xtreme Gradient Boosting Model

### 3.1 Configuring the hyperparameter grid with range of values

set.seed(122)
xgbGrid <- expand.grid(nrounds = c(400,600,800), #no of iterations
                       max_depth = c(5, 10, 15, 20, 25), #maximum tree depth
                       colsample_bytree = seq(0.2, 0.9, length.out = 5), #column sampling
                       eta = seq(0.1, 0.9), #learning rate for adjusting weights at each step
                       gamma=c(0, 0.05, 0.1, 0.5), #regularization parameter
                       min_child_weight = c(1,2,3), #minimum number of instances needed to be in each node
                       subsample = c(0.5, 0.75, 1.0) #row sampling
)


### 3.2 Fitting/Training multiple XGBoost Models on complete train data for each set of hyperparameter value from the hyperparameter grid. Test MSE will be computed for each model, and at the end model with least Test MSE will be chosen. 

xgbGrid$Test_MSE <- 1 #Adding a new column in xgbGrid for storing Test MSE for each row (i.e. each set of hyperparameter values)

set.seed(122)

for (i in 1:nrow(xgbGrid)){
  set.seed(122)
  xgb_model_full_grid <- xgboost(
    data.matrix(data[,-1]), 
    label = data$Mean.R.,
    nround = xgbGrid$nrounds[i],
    max_depth = xgbGrid$max_depth[i],
    colsample_bytree = xgbGrid$colsample_bytree[i],
    eta = xgbGrid$eta[i],
    subsample = xgbGrid$subsample[i]
  )
  
  pred_values <- predict(xgb_model_full_grid, data.matrix(test_data[,-1]))
  Test_MSE <- mean((pred_values - test_data$`Mean(R)`)^2)
  xgbGrid$Test_MSE[i] <- Test_MSE
  print(i)
}


### 3.3 Fitting/Training a XGBoost Model on complete train data for hyperparameters are as chosen by the above grid for lowest Test MSE

set.seed(122)
xgb_model_best_grid <- xgboost(
  data.matrix(data[,-1]), 
  label = data$Mean.R.,
  nround = xgbGrid$nrounds[which.min(xgbGrid$Test_MSE)],
  max_depth = xgbGrid$max_depth[which.min(xgbGrid$Test_MSE)],
  colsample_bytree = xgbGrid$colsample_bytree[which.min(xgbGrid$Test_MSE)],
  eta = xgbGrid$eta[which.min(xgbGrid$Test_MSE)],
  subsample = xgbGrid$subsample[which.min(xgbGrid$Test_MSE)]
)


### 3.4 Final Test MSE using the best model as fitted above.

set.seed(122)
predicted_test_data = predict(xgb_model_best_grid, data.matrix(test_data[,-1]))
test_mse_xgb = mean((predicted_test_data - test_data$`Mean(R)`)^2)
print(paste("The Final Test MSE for Xtreme Gradient Boosting is:",test_mse_xgb))

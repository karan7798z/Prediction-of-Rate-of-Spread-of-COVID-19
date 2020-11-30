# Prediction-of-Rate-of-Spread-of-COVID-19

In this Project, we have tried to predict the current speed of COVID-19 spread using mobility data from Google - [https://www.google.com/covid19/mobility/]. 
This mobility data shows how visits to places such as Grocery stores and Parks, are changing in a sample locality. 

To predict the rate of spread we have tried using different models, which majorly fall under 2 approaches: 

**1. Regression methods** - For this, we have used the following approaches - 

  * 1.1. Linear Regression on complete dataset - This approach was majorly used to be able to compute the Variance Inflation Factor of the predictors, because we expected heavy multicollinearity to exist between them. The dataset involves 6 different mobility types, recorded for t-11 days thus composing the 66 predictors. The computed VIF confirmed our assumption of multicollinearity, as indicated by the extremely high factor values for each predictor. We thus proceeded towards subset selection and shrinkage methods to reduce the feature space.
  
  * 1.2. Backward Subset Selection
  * 1.3. Lasso Regression
  * 1.4. Ridge Regression
  * 1.5. Best Subset Selection from amongst the chosen Principal Components after PCA
  
**2. Tree-Based Methods** - For this, we have used the following approaches

  * 2.1. Bagging/Random Forest
  * 2.2. Xtreme Gradient Boosting

Our final model is an Xtreme Gradient Boosting Model which has been chosen as the final model because it provided the smallest Test Mean Squared error, when trained upon the provided Train data which consisted of 66 features and 152 rows. The final test MSE provided by our XGBoost Model, which was fitted after choosing the best set of hyperparameter values (chosen by fitting a prior model for a grid of hyperparameter values and evaluating the test mse for each value set, choosing the value set that resulted in the smallest test mse), was reported to be 0.00373547855

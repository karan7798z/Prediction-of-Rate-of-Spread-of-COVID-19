# Prediction-of-Rate-of-Spread-of-COVID-19

In this Project, we have tried to predict the current speed of COVID-19 spread using mobility data from Google - [https://www.google.com/covid19/mobility/]. 
This mobility data shows how visits to places such as Grocery stores and Parks, are changing in a sample locality. 

Our final model is an Xtreme Gradient Boosting Model which has been chosen as thet final model because it provided the smallest Test Mean Squared error, when trained upon the provided Train data which consisted of 66 features and 152 rows. The final test MSE provided by our XGBoost Model, which was fitted after choosing the best set of hyperparameter values (chosen by fitting a prior model for a grid of hyperparameter values and evaluating the test mse for each value set), was reported to be 0.00373547855

# K-Fold CV in regression

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

# Set seed for reproducibility
SEED = 123

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=SEED)

# Instantiate decision tree regressor and assign it to 'dt'
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.14, random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(dt, X_train, y_train, cv= 10, scoring= 'neg_mean_squared_error' , n_jobs = -1)

# Fit 'dt' to the training set
dt.fit(X_train, y_train)

# Predict the labels of training set
y_predict_train = dt.predict(X_train)

# Predict the labels of test set
y_predict_test = dt.predict(X_test)

# CV MSE
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))

# Training set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_predict_train)))

# Test set MSE
print('Test MSE: {:.2f}'.format(MSE(y_test, y_predict_test)))

# Suppose CV MSE = 20.51, Train MSE = 15.30 and Test MSE = 20.92
# Train MSE < CV MSE.
# Suggested that model is overfit and is suffering from high variance.
# CV MSE and Test MSE are roughly equal

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV.mean())**(1/2)
# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)
# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))

# Suppose, RMSE_CV = 5.14, RMSE_train = 5.15 and baseline_RMSE = 5.1
# RMSE_CV < RMSE_train means dt suffers from high bias because RMSE_CV ≈ RMSE_train and both scores are greater than baseline_RMSE.
# dt is indeed underfitting the training set as the model is too constrained to capture the nonlinear dependencies between features and labels

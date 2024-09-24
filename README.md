# Offline Casual Neural Decoder to Drive a Prosthetic Limb

## Data
The monkey_training.m data consists of spike trains recorded from 98 single and multi-neuron units from a monkey’s motor cortex, alongside hand coordinates as it moves in eight distinct angles. Recordings extend from 300ms before movement onset to 100ms after the end of the movement. There are 100 trials in each direction.

## Position Estimation Algorithm
### Position Estimator Training
The positionEstimatorTraining.mat file takes the training data set, resizes it for all spike trains to be of equal sizes and generates both an angle predictor and a hand postion regression model.  
1) Angle Classification: a K-nearest neighbours (kNN) with k=15 takes the spike count in the first 320ms of each trial, since the reaching angle remains consistent throughout a trial. This results in 800 vectors of size 1x98, one per trial and angle, and this is stored as a matrix.
2) Hand Postion Regression: a linear regression model predicts hand position using firing rates in 20ms windows. The average velocity is computed using hand position data, and a regression model is fit using the corresponding average firing rates of neurons across trials. Before fitting the model, PCA with 2 principal components is performed to reduce the dimensionality of the firing rate data.  
### Position Estimator
The positionEstimator.mat file processes a testing data set to estimate angle and hand position.  
1) Angle Classification: using the knn classifier, it compares the sum of spike counts within the first 320 ms of the test data to a matrix of 800 precomputed vectors (from training trials) and calculates the Euclidean distance between them. The angle corresponding to the nearest neighbors is selected as the predicted reaching direction.
2) Hand Position Regression: After predicting the angle, the corresponding linear regression model (trained earlier) is used to estimate the hand's velocity. The neural firing rates from the current time window (usually 20 ms) are transformed into a reduced-dimensional space using Principal Component Analysis (PCA). These features are then mapped to hand velocity in both the x and y directions. The hand's position is updated at each time step by integrating the velocity over the current window. If this is the first time step (within the first 320 ms), the hand position is initialized to the starting hand position provided in the test data. After 500 ms, the hand is assumed to have stopped moving.  
### Test Function
The testFunction.m file evaluates the performance of the neural decoder on unseen monkey data. The file first trains a model using positionEstimatorTraining, and then decodes hand trajectories by calling positionEstimator. The root mean square error (RMSE) is calculated as the performance metric.

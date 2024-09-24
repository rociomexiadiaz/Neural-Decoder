# Offline Casual Neural Decoder to Drive a Prosthetic Limb

## Data
The monkey_training.m data consists of spike trains recorded from 98 single and multi-neuron units from a monkey’s motor cortex, alongside hand coordinates as it moves in eight distinct angles. Recordings extend from 300ms before movement onset to 100ms after the end of the movement. There are 100 trials in each direction.

## Position Estimation Algorithm
### Position Estimator Training
The positionEstimatorTraining.mat file takes some training data, resizes it for all neural data vectors to be of equal sizes and generates both an angle predictor and a hand postion regression model.  
The K-nearest neighbours (kNN) with k=15 used for angle classification, takes the spike count in the first 320ms since the reaching angle remains consistent throughout a trial. This resulted in 800 vectors of size 1x98, one per trial and angle.  
The linear regression model was found by splitting the data into 20ms windows, and for each window computing the average velocity using the hand position data in the training set. This was fit onto the model with average firing rates in those windows,  done by averaging all trials in each direction.  
PCA was used to reduce the dimensionality of firing rates.  
### Position Estimator
The positionEstimator.mat file takes a training data and, using the matrix of 800 vectors obtained from the knn algorithm and finding the smallest euclidean distance, estimates the angle in which the arm is reaching towards.  
Now using the angle defined, the corresponding linear regression model is maped to give the hand position at each 20ms time window.  
### Test Function
The testFunction.mat file takes unseen monkey data, in the same format as the data described above and calculates its performance metric- RMSE.

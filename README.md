# Offline Casual Neural Decoder to Drive a Prosthetic Limb

## Data
The monkey_training.m data consists of spike trains recorded from 98 single and multi-neuron units from a monkey’s motor cortex, alongside hand coordinates as it moves in eight distinct angles. Recordings extend from 300ms before movement onset to 100ms after the end of the movement. This was repeated for 100 trials per direction, which was later split into training and testing subsets.  

## Algorithm
K-nearest neighbours (kNN) with k=15 was used for angle classification. The firing rate was defined as the spike count in the 320ms window, since the reaching angle remains consistent throughout a trial. This resulted in 800 vectors of size 1x98, one per trial and angle. PCA and Linear regression were performed for hand position. By running the testFunction_for_student_MTb.m the algorithm is tested for its efficacy.

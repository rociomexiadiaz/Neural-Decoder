function [x, y] = positionEstimator(test_data, modelParameters)
    % Find length of spikes in test data
    spikes_test = test_data.spikes;

    %% CALCULATE TIMESTEP OF TEST FUNCTION (their intervals)
    % When first called - set previous spike count to 0
    if ~isfield(modelParameters, 'previousSpikeCount')
        modelParameters.previousSpikeCount = 0;
    end
   
    currentSpikeCount = size(test_data.spikes, 2); % Size of current spike
    if modelParameters.previousSpikeCount > 0
        timeStep = currentSpikeCount - modelParameters.previousSpikeCount; % Difference between current & previous spike length
    else
        timeStep = 20; % If first time - assume 20ms (in testFunction)
    end
    
    % Update the spike count for the next call
    modelParameters.previousSpikeCount = currentSpikeCount;


    %% KNN - Feature Vector

    % Matrix to assign angle
    trains= [];
    for i=1:8
        trains = [trains, i*ones(1,modelParameters.num_trials)];
    end
    
    
    new_test = sum(spikes_test(:, 1:320), 2); % Sum test spikes over first 320ms
    distances = sqrt(sum((modelParameters.sumTime - new_test').^2, 2)); % Calculates the difference between training to testing sum
    [~, indices] = sort(distances); 
    neighbours = trains(indices(1:modelParameters.knn)); % Find angle from indicies with shortest distance
    predicted_angle = mode(neighbours);
    
    % Calculate mean firing rate for current window
    current_window_FR = mean(spikes_test(:, end-modelParameters.window_size+1:end),2);
    current_window_FR = current_window_FR - mean(current_window_FR, 1); % Center data for PCA
    
    % Transform test data to PCA space
    current_window_FR_pca = current_window_FR' * modelParameters.pcacoeffs{predicted_angle}(:,1:2);
    
    % Use linear regression coefficients for predicted angle to estimate velocity
    %Vx = current_window_FR_pca * modelParameters.coeff{predicted_angle}(:,1); % Velocity in X direction
    %Vy = current_window_FR_pca * modelParameters.coeff{predicted_angle}(:,2); % Velocity in Y direction
    Vx = predict(modelParameters.coeff{predicted_angle,1}, current_window_FR_pca);
    Vy = predict(modelParameters.coeff{predicted_angle,2}, current_window_FR_pca);
   

    % Calculate displacement in that time step
    dis_x = Vx*timeStep;
    dis_y = Vy*timeStep;

    % Start at the given starting position
    if length(test_data.spikes)<=320
        test_data.decodedHandPos=test_data.startHandPos; 
        x = test_data.decodedHandPos(1,end);
        y = test_data.decodedHandPos(2,end);
        test_data.decodedHandPos = [x; y];
    end

    % No movement in last 100ms
    if length(test_data.spikes)>=500
        dis_x = 0;
        dis_y = 0;
    end

    % Update x and y positions
    if length(test_data.spikes)>320
        x = test_data.decodedHandPos(1,end)+dis_x;
        y = test_data.decodedHandPos(2,end)+dis_y;
        test_data.decodedHandPos = [x; y];
    end

end

function [modelParameters] = positionEstimatorTraining(training_data)
    [num_trials, num_angles] = size(training_data);
    num_neurons = length(training_data(1,1).spikes(:,1) );
    knn = 15; % k-nearest neighbours
    window_size = 20;
    
    %% DATA PREPROCESSING - Crop length of training data to ensure all same length
    minLen_spike = inf;
    minLen_hand = inf;

    % Loop through angles, trials, and neurons - finding shortest length
    for angle = 1:num_angles
        for trial = 1:num_trials
            for neuron = 1:num_neurons
                currentLen_spike = length(training_data(trial,angle).spikes);
                % Find minimum length 
                if currentLen_spike < minLen_spike
                    minLen_spike = currentLen_spike;
                end
            end
        end
    end

    % Resize each spike and hand position vector to match the minimum length
    for trial = 1:num_trials
        for angle = 1:num_angles
            training_data(trial,angle).spikes = training_data(trial,angle).spikes(:, 1:minLen_spike);
            training_data(trial,angle).handPos = training_data(trial,angle).handPos(:, 1:minLen_spike);
        end
    end
    
    
    %% KNN - PREDICT ANGLE
    % Feature Vector (one sum per neuron per trial for the first 320ms)
    sumTime =[];
    for angle = 1:num_angles 
        % All spikes before 320 from all trials and all neurons
        sum_neuron_B4 = zeros(num_trials, num_neurons); % number trials x number neurons    
        for neuron = 1:num_neurons 
            for trial = 1:num_trials
                sum_neuron_B4(trial,neuron) = sum(training_data(trial, angle).spikes(neuron, 1:320));
            end            
        end
        % Appends sum_neuron_B4 below current contents of sumTime (number of angles x number of training trials = _ rows)- first (number of training trials) for angle 1, next for angle 2... 
        sumTime = [sumTime; sum_neuron_B4]; % each column is a neuron, each cell is sum of neuron for that trial
    end
    
    
    %% LINEAR REGRESSION - PREDICT VELOCITY
    % Find Velocity for each time window
    velocity_matrix = cell(num_trials, num_angles); 
    for trial = 1:num_trials 
        for angle = 1:num_angles
            handPosition = training_data(trial, angle).handPos;

            % Calculate differences between adjacent elements
            displacement_X = diff(handPosition(1, :)); 
            displacement_Y = diff(handPosition(2, :));
            
            disX = [];
            disY= [];
            
            % Calculate total displacement per window
            for t = 300:window_size:length(displacement_X)-window_size
                disX = [disX, sum(displacement_X(t:t+window_size))]; % Keeps appending new columns with the displacement per window for that angle/trial 
                disY = [disY, sum(displacement_Y(t:t+window_size))];
            end
            velocity_X = disX / window_size; % Velocity per window 
            velocity_Y = disY / window_size;
            velocity_matrix{trial, angle} = [velocity_X; velocity_Y]; % number training trials x number of angles - each cell is 2 x number of windows, top is X, bottom is Y velocity
        end
    end

    % Average velocity per window per angle over all trials
    avgVel = cell(1, num_angles);
    for angle = 1:num_angles
        x_vel_sum = zeros (num_trials, length(velocity_matrix{1,angle}(1,:))); 
        y_vel_sum = zeros (num_trials, length(velocity_matrix{1,angle}(2,:))); 
        for trial = 1:num_trials
            x_vel_sum(trial,:) = velocity_matrix{trial,angle}(1,:);
            x_vel_average = mean(x_vel_sum, 1);
            y_vel_sum(trial,:) = velocity_matrix{trial,angle}(2,:);
            y_vel_average = mean (y_vel_sum, 1);
        end
        avgVel{1, angle} = [x_vel_average; y_vel_average]; % 1 x number of angles
    end
    
    % Average firing rate (FR) - per window per angle over all trials
    average_FR = cell(num_neurons, num_angles); 
    for angle = 1:num_angles
        for neuron = 1:num_neurons 
            FR = [];
            for trial = 1:num_trials
                spikes = training_data(trial,angle).spikes(neuron,:); % All spike data
                FR_binned = [];

                for t = 300 : window_size : length(training_data(trial,angle).spikes(neuron,:)) - window_size % binning spike data
                    % Mean spike over window for each neuron, trial, & angle
                    FR_binned = [FR_binned, mean(spikes(t:t+window_size))]; 
                end

                % Append this mean over window per trial 
                % number trials x number window
                FR = [FR; FR_binned]; 
    
            end
            average_FR{neuron,angle} = FR; 
        end
    end
    
    % Store linear regression coefficients
    coeff = cell(num_angles,2);

    % PCA to reduce dimensionality of data
    pcacoeffs = cell(num_angles,1); 

    for angle = 1:num_angles
        velocity = avgVel{1,angle};
        average_FR_combined = [];
    
        for neuron = 1:num_neurons 
            % Adds FR mean per neuron per window in new row
            average_FR_combined = [average_FR_combined; mean(average_FR{neuron,angle})];    
        end
        
        average_FR_combined = average_FR_combined - mean(average_FR_combined,1); % Centering data for PCA
        [pcacoeffs_angle, ~] = mypca(average_FR_combined'); % Function to find principal compoenent coefficient
        numComponents = 2; % Keep 2 principal components
        pcacoeffs{angle} = pcacoeffs_angle; % Storing the PCA coeff for each angle
        disp(size(pcacoeffs_angle))
        disp(size((average_FR_combined)))
        pcaspace_x = pcacoeffs_angle(:,1:numComponents)'*(average_FR_combined); % Transforms data to PCA space
        coeff{angle} = lsqminnorm(pcaspace_x', velocity'); % Finds the linear regression coefficients

        % SVR
        % SVMmodel_x = fitrsvm(pcaspace_x',velocity(1,:)','Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        % SVMmodel_y = fitrsvm(pcaspace_x',velocity(2,:)','Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
        % coeff{angle,1} = SVMmodel_x;
        % coeff{angle,2} = SVMmodel_y;

    end

    %% VARIABLES TO RETAIN
    modelParameters.coeff = coeff;
    modelParameters.knn = knn;
    modelParameters.sumTime = sumTime;
    modelParameters.num_trials = num_trials;
    modelParameters.window_size = window_size;
    modelParameters.pcacoeffs = pcacoeffs;


end

%% PCA Function - must first center data
function [eigenvectors, PCAscores]=mypca(data) 
    covariance = cov(data);
    [eigenvectors, eigenvalues_matrix] = eig(covariance);
    eigenvalues = diag(eigenvalues_matrix);
    [eigenvalues, sortIDx] = sort(eigenvalues, 'descend');
    eigenvectors = eigenvectors(:, sortIDx);  
    PCAscores = data * eigenvectors;
end


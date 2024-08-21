clc;
clear;
close all;

% Load data
data = readtable("german_dataset.csv");

% Specify the columns to use
columns = {'Target','Age_years_', 'ConcurrentCredits', 'CreditAmount','DurationOfCredit_month_', 'ForeignWorker','LengthOfCurrentEmployment', 'MostValuableAvailableAsset','PaymentStatusOfPreviousCredit', 'Purpose', 'Sex_MaritalStatus','TypeOfApartment', 'ValueSavings_Stocks'};

cols = columns(ismember(columns, data.Properties.VariableNames));
data = data(:, cols);

% Prepare input and target data
x = data(:, 2:end);
y = data(:, 1);
y = categorical(table2array(y)); % Convert target to categorical
y = onehotencode(y, 2); % One-hot encode the categorical target variable
x = table2array(x)';
y = y';

% Define the network architecture
trainFcn = 'trainscg';
hiddenLayerSize = [20, 10, 6];
net = patternnet(hiddenLayerSize, trainFcn);

% Set ReLU as the activation function for hidden layers
for i = 1:length(net.layers) - 1 % Excluding the output layer
    net.layers{i}.transferFcn = 'poslin'; % ReLU function
end


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Set batch size and number of epochs
net.trainParam.epochs = 1000; % Set the number of epochs
net.trainParam.showWindow = true; % Optional: Disable training window popup
net.trainParam.showCommandLine = true; % Optional: Show training progress in command line
net.trainParam.batchSize = 50; % Set the batch size

% Train the Network
[net, tr] = train(net, x, y);

% Make predictions
ypred = net(x);

% Save the trained network
save('trainedModel.mat', 'net');

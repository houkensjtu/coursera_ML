clear ; close all; clc
fprintf('Loading data ...\n');
data = load('hall_MLsample.txt');

X = data(:,1);
X = [X,X.^2,X.^3];
y = data(:,2);

[X mu sigma] = featureNormalize(X);
m = size(X,1);
X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.05;
num_iters = 200;

% Init Theta and Run Gradient Descent 
theta = zeros(4, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);

xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
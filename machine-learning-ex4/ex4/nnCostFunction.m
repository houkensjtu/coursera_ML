function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% To recode a new y matrix, convert a single digit into a R10 vector.
y_recode = zeros(m,num_labels);

for c = 1:num_labels
  for i = 1:m
    if y(i)==c
        y_recode(i,c) = 1;
    endif
  endfor
endfor

% Append a bias column to X.
a1 = [ones(m,1) X];

z2 = a1*Theta1';

a2 = sigmoid(z2);

a2 = [ones(m, 1) a2];

z3 = a2*Theta2';

a3 = sigmoid(z3);

% Notice the .* operator in cost function.
J = 1/m* sum(sum( -y_recode.*log(a3) - (1-y_recode).*log(1-a3)));

% Get the size of theta matrix, so that it can apply to theta of any size.
k_Theta1 = size(Theta1,2);
k_Theta2 = size(Theta2,2);

% Add regularization term to the cost function.
J = J + lambda/(2*m)*( sum(sum(Theta1.*Theta1)(:,2:k_Theta1)) + ...
    sum(sum(Theta2.*Theta2 )(:,2:k_Theta2)));

% -------------------------------------------------------------

% =========================================================================

Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));

a_1 = zeros(m,size(X,2)+1);
z_2 = zeros(m,size(Theta1,1));
a_2 = zeros(m,size(Theta2,2));
z_3 = zeros(m,size(Theta2,1));
a_3 = zeros(m,size(Theta2,1));
delta_3 = zeros(m,size(Theta2,1));
delta_2 = zeros(m,size(Theta2,2));

%a_1 = [ones(m,1) X];
%z_2 = a_1 * Theta1';
%a_2 = [ones(m,1) sigmoid(z_2)];
%z_3 = a_2 * Theta2';
%a_3 = sigmoid(z_3);
%delta_3 = a_3 - y_recode;
%delta_2 = delta_3 * Theta2 .* sigmoidGradient(a_2);
for t = 1:m

    a_1(t,:) = [1 X(t,:)]; % 1x401
    
    z_2(t,:) = a_1(t,:) * Theta1'; % 1x25
    a_2(t,:) = [1 sigmoid(z_2(t,:))]; % 1x26
    
    z_3(t,:) = a_2(t,:) * Theta2'; % 1x10
    a_3(t,:) = sigmoid(z_3(t,:)); % 1x10
    
    delta_3(t,:) = a_3(t,:) - y_recode(t,:); % 1x10
    
    delta_2(t,:) = delta_3(t,:) * Theta2 .* a_2(t,:) .* (1-a_2(t,:)); %1x26
    
    Delta_2 = Delta_2 + delta_3(t,:)' * a_2(t,:); % 10x26   
    Delta_1 = Delta_1 + delta_2(t,2:end)' * a_1(t,:); % 25x401
endfor 

Theta1_grad = 1/m * Delta_1;
Theta2_grad = 1/m * Delta_2;

Theta1_grad(:,2:k_Theta1) = Theta1_grad(:,2:k_Theta1) + lambda/m*Theta1(:,2:k_Theta1);
Theta2_grad(:,2:k_Theta2) = Theta2_grad(:,2:k_Theta2) + lambda/m*Theta2(:,2:k_Theta2);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

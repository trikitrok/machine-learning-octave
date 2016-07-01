function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = size(theta);

inv_m = 1/m;

z = X * theta;
h = sigmoid(z);
term1 = - y' * log(h);
term2 = - (1 - y)' * log(1 - h);
theta_no_1 = theta(2:n,:);
term3 = theta_no_1' * theta_no_1;
J = inv_m * (term1 + term2 + term3 * lambda/2);

grad_reg = zeros(size(theta));
grad_reg(2:n,:) = inv_m * lambda * theta_no_1;

r = h - y;
grad = inv_m * (X' * r) + grad_reg;

% =============================================================

end

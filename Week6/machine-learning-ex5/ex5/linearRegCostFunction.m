function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

n = size(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

inv_m = 1/m;

r = X * theta - y;
J = 0.5 * inv_m * r'*r;

theta_no_1 = theta(2:n,:);
J = J + lambda * 0.5 * inv_m * theta_no_1' * theta_no_1;

grad_reg = zeros(size(theta));
grad_reg(2:n,:) = inv_m * lambda * theta_no_1;

grad = inv_m * (X' * r) + grad_reg;

% =========================================================================

grad = grad(:);

end

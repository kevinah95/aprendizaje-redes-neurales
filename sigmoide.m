function g = sigmoide(z)
%SIGMOIDE Compute sigmoid functoon
%   J = SIGMOIDE(z) calcula la sigmoide de z.

g = 1.0 ./ (1.0 + exp(-z));
end

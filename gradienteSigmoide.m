function g = gradienteSigmoide(z)
%GRADIENTESIGMOIDE retorna el gradiente de la función sigmoide en z
%   g = GRADIENTESIGMOIDE(z) 

g = zeros(size(z));

% ====================== SU CÓDIGO AQUÍ ======================
% Instrucciones: Calcula el gradiente de la función sigmoide

% prueba
% Cuando z = 0, el gradiente debe ser igual a 0.25.


% g(z)(1 − g(z))
% sigmoide(z) = g(z)

g = sigmoid(z) .* (1 - sigmoid(z));











% =============================================================




end

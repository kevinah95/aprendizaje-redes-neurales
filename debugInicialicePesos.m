function W = debugInicialicePesos(fan_out, fan_in)
%DEBUGINICIALICEPESOS Initializa los pesos de una capa con fan_in conexiones
%   entrates y fan_out salientes. Utiliza una estratégia fija
%
%   Note que W debe ser una matriz de (1 + fan_in, fan_out) ya que se
%   ocupa agregar el sesgo
%

% Ponga ceros en W
W = zeros(fan_out, 1 + fan_in);

% Initialice W ustilizando "sin", esto asegura que siempre será la misma
% y sirve para el debugging
W = reshape(sin(1:numel(W)), size(W)) / 10;

% =========================================================================

end

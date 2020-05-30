function gradnum = calculeGradienteNumerico(J, theta)
%CALCULEGRADIENTENUMERICO Utiliza diferencias finitas
% y estima el gradiente 
%   gradnum = CALCULEGRADIENTENUMERICO(J, theta) calcula gradiente num.
%   alrededor de theta. Llamando y = J(theta) debería devolver el 
%   valor de la función en theta


gradnum = zeros(size(theta));
perturb = zeros(size(theta));
e = 1e-4;
for p = 1:numel(theta)
    % Ponga el vector perturbado
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    % Calcule el gradiente numérico
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
end

end

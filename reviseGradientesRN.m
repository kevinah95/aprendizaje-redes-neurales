function reviseGradientesRN(lambda)
%REVISEGRADIENTESRN Crea una pequeña red para revisar los gradientes de 
%   REVISEGRADIENTESRN(lambda) 
%   Calcula valores analíticos y los numéricos del gradiente
%   producidos por su código (en calculeGradienteNumerico). Estos
%   dos valores deben ser muy similares


if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

tam_capa_entrada = 3;
tam_capa_oculta = 5;
num_etiquetas = 3;
m = 5;

% Generamos datos 'random' 
Theta1 = debugInicialicePesos(tam_capa_oculta, tam_capa_entrada);
Theta2 = debugInicialicePesos(num_etiquetas, tam_capa_oculta);
% Reutilizar debugInicialicePesos para generar X
X  = debugInicialicePesos(m, tam_capa_entrada - 1);
y  = 1 + mod(1:m, num_etiquetas)';

% Desenrolle los parámetros 
params_rn = [Theta1(:) ; Theta2(:)];

% Función de costo
funcionDeCosto = @(p) funcionDeCostoRN(p, tam_capa_entrada, tam_capa_oculta, ...
                               num_etiquetas, X, y, lambda);

[costo, grad] = funcionDeCosto(params_rn);
gradnum = calculeGradienteNumerico(funcionDeCosto, params_rn);

% Examine visualmente los dos cómputos de gradiente.  Las dos columnas
% deben ser muy similares
disp([gradnum grad]);
fprintf(['Las dos columnas anteriores deben ser muy similiares.\n' ...
         '(Izq.-Gradiente Numérico, Der.-Gradiente Analítico)\n\n']);

% Evalue la norma de la diferencia de las dos soluciones.  
% Si tiene una implementación correcta, y asumiento que utilizó un
% EPSILON = 0.0001, la diferencia debe ser menor a 1e-9
diff = norm(gradnum-grad)/norm(gradnum+grad);

fprintf(['Si su implementación esta correcta, entonces \n' ...
         'la diferencia debe ser menor a 1e-9. \n' ...
         '\nDiferencia relativa: %g\n'], diff);

end

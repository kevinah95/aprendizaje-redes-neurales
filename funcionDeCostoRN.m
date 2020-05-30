function [J grad] = funcionDeCostoRN(params_rn, ...
                                     tam_capa_entrada, ...
                                     tam_capa_oculta, ...
                                     num_etiquetas, ...
                                     X, y, lambda)
%FUNCIONDECOSTORN Implementa el costo de la red neural
%   para una red de dos capas 
%   [J grad] = FUNCIONDECOSTORN(params_rn, tam_capa_entrada, tam_capa_oculta, ...
%   num_etiquetas,X, y, lambda) calcula el costo y gradiente de la red.
%   los parámetros son desenrollados en un vector y necesitan ser
%   reconvertidos a sus respectivas matrices de pesos
%
%   El parámetro devuelto en grad es el gradiente desenrollado de J.
%

% Reobtenga las parámetros Theta1 y Theta2 de params_rn
Theta1 = reshape(params_rn(1:tam_capa_oculta * (tam_capa_entrada + 1)), ...
                 tam_capa_oculta, (tam_capa_entrada + 1));

Theta2 = reshape(params_rn((1 + (tam_capa_oculta * (tam_capa_entrada + 1))):end), ...
                 num_etiquetas, (tam_capa_oculta + 1));

% Valores útiles
m = size(X, 1);
         
% Necesita devolver los siguientes valores correctamente
J = 0;
grad_Theta1 = zeros(size(Theta1));
grad_Theta2 = zeros(size(Theta2));

% ====================== SU CÓDIGO AQUÍ ======================
% Instrucciones: Usted debe completar el código trabajando en
%                las siguientes partes
%
% 1era Parte: Propagación hacia adelante y retorne el valor de J
%             después de calcularlo puede verificar el valor
%             con la prueba que se encuentra en ej4.m
%
% 2nda Parte: Implemente el algoritmo de retropropagación para calcular
%             las variables grad_Theta1 y grad_Theta2. Debe retornar las
%             derivadas parciales de Theta1 y Theta2 en grad_Theta1 y
%             grad_Theta2 respectivamente. Después de impletar esta
%             segunda parte, usted puede revisar que su implementación
%             es correcta corriendo revisarGradientesRN
%
%         Nota: El vector que se pasa a la función es un vector de etiquetas
%               conteniendo valores de 1..K, usted tiene que mapear este vector
%               a vectores binarios de 1s y 0s que necesita la función de 
%               costo de la red.
%
%         Pista: Le recomendamos implementar retropropagación utilizando un
%                ciclo for si es la primera vez que la implementa
%
% 3ra Parte: Implemente regularización a los costos y gradientes
%
%         Pista: Usted puede implementar esto aparte. Esto es, puede
%                calcular la regularización separadamente y luego
%                sumársela a grad_Theta1 y grad_Theta2
%



















% -------------------------------------------------------------

% =========================================================================

% Desenrollar gradientes
grad = [grad_Theta1(:) ; grad_Theta2(:)];


end

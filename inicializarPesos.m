function W = initializarPesos(L_in, L_out)
%INICIALIZARPESOS inicializa aleatoriamente los pesos
%

% Necesita retornar la siguiente variable correctamente
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instrucciones: Initialice W con pesos aleatoreos para romper
%                la simetría y permitir que la red entrene
%
% Nota: La primer fila corresponde a los parámetros de la neurona
%       de sesgo
%


epsilon_init = sqrt(6)/(sqrt(L_in) + sqrt(L_out));

W = rand(L_out, 1 + L_in)*2*epsilon_init - epsilon_init;






% =========================================================================

end

function [h, display_array] = despliegueDatos(X, example_width)
%DESPLIEGUEDATOS Despliega datos 2D en una grilla
%   [h, display_array] = DESPLIEGUEDATOS(X, example_width) despliegua
%   datos 2D guardados en X. Retorna el handle de la figura y el
%   array de despliegue si es requerido.
% 

% Ponga el example_width automáticamente si no es proveido
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

% Imagen en escala de grises
colormap(gray);

% Calcule filas, cols
[m n] = size(X);
example_height = (n / example_width);

% Calcule el número de items a desplegar
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Padding entre las imágenes
pad = 1;

% Inicialice el despliegue en blanco
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copie cada ejemplo a una celda del array de despliegue
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copie a la celda
		
		% Obtenga el valor máximo de la celda
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Despliegue la imagen
h = imagesc(display_array, [-1 1]);

% No muestre los ejes
axis image off

drawnow;

end

%% Aprendizaje - Ejercicio Redes Neurales

%  Instrucciones
%  -------------
% 
%  Este archivo contiene código para empezar el ejercicio 
%  de aprendizaje en redes neurales. debe completar las
%  siguientes funciones:
%
%     gradienteSigmoide.m
%     inicializarPesos.m
%     funcionDeCostoRN.m
%
%  No debe modificar código en este archivo,
%  solo en los mencionados anteriormente.
%

%% Initialización
clear ; close all; clc

%% Alistar los parámetros para este ejercicio
tam_capa_entrada = 400;  % Imágenes de dítigos de 20x20 
tam_capa_oculta  = 25;   % 25 unidades ocultas
num_etiquetas    = 10;   % 10 etiquetas, del 1 al 10 
                            % (se mapea el "0" a 10)

%% =========== 1era Parte: Cargar y visualizar los datos =============
%  Empezamos cargando y visualizando los datos.
%

% Cargando datos de entrenamiento
fprintf('Cargando y visualizado los datos ...\n')

load('ej4data1.mat');
m = size(X, 1);

% Seleccione aleatoriamente 100 ejemplos para desplegar
sel = randperm(size(X, 1));
sel = sel(1:100);

despliegueDatos(X(sel, :));

fprintf('Programa en pausa. Oprima enter para continuar.\n');
% pause;


%% ================ 2da Parte: Cargando Parámetros ================
% En esta parte del ejercicio, cargamos datos pre-inicializados
% para la red neural.

fprintf('\nCargando Parámetros de la Red ...\n')

% Cargar pesos en las variables Theta1 and Theta2
load('ej4pesos.mat');

% Desenrolle los parámetros 
params_rn = [Theta1(:) ; Theta2(:)];

%% ============== 3ra Parte: Calcule el Costo (Hacia Adelante) ==============
%  Debe empezar implementando la propagación hacia adelante que retorna el
%  costo nada más. Este código va en funcionDeCostoRN.m, una vez implentada
%  la propagación hacia adelante y el cálculo del costo puede verificar que
%  tiene el mismo costo que el fijado por los parámetrios para debugging.
%
%  Se sugiere implementar primero el costo *sin* regularización, será más
%  facil de depurar. Más tarde, en la cuarta parte, implementará el costo
%  con regularización.
%
fprintf('\nPropagación hacia adelante con la Red Neural ...\n')

% Parámetro de regularización (lo ponemos en 0 aquí).
lambda = 0;

J = funcionDeCostoRN(params_rn, tam_capa_entrada, tam_capa_oculta, ...
                   num_etiquetas, X, y, lambda);

fprintf(['Costo en los parámetros (cargados de pesosEj4): %f '...
         '\n(debería ser como 0.287629)\n'], J);

fprintf('\nPrograma en pausa. Oprima enter para continuar.\n');
% pause;

%% =============== 4ta Parte: Implemente la Regularización ===============
%  Una vez que su función de costo este correcta, debe continuar 
%  implementando el costo con la regularización.
%

fprintf('\nRevisando la Función de Costo (con Regularización) ... \n')

% Parámetros de peso de la regularización (lo ponemos en 1).
lambda = 1;

J = funcionDeCostoRN(params_rn, tam_capa_entrada, tam_capa_oculta, ...
                   num_etiquetas, X, y, lambda);

fprintf(['Costo en los parámetros (carados de pesosEj4): %f '...
         '\n(este valor debe ser como 0.383770)\n'], J);

fprintf('Programa en pausa. Oprima enter para continuar.\n');
% pause;


%% ================ 5ta Parte: Gradiente de Sigmoide  ================
%  Antes de implementar la red neural, primero implementará el 
%  gradiente de la función sigmoide. Debe completar el código en
%  el archivo gradientSimoide.m
%

fprintf('\nEvaluando el gradiente de la función sigmoide...\n')

g = gradienteSigmoide([1 -0.5 0 0.5 1]);
fprintf('Gradiente de la sigmoide evaluado en [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Programa en pausa. Oprima enter para continuar.\n');
% pause;


%% ================ 6ta Parte: Initializar Parámetros ================
%  En esta parte del ejercicios, usted empezará a implementar una red
%  Neural de dos capas que clasifica dígitos. Empezará implementando
%  la función para inicializar los pesos de la red neural
%  (inicializarPesos.m)

fprintf('\nInitializando los Parámetros de la Red Neural ...\n')

Theta1_inicial = inicializarPesos(tam_capa_entrada, tam_capa_oculta);
Theta2_inicial = inicializarPesos(tam_capa_oculta, num_etiquetas);

% Desenrollar parámetros
params_iniciales_rn = [Theta1_inicial(:) ; Theta2_inicial(:)];


%% =============== 7ma Parte: Implementar la Retropropagación ===============
%  Una vez que sus costos son los correctos, debe proceder a implementar 
%  el algoritmo de retropropagación para su red neural. Debe agregar
%  el código en el archivo funcionDeCostoRN.m y retornar las
%  derivadas parciales de los parámetros (pesos)

%
fprintf('\nRevisando la Retropropagación... \n');

%  Revisar los gradientes con reviseGradientesRN
reviseGradientesRN;

fprintf('\nPrograma en pausa. Oprima enter para continuar.\n');
% pause;


%% =============== 8va Parte: Implementar la Regularización ===============
%  Una vez que la implementación de retropropagación esté bien, debe
%  continuar con la implementación de la regularización del costo y del
%  gradiente
%

fprintf('\nRevisando Retropropagación (con/ Regularización) ... \n')

%  Revisar gradientes corriento reviseGradientesRN
lambda = 3;
reviseGradientesRN(lambda);

% También obtenga el valor de la función de costo
J_debug  = funcionDeCostoRN(params_rn, tam_capa_entrada, ...
                          tam_capa_oculta, num_etiquetas, X, y, lambda);

fprintf(['\n\nCosto a parámetros (fijos) de debugging (con/ lambda = 10): %f ' ...
         '\n(este valor debe ser como 0.576051)\n\n'], J_debug);

fprintf('Programa en pausa. Oprima enter para continuar.\n');
% pause;


%% =================== Parte 9: Entrenar la RN ===================
%  Ya impleentó todo el código necesario para entrenar la red
%  neural. Para entrenarla, estará usando  "fmincg", que es una
%  función similar a "fminunc". Estos optimizadores pueden calcular
%  el costo eficientemente siempre y cuando se les provea el cálculo
%  del gradiente
%
fprintf('\nEntrenando a la Red Neural... \n')

%  Después de haber completado la tarea, cambie el valor de MaxIter a
%  un valor más grande para ver como esto ayuda en el entrenamiento

opciones = optimset('MaxIter', 50);

%  También debería probar valores distintos de lambda
lambda = 1;

% Macro para minimizar el costo
funcionDeCosto = @(p) funcionDeCostoRN(p, ...
                                   tam_capa_entrada, ...
                                   tam_capa_oculta, ...
                                   num_etiquetas, X, y, lambda);

% Ahora, funcionDeCosto es una función que toma solo un argumento
% (los parámetros de la red)
[params_rn, costo] = fmincg(funcionDeCosto, params_iniciales_rn, opciones);

% Obtenta Theta1 y Theta2 de params_rn
Theta1 = reshape(params_rn(1:tam_capa_oculta * (tam_capa_entrada + 1)), ...
                 tam_capa_oculta, (tam_capa_entrada + 1));

Theta2 = reshape(params_rn((1 + (tam_capa_oculta * (tam_capa_entrada + 1))):end), ...
                 num_etiquetas, (tam_capa_oculta + 1));

fprintf('Program en pausa. Oprima enter para continuar.\n');
% pause;


%% ================= 10ma Parte: Visualizar Los Pesos =================
%  Usted puede "visualizar" los pesos que aprende la red neural
%  desplegando las unidades ocultas para ver que están capturando
%  de los datos

fprintf('\nVisualizando la Red Neural ... \n')

despliegueDatos(Theta1(:, 2:end));

fprintf('\nPrograma en pausea. Oprima enter para continuar.\n');
% pause;

%% ================= 11va Parte: Implemente la Predicción =================
%  Después de entrenar la red le gustaría utilizarla para la predicción
%  de etiquetas, implementa la función "prediccion" para revisar la
%  precisión de su entrenamiento

pred = prediccion(Theta1, Theta2, X);

fprintf('\nPrecisión el entrenamiento: %f\n', mean(double(pred == y)) * 100);



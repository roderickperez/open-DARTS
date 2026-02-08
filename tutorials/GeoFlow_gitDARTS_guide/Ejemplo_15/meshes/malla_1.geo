// Gmsh project created on Wed Aug 13 14:41:02 2025
SetFactory("OpenCASCADE");

height_res= 100;
rsv_layers=3;
lc=80;
lc_wells=10;

// --- Puntos frontera ---
Point(1) = {200.0, 700.0, 0.0, lc};
Point(2) = {150.0, 850.0, 0.0, lc};
Point(3) = {350.0, 950.0, 0.0, lc};
Point(4) = {500.0, 850.0, 0.0, lc};
Point(5) = {450.0, 750.0, 0.0, lc};
Point(6) = {500.0, 600.0, 0.0, lc};
Point(7) = {750.0, 550.0, 0.0, lc};
Point(8) = {900.0, 400.0, 0.0, lc};
Point(9) = {800.0, 200.0, 0.0, lc};
Point(10) = {600.0, 300.0, 0.0, lc};
Point(11) = {350.0, 350.0, 0.0, lc};
Point(12) = {250.0, 500.0, 0.0, lc};

// --- Curva cerrada (Spline) ---
// IMPORTANTE: incluir el primer punto al final para cerrar
Spline(1) = {1,2,3,4,5,6,7,8,9,10,11,12,1};

// --- Crear un Line Loop con la curva cerrada ---
Line Loop(1) = {1};

// --- Crear superficie a partir del loop ---
Plane Surface(1) = {1};

// --- Pozos (puntos de refinamiento) ---
Point(13) = {350, 800, 0, lc}; // inyector
Point(14) = {750, 300, 0, lc}; // productor

// --- Refinamiento de malla usando campos ---
Field[1] = Distance;
Field[1].NodesList = {13,14};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc_wells;
Field[2].SizeMax = lc;
Field[2].DistMin = 50;
Field[2].DistMax = 300;

// --- Aplicar campo ---
Background Field = 2;



// Altura de cada capa
h_layer = height_res / rsv_layers;

// Extrude surface with embedded features

// --- Extrusión capa 1 ---
sr1[] = Extrude {0, 0, h_layer}{ Surface {1}; Layers{1}; Recombine;};
Physical Volume("matrix_layer1", 9991) = {1};

// --- Extrusión capa 2 ---
sr2[] = Extrude {0,0,h_layer} { Surface{sr1[0]}; Layers{1}; Recombine; };
Physical Volume("matrix_layer2", 9992) = {2};

// --- Extrusión capa 3 ---
sr3[] = Extrude {0,0,h_layer} { Surface{sr2[0]}; Layers{1}; Recombine; };
Physical Volume("matrix_layer3", 9993) = {3};



// Mallado
Mesh 3;
Coherence Mesh;
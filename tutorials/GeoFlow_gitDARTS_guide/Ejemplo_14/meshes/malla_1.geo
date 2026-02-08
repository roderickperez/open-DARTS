// Gmsh project created on Wed Aug 13 14:41:02 2025
SetFactory("OpenCASCADE");

height_res= 100;
rsv_layers=3;
lc=50;
//lc_wells=20;

// Extra points for boundary of domain:
Point(1) = { 0.00000,  0.00000,  0.00000, lc};
Point(2) = {1000.00000,  0.00000,  0.00000, lc};
Point(3) = {1000.00000, 1000.00000,  0.00000, lc};
Point(4) = { 0.00000, 1000.00000,  0.00000, lc};


// Punto donde se desea refinar la malla
Point(5) = {0, 1000, 0, lc}; //  inyector well coordinate 
Point(6) = {1000, 0, 0, lc}; //  producer well coordinate


// --- Campo Distance ---
//Field[1] = Distance;
//Field[1].NodesList = {5,6};  // Referencia al punto donde quieres refinar

// --- Campo Threshold ---
//Field[2] = Threshold;
//Field[2].InField = 1;         // Se basa en el campo Distance
//Field[2].SizeMin = lc_wells; // Tamaño mínimo de malla cerca del punto
//Field[2].SizeMax = lc;       // Tamaño máximo lejos del punto
//Field[2].DistMin = 300;      // Dentro de este radio se aplica SizeMin
//Field[2].DistMax = 1000;     // Después de este radio se aplica SizeMax

// Aplicar campo como campo de fondo
//Background Field = 2;



// Extra lines for boundary of domain:
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


// Create line loop for boundary surface:
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

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

// Horizontal surfaces
Physical Surface(1) = {sr3[0]}; // top
Physical Surface(2) = {sr1[1]}; // bottom

// Caras laterales
Physical Surface(30) = {sr1[2]};
Physical Surface(40) = {sr1[3]};
Physical Surface(50) = {sr1[4]};
Physical Surface(60) = {sr1[5]};
Physical Surface(31) = {sr2[2]};
Physical Surface(41) = {sr2[3]};
Physical Surface(51) = {sr2[4]};
Physical Surface(61) = {sr2[5]};
Physical Surface(32) = {sr3[2]};
Physical Surface(42) = {sr3[3]};
Physical Surface(52) = {sr3[4]};
Physical Surface(62) = {sr3[5]};


// Mallado
Mesh 3;
Coherence Mesh;

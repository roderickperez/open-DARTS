// Gmsh project created on Tue Aug 05 21:50:47 2025
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
//Point(5) = {0, 1000, 0, lc}; //  inyector well coordinate 
//Point(6) = {1000, 0, 0, lc}; //  producer well coordinate

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


// Extrude surface with embedded features

// Reservoir
sr[] = Extrude {0, 0, height_res}{ Surface {1}; Layers{rsv_layers}; Recombine;};

// Horizontal surfaces
Physical Surface(1) = {sr[0]}; // top
Physical Surface(2) = {sr[1]}; // bottom
Physical Surface(3) = {sr[2]}; // Y-
Physical Surface(4) = {sr[3]}; // X+
Physical Surface(5) = {sr[4]}; // Y+
Physical Surface(6) = {sr[5]}; // X-

//Reservoir
Physical Volume("matrix", 9991) = {1};


Mesh 3;  // Generate 3D mesh
Coherence Mesh;  // Remove duplicate entities
//Mesh.MshFileVersion = 2.1;



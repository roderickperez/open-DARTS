// Geo file which meshes the input mesh from act_frac_sys.
// Change mesh-elements size by varying "lc" below.

lc = 80.000;
lc_box = 80.000;
lc_well = 80.000;
height_res = 50.000;

rsv_layers = 1;
overburden_thickness = 0.000;
overburden_layers = 0;
underburden_thickness = 0.000;
underburden_layers = 0;
overburden_2_thickness = 0.000;
overburden_2_layers = 0;
underburden_2_thickness = 0.000;
underburden_2_layers = 0;
Point(1) = {1000.00000, 500.00000,  0.00000, lc};
Point(4) = {3000.00000, 2500.00000,  0.00000, lc};
Line(1) = {1, 4};

Point(2) = {1000.00000, 2000.00000,  0.00000, lc};
Point(3) = {2000.00000, 500.00000,  0.00000, lc};
Line(2) = {2, 3};

num_points_frac = newp - 1;
num_lines_frac = newl - 1;

// Extra points for boundary of domain:
Point(5) = { 0.00000,  0.00000,  0.00000, lc_box};
Point(6) = {4000.00000,  0.00000,  0.00000, lc_box};
Point(7) = {4000.00000, 4000.00000,  0.00000, lc_box};
Point(8) = { 0.00000, 4000.00000,  0.00000, lc_box};

// Extra lines for boundary of domain:
Line(3) = {5, 6};
Line(4) = {6, 7};
Line(5) = {7, 8};
Line(6) = {8, 5};

// Create line loop for boundary surface:
Curve Loop(1) = {3, 4, 5, 6};
Plane Surface(1) = {1};

Curve{1:num_lines_frac} In Surface{1};

// Extrude surface with embedded features

// Reservoir
sr[] = Extrude {0, 0, height_res}{ Surface {1}; Layers{rsv_layers}; Recombine;};
// Horizontal surfaces
Physical Surface(1) = {sr[0]}; // top
Physical Surface(2) = {1}; // bottom
Physical Surface(3) = {sr[2]}; // Y-
Physical Surface(4) = {sr[3]}; // X+
Physical Surface(5) = {sr[4]}; // Y+
Physical Surface(6) = {sr[5]}; // X-

// Extrude fractures

// Fracture {1}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {1}; Layers{rsv_layers}; Recombine;};
Physical Surface(90000) = {news - 1};

// Fracture {2}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {2}; Layers{rsv_layers}; Recombine;};
Physical Surface(90001) = {news - 1};

num_surfaces_before = news;
num_surfaces_after = news - 1;
num_surfaces_fracs = num_surfaces_after - num_surfaces_before;

//Reservoir
Physical Volume("matrix", 9991) = {1};


Mesh 3;  // Generate 3D mesh
Coherence Mesh;  // Remove duplicate entities
//Mesh.MshFileVersion = 2.1;

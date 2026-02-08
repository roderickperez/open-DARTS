// Geo file which meshes the input mesh from act_frac_sys.
// Change mesh-elements size by varying "lc" below.

lc = 200.000;
lc_box = 200.000;
lc_well = 200.000;
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
Point(1) = {1500.00000, 500.00000,  0.00000, lc};
Point(7) = {2000.00000, 500.00000,  0.00000, lc};
Line(1) = {1, 7};

Point(2) = {1500.00000, 1000.00000,  0.00000, lc};
Point(8) = {2000.00000, 1000.00000,  0.00000, lc};
Line(2) = {2, 8};

Point(3) = {1500.00000, 1800.00000,  0.00000, lc};
Point(9) = {2000.00000, 2000.00000,  0.00000, lc};
Line(3) = {3, 9};

Point(4) = {1500.00000, 2200.00000,  0.00000, lc};
Line(4) = {4, 9};

Point(5) = {1500.00000, 2800.00000,  0.00000, lc};
Point(13) = {2500.00000, 2900.00000,  0.00000, lc};
Line(5) = {5, 13};

Point(6) = {1500.00000, 3000.00000,  0.00000, lc};
Point(14) = {2500.00000, 3000.00000,  0.00000, lc};
Line(6) = {6, 14};

Point(10) = {2500.00000, 600.00000,  0.00000, lc};
Line(7) = {7, 10};

Point(11) = {2500.00000, 1300.00000,  0.00000, lc};
Line(8) = {8, 11};

Point(12) = {2500.00000, 2000.00000,  0.00000, lc};
Line(9) = {9, 12};

Point(15) = {3000.00000, 1900.00000,  0.00000, lc};
Line(10) = {12, 15};

Point(16) = {3000.00000, 2100.00000,  0.00000, lc};
Line(11) = {12, 16};

num_points_frac = newp - 1;
num_lines_frac = newl - 1;

// Extra points for boundary of domain:
Point(17) = { 0.00000,  0.00000,  0.00000, lc_box};
Point(18) = {4000.00000,  0.00000,  0.00000, lc_box};
Point(19) = {4000.00000, 4000.00000,  0.00000, lc_box};
Point(20) = { 0.00000, 4000.00000,  0.00000, lc_box};

// Extra lines for boundary of domain:
Line(12) = {17, 18};
Line(13) = {18, 19};
Line(14) = {19, 20};
Line(15) = {20, 17};

// Create line loop for boundary surface:
Curve Loop(1) = {12, 13, 14, 15};
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

// Fracture {3}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {3}; Layers{rsv_layers}; Recombine;};
Physical Surface(90002) = {news - 1};

// Fracture {4}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {4}; Layers{rsv_layers}; Recombine;};
Physical Surface(90003) = {news - 1};

// Fracture {5}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {5}; Layers{rsv_layers}; Recombine;};
Physical Surface(90004) = {news - 1};

// Fracture {6}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {6}; Layers{rsv_layers}; Recombine;};
Physical Surface(90005) = {news - 1};

// Fracture {7}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {7}; Layers{rsv_layers}; Recombine;};
Physical Surface(90006) = {news - 1};

// Fracture {8}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {8}; Layers{rsv_layers}; Recombine;};
Physical Surface(90007) = {news - 1};

// Fracture {9}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {9}; Layers{rsv_layers}; Recombine;};
Physical Surface(90008) = {news - 1};

// Fracture {10}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {10}; Layers{rsv_layers}; Recombine;};
Physical Surface(90009) = {news - 1};

// Fracture {11}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {11}; Layers{rsv_layers}; Recombine;};
Physical Surface(90010) = {news - 1};

num_surfaces_before = news;
num_surfaces_after = news - 1;
num_surfaces_fracs = num_surfaces_after - num_surfaces_before;

//Reservoir
Physical Volume("matrix", 9991) = {1};


Mesh 3;  // Generate 3D mesh
Coherence Mesh;  // Remove duplicate entities
//Mesh.MshFileVersion = 2.1;

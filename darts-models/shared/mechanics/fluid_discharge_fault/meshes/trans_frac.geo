a = 1.0;
lc = 0.05 * a;
lc_frac = 0.01 * a;
l = 0.5 * a;
phi = Pi / 10;
h = 10;
Nx_pt = 128 + 1;
Ny_pt = 128 + 1;
N_frac = 64 + 1;
Nz = 1;

Point(1) = {0, 0, 0, lc};
Point(2) = {a, 0,  0, lc};
Point(3) = {a, a, 0, lc};
Point(4) = {0, a, 0, lc};
Point(5) = {a/2 - l / 2 * Cos(phi), a/2 - l / 2 * Sin(phi), 0, lc_frac};
Point(6) = {a/2 + l / 2 * Cos(phi), a/2 + l / 2 * Sin(phi), 0, lc_frac};
Point(7) = {0, a / 2, 0, lc};
Point(8) = {a, a / 2, 0, lc};

// Bot & centerline
Line(1) = {1,2};
Line(2) = {2,8};
Line(3) = {8,6};
Line(4) = {6,5};
Line(5) = {5,7};
Line(6) = {7,1};
// Top
Line(7) = {8,3};
Line(8) = {3,4};
Line(9) = {4,7};

Transfinite Curve{1} = Nx_pt;
Transfinite Curve{2} = (Ny_pt - 1) / 2 + 1;
Transfinite Curve{3} = (Nx_pt - N_frac) / 2 + 1;
Transfinite Curve{4} = N_frac;
Transfinite Curve{5} = (Nx_pt - N_frac) / 2 + 1;
Transfinite Curve{6} = (Ny_pt - 1) / 2 + 1;
Transfinite Curve{7} = (Ny_pt - 1) / 2 + 1;
Transfinite Curve{8} = Nx_pt;
Transfinite Curve{9} = (Ny_pt - 1) / 2 + 1;

Line Loop(1) = {1,2,3,4,5,6};
Line Loop(2) = {7, 8, 9, -5, -4, -3};
//Line Loop(2) = {5, -5};

//Plane Surface(1) = {1};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Transfinite Surface{1} = {1,2,8,7};
Transfinite Surface{2} = {7,8,3,4};


Recombine Surface{:};
out[] = Extrude {0, 0, h} { 
		Surface{1:2};
		Layers{Nz};
		Recombine;
	};

Physical Volume("matrix", 99991) = {out[1], out[9]};
Physical Surface("boundary_xm", 991) = {out[7], out[12]};
Physical Surface("boundary_xp", 992) = {out[3], out[10]};
Physical Surface("boundary_ym", 993) = {out[2]};
Physical Surface("boundary_yp", 994) = {out[11]};
Physical Surface("boundary_zm", 995) = {1, 2};
Physical Surface("boundary_zp", 996) = {out[0], out[8]};
Physical Surface("fracture", 91) = {out[5]};
Physical Curve("fracture_stuck", 1) = {27, 31};
Physical Curve("fracture_free", 2) = {4, 14};
Mesh 3;
Coherence Mesh;
Mesh.MshFileVersion = 2.1;
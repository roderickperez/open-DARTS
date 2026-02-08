a = 29.9 / 1000.0;
a1 = a / 2;
Nin = 10 + 1;
Nout = 4 + 1;
Nh = 14;//20;
h = 71.5 / 2.0 / 1000.0;
k = Sqrt(2);
Phi = Pi / 3;
lc = 1.0;
lc_frac = 1.0;
inR = Sqrt(a1 ^ 2 / 4 * (k + 1) ^ 2 + a1 ^ 2 / 4);

Point(1) = {0, 0, 0, lc};
Point(2) = {a1 / 2, a1 / 2, 0, lc};
Point(3) = {Sqrt(2) * a / 4, Sqrt(2) * a / 4, 0, lc};
Point(4) = {-Sqrt(2) * a / 4, Sqrt(2) * a / 4, 0, lc};
//Point(5) = {a / 2, 0, 0, lc};
Point(5) = {Sqrt(2) * a / 4, -Sqrt(2) * a / 4, 0, lc};
//Point(6) = {0, Sqrt(a1 ^ 2 / 4 * (k + 1) ^ 2 + a1 ^ 2 / 4) - k * a1 / 2, 0, lc};
Point(6) = {-a1 / 2, a1 / 2, 0, lc};
//Point(7) = {Sqrt(a1 ^ 2 / 4 * (k + 1) ^ 2 + a1 ^ 2 / 4) - k * a1 / 2, 0, 0, lc};
Point(7) = {a1 / 2, -a1 / 2, 0, lc};
Point(8) = {-k * a1 / 2, 0, 0, lc};
Point(9) = {0, -k * a1 / 2, 0, lc};
Point(10) = {k * a1 / 2, 0, 0, lc};
Point(11) = {0, k * a1 / 2, 0, lc};
Point(12) = {-a1 / 2, -a1 / 2, 0, lc};
Point(13) = {-Sqrt(2) * a / 4, -Sqrt(2) * a / 4, 0, lc};

Point(20) = {0, a / 2, 0, lc};
Point(21) = {0, Sqrt(a1 ^ 2 / 4 * (k + 1) ^ 2 + a1 ^ 2 / 4) - k * a1 / 2, 0, lc};
Point(22) = {0, -Sqrt(a1 ^ 2 / 4 * (k + 1) ^ 2 + a1 ^ 2 / 4) + k * a1 / 2, 0, lc};
Point(23) = {0, -a / 2, 0, lc};

Circle(1) = {5, 1, 3};
Circle(21) = {3, 1, 20};
Circle(22) = {20, 1, 4};
Circle(3) = {2, 8, 7};
Circle(41) = {2, 9, 21};
Circle(42) = {21, 9, 6};
Circle(51) = {13, 1, 23};
Circle(52) = {23, 1, 5};
Line(6) = {7, 5};
Circle(7) = {4, 1, 13};
Line(8) = {6, 4};
Line(9) = {2, 3};
//Circle(10) = {4, 1, 5};
Circle(111) = {7, 11, 22};
Circle(112) = {22, 11, 12};
Circle(12) = {12, 10, 6};
Line(13) = {12, 13};

Line(14) = {20, 21};
Line(15) = {21, 22};
Line(16) = {22, 23};

Transfinite Curve{6} = Nout;
Transfinite Curve{8} = Nout;
Transfinite Curve{9} = Nout;
Transfinite Curve{13} = Nout;

Transfinite Curve{1} = Nin;
Transfinite Curve{21} = (Nin - 1) / 2 + 1;
Transfinite Curve{22} = (Nin - 1) / 2 + 1;
Transfinite Curve{51} = (Nin - 1) / 2 + 1;
Transfinite Curve{52} = (Nin - 1) / 2 + 1;
Transfinite Curve{7} = Nin;

Transfinite Curve{3} = Nin;
Transfinite Curve{41} = (Nin - 1) / 2 + 1;
Transfinite Curve{42} = (Nin - 1) / 2 + 1;
Transfinite Curve{111} = (Nin - 1) / 2 + 1;
Transfinite Curve{112} = (Nin - 1) / 2 + 1;
Transfinite Curve{12} = Nin;

Transfinite Curve{14} = Nout;
Transfinite Curve{15} = Nin;
Transfinite Curve{16} = Nout;

Curve Loop(1) = {1, -9, 3, 6};
Curve Loop(2) = {9, 21, 14, -41};
Curve Loop(3) = {-3, 41, 15, -111};
Curve Loop(4) = {6, -52, -16, -111};

Curve Loop(5) = {-12, 13, -7, -8};
Curve Loop(6) = {42, 8, -22, 14};
Curve Loop(7) = {42, -12, -112, -15};
Curve Loop(8) = {112, 13, 51, -16};

Plane Surface(1) = {1}; 
Plane Surface(2) = {2}; 
Plane Surface(3) = {3};
Plane Surface(4) = {4};

Plane Surface(5) = {5};
Plane Surface(6) = {6};
Plane Surface(7) = {7};
Plane Surface(8) = {8};

Transfinite Surface{1} = {5, 3, 2, 7};
Transfinite Surface{2} = {3, 20, 21, 2};
Transfinite Surface{3} = {2, 21, 22, 7};
Transfinite Surface{4} = {5, 7, 22, 23};

Transfinite Surface{5} = {4, 6, 12, 13};
Transfinite Surface{6} = {6, 21, 20, 4};
Transfinite Surface{7} = {12, 22, 21, 6};
Transfinite Surface{8} = {22, 12, 13, 23};

Recombine Surface{:};

out1[] = Extrude {0, 0, h} { 
		Surface{1:8};
		Layers{Nh};
		Recombine;
	};

out2[] = Extrude {0, 0, -h} { 
		Surface{1:8};
		Layers{Nh};
		Recombine;
	};


Physical Volume("matrix1", 99991) = {out1[1], out1[7], out1[13], out1[19]};
Physical Volume("matrix2", 99992) = {out1[25], out1[31], out1[37], out1[43]};
Physical Volume("matrix3", 99993) = {out2[1], out2[7], out2[13], out2[19]};
Physical Volume("matrix4", 99994) = {out2[25], out2[31], out2[37], out2[43]};
Physical Surface("fracture1", 9991) = {out1[16], out1[10], out1[22], out2[16], out2[10], out2[22]};
Physical Curve("fracture1_bot", 11) = {358, 336, 314};
Physical Curve("fracture1_top", 12) = {182, 160, 138};
Physical Curve("fracture1_side", 13) = {146, 322, 190, 366};

//Physical Surface("fracture2", 9992) = {1:8};
//Physical Curve("fracture2_side", 23) = {51, 52, 1, 21, 22, 7};

Physical Surface("bot", 991) = {out2[0], out2[6], out2[12], out2[18], out2[24], out2[30], out2[36], out2[42]};
Physical Surface("top", 992) = {out1[0], out1[6], out1[12], out1[18], out1[24], out1[30], out1[36], out1[42]};
Physical Surface("side", 993) = {out1[2], out1[9], out1[21], out1[28], out1[34], out1[46], out2[2], out2[9], out2[21], out2[28], out2[34], out2[46]};

//Mesh.OptimizeMesh = "HighOrderElastic";
//Smoother Surface{304} = 40; 
Mesh 3;
Coherence Mesh;
Mesh.MshFileVersion = 2.1;//+
//Show "*";

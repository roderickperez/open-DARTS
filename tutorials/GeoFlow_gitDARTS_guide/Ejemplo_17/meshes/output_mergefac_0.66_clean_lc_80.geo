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
Point(4) = {1057.14286, 557.14286,  0.00000, lc};
Line(1) = {1, 4};

Point(6) = {1114.28571, 614.28571,  0.00000, lc};
Line(2) = {4, 6};

Point(8) = {1171.42857, 671.42857,  0.00000, lc};
Line(3) = {6, 8};

Point(11) = {1228.57143, 728.57143,  0.00000, lc};
Line(4) = {8, 11};

Point(13) = {1285.71429, 785.71429,  0.00000, lc};
Line(5) = {11, 13};

Point(15) = {1342.85714, 842.85714,  0.00000, lc};
Line(6) = {13, 15};

Point(18) = {1400.00000, 900.00000,  0.00000, lc};
Line(7) = {15, 18};

Point(20) = {1457.14286, 957.14286,  0.00000, lc};
Line(8) = {18, 20};

Point(22) = {1514.28571, 1014.28571,  0.00000, lc};
Line(9) = {20, 22};

Point(25) = {1571.42857, 1071.42857,  0.00000, lc};
Line(10) = {22, 25};

Point(26) = {1628.57143, 1128.57143,  0.00000, lc};
Line(11) = {25, 26};

Point(28) = {1685.71429, 1185.71429,  0.00000, lc};
Line(12) = {26, 28};

Point(31) = {1742.85714, 1242.85714,  0.00000, lc};
Line(13) = {28, 31};

Point(33) = {1800.00000, 1300.00000,  0.00000, lc};
Line(14) = {31, 33};

Point(35) = {1857.14286, 1357.14286,  0.00000, lc};
Line(15) = {33, 35};

Point(38) = {1914.28571, 1414.28571,  0.00000, lc};
Line(16) = {35, 38};

Point(40) = {1971.42857, 1471.42857,  0.00000, lc};
Line(17) = {38, 40};

Point(42) = {2028.57143, 1528.57143,  0.00000, lc};
Line(18) = {40, 42};

Point(43) = {2085.71429, 1585.71429,  0.00000, lc};
Line(19) = {42, 43};

Point(44) = {2142.85714, 1642.85714,  0.00000, lc};
Line(20) = {43, 44};

Point(45) = {2200.00000, 1700.00000,  0.00000, lc};
Line(21) = {44, 45};

Point(46) = {2257.14286, 1757.14286,  0.00000, lc};
Line(22) = {45, 46};

Point(47) = {2314.28571, 1814.28571,  0.00000, lc};
Line(23) = {46, 47};

Point(48) = {2371.42857, 1871.42857,  0.00000, lc};
Line(24) = {47, 48};

Point(49) = {2428.57143, 1928.57143,  0.00000, lc};
Line(25) = {48, 49};

Point(50) = {2485.71429, 1985.71429,  0.00000, lc};
Line(26) = {49, 50};

Point(51) = {2542.85714, 2042.85714,  0.00000, lc};
Line(27) = {50, 51};

Point(52) = {2600.00000, 2100.00000,  0.00000, lc};
Line(28) = {51, 52};

Point(53) = {2657.14286, 2157.14286,  0.00000, lc};
Line(29) = {52, 53};

Point(54) = {2714.28571, 2214.28571,  0.00000, lc};
Line(30) = {53, 54};

Point(55) = {2771.42857, 2271.42857,  0.00000, lc};
Line(31) = {54, 55};

Point(56) = {2828.57143, 2328.57143,  0.00000, lc};
Line(32) = {55, 56};

Point(57) = {2885.71429, 2385.71429,  0.00000, lc};
Line(33) = {56, 57};

Point(58) = {2942.85714, 2442.85714,  0.00000, lc};
Line(34) = {57, 58};

Point(59) = {3000.00000, 2500.00000,  0.00000, lc};
Line(35) = {58, 59};

Point(2) = {1000.00000, 2000.00000,  0.00000, lc};
Point(3) = {1043.47826, 1934.78261,  0.00000, lc};
Line(36) = {2, 3};

Point(5) = {1086.95652, 1869.56522,  0.00000, lc};
Line(37) = {3, 5};

Point(7) = {1130.43478, 1804.34783,  0.00000, lc};
Line(38) = {5, 7};

Point(9) = {1173.91304, 1739.13043,  0.00000, lc};
Line(39) = {7, 9};

Point(10) = {1217.39130, 1673.91304,  0.00000, lc};
Line(40) = {9, 10};

Point(12) = {1260.86957, 1608.69565,  0.00000, lc};
Line(41) = {10, 12};

Point(14) = {1304.34783, 1543.47826,  0.00000, lc};
Line(42) = {12, 14};

Point(16) = {1347.82609, 1478.26087,  0.00000, lc};
Line(43) = {14, 16};

Point(17) = {1391.30435, 1413.04348,  0.00000, lc};
Line(44) = {16, 17};

Point(19) = {1434.78261, 1347.82609,  0.00000, lc};
Line(45) = {17, 19};

Point(21) = {1478.26087, 1282.60870,  0.00000, lc};
Line(46) = {19, 21};

Point(23) = {1521.73913, 1217.39130,  0.00000, lc};
Line(47) = {21, 23};

Point(24) = {1565.21739, 1152.17391,  0.00000, lc};
Line(48) = {23, 24};

Line(49) = {24, 25};

Point(27) = {1652.17391, 1021.73913,  0.00000, lc};
Line(50) = {25, 27};

Point(29) = {1695.65217, 956.52174,  0.00000, lc};
Line(51) = {27, 29};

Point(30) = {1739.13043, 891.30435,  0.00000, lc};
Line(52) = {29, 30};

Point(32) = {1782.60870, 826.08696,  0.00000, lc};
Line(53) = {30, 32};

Point(34) = {1826.08696, 760.86957,  0.00000, lc};
Line(54) = {32, 34};

Point(36) = {1869.56522, 695.65217,  0.00000, lc};
Line(55) = {34, 36};

Point(37) = {1913.04348, 630.43478,  0.00000, lc};
Line(56) = {36, 37};

Point(39) = {1956.52174, 565.21739,  0.00000, lc};
Line(57) = {37, 39};

Point(41) = {2000.00000, 500.00000,  0.00000, lc};
Line(58) = {39, 41};

num_points_frac = newp - 1;
num_lines_frac = newl - 1;

// Extra points for boundary of domain:
Point(60) = { 0.00000,  0.00000,  0.00000, lc_box};
Point(61) = {4000.00000,  0.00000,  0.00000, lc_box};
Point(62) = {4000.00000, 4000.00000,  0.00000, lc_box};
Point(63) = { 0.00000, 4000.00000,  0.00000, lc_box};

// Extra lines for boundary of domain:
Line(59) = {60, 61};
Line(60) = {61, 62};
Line(61) = {62, 63};
Line(62) = {63, 60};

// Create line loop for boundary surface:
Curve Loop(1) = {59, 60, 61, 62};
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

// Fracture {12}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {12}; Layers{rsv_layers}; Recombine;};
Physical Surface(90011) = {news - 1};

// Fracture {13}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {13}; Layers{rsv_layers}; Recombine;};
Physical Surface(90012) = {news - 1};

// Fracture {14}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {14}; Layers{rsv_layers}; Recombine;};
Physical Surface(90013) = {news - 1};

// Fracture {15}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {15}; Layers{rsv_layers}; Recombine;};
Physical Surface(90014) = {news - 1};

// Fracture {16}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {16}; Layers{rsv_layers}; Recombine;};
Physical Surface(90015) = {news - 1};

// Fracture {17}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {17}; Layers{rsv_layers}; Recombine;};
Physical Surface(90016) = {news - 1};

// Fracture {18}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {18}; Layers{rsv_layers}; Recombine;};
Physical Surface(90017) = {news - 1};

// Fracture {19}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {19}; Layers{rsv_layers}; Recombine;};
Physical Surface(90018) = {news - 1};

// Fracture {20}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {20}; Layers{rsv_layers}; Recombine;};
Physical Surface(90019) = {news - 1};

// Fracture {21}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {21}; Layers{rsv_layers}; Recombine;};
Physical Surface(90020) = {news - 1};

// Fracture {22}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {22}; Layers{rsv_layers}; Recombine;};
Physical Surface(90021) = {news - 1};

// Fracture {23}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {23}; Layers{rsv_layers}; Recombine;};
Physical Surface(90022) = {news - 1};

// Fracture {24}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {24}; Layers{rsv_layers}; Recombine;};
Physical Surface(90023) = {news - 1};

// Fracture {25}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {25}; Layers{rsv_layers}; Recombine;};
Physical Surface(90024) = {news - 1};

// Fracture {26}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {26}; Layers{rsv_layers}; Recombine;};
Physical Surface(90025) = {news - 1};

// Fracture {27}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {27}; Layers{rsv_layers}; Recombine;};
Physical Surface(90026) = {news - 1};

// Fracture {28}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {28}; Layers{rsv_layers}; Recombine;};
Physical Surface(90027) = {news - 1};

// Fracture {29}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {29}; Layers{rsv_layers}; Recombine;};
Physical Surface(90028) = {news - 1};

// Fracture {30}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {30}; Layers{rsv_layers}; Recombine;};
Physical Surface(90029) = {news - 1};

// Fracture {31}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {31}; Layers{rsv_layers}; Recombine;};
Physical Surface(90030) = {news - 1};

// Fracture {32}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {32}; Layers{rsv_layers}; Recombine;};
Physical Surface(90031) = {news - 1};

// Fracture {33}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {33}; Layers{rsv_layers}; Recombine;};
Physical Surface(90032) = {news - 1};

// Fracture {34}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {34}; Layers{rsv_layers}; Recombine;};
Physical Surface(90033) = {news - 1};

// Fracture {35}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {35}; Layers{rsv_layers}; Recombine;};
Physical Surface(90034) = {news - 1};

// Fracture {36}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {36}; Layers{rsv_layers}; Recombine;};
Physical Surface(90035) = {news - 1};

// Fracture {37}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {37}; Layers{rsv_layers}; Recombine;};
Physical Surface(90036) = {news - 1};

// Fracture {38}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {38}; Layers{rsv_layers}; Recombine;};
Physical Surface(90037) = {news - 1};

// Fracture {39}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {39}; Layers{rsv_layers}; Recombine;};
Physical Surface(90038) = {news - 1};

// Fracture {40}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {40}; Layers{rsv_layers}; Recombine;};
Physical Surface(90039) = {news - 1};

// Fracture {41}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {41}; Layers{rsv_layers}; Recombine;};
Physical Surface(90040) = {news - 1};

// Fracture {42}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {42}; Layers{rsv_layers}; Recombine;};
Physical Surface(90041) = {news - 1};

// Fracture {43}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {43}; Layers{rsv_layers}; Recombine;};
Physical Surface(90042) = {news - 1};

// Fracture {44}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {44}; Layers{rsv_layers}; Recombine;};
Physical Surface(90043) = {news - 1};

// Fracture {45}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {45}; Layers{rsv_layers}; Recombine;};
Physical Surface(90044) = {news - 1};

// Fracture {46}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {46}; Layers{rsv_layers}; Recombine;};
Physical Surface(90045) = {news - 1};

// Fracture {47}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {47}; Layers{rsv_layers}; Recombine;};
Physical Surface(90046) = {news - 1};

// Fracture {48}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {48}; Layers{rsv_layers}; Recombine;};
Physical Surface(90047) = {news - 1};

// Fracture {49}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {49}; Layers{rsv_layers}; Recombine;};
Physical Surface(90048) = {news - 1};

// Fracture {50}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {50}; Layers{rsv_layers}; Recombine;};
Physical Surface(90049) = {news - 1};

// Fracture {51}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {51}; Layers{rsv_layers}; Recombine;};
Physical Surface(90050) = {news - 1};

// Fracture {52}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {52}; Layers{rsv_layers}; Recombine;};
Physical Surface(90051) = {news - 1};

// Fracture {53}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {53}; Layers{rsv_layers}; Recombine;};
Physical Surface(90052) = {news - 1};

// Fracture {54}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {54}; Layers{rsv_layers}; Recombine;};
Physical Surface(90053) = {news - 1};

// Fracture {55}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {55}; Layers{rsv_layers}; Recombine;};
Physical Surface(90054) = {news - 1};

// Fracture {56}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {56}; Layers{rsv_layers}; Recombine;};
Physical Surface(90055) = {news - 1};

// Fracture {57}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {57}; Layers{rsv_layers}; Recombine;};
Physical Surface(90056) = {news - 1};

// Fracture {58}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {58}; Layers{rsv_layers}; Recombine;};
Physical Surface(90057) = {news - 1};

num_surfaces_before = news;
num_surfaces_after = news - 1;
num_surfaces_fracs = num_surfaces_after - num_surfaces_before;

//Reservoir
Physical Volume("matrix", 9991) = {1};


Mesh 3;  // Generate 3D mesh
Coherence Mesh;  // Remove duplicate entities
//Mesh.MshFileVersion = 2.1;

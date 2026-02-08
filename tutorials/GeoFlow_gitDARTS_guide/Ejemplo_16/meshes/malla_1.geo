// Gmsh project created on Wed Aug 13 14:41:02 2025
SetFactory("OpenCASCADE");

height_res= 100;
rsv_layers=3;
lc=20;
lc_wells=5;

// --- Puntos frontera ---
Point(1) = {0.0, 0.0, 0.0,  lc};
Point(2) = {0.0, 0.0, 500.0,  lc};
Point(3) = {1000.0, 0.0 , 500.0, lc};
Point(4) = {1000.0, 0.0, 0.0,  lc};


/////////////////

Point(5) = {400.0, 0.0, 0.0,  lc};
Point(6) = {370.0, 0.0, 60.0,  lc};
Point(7) = {340.0, 0.0, 100.0, lc};
Point(8) = {290.0, 0.0, 300.0,  lc};
Point(9) = {260.0, 0.0, 340.0, lc};
Point(10) = {200.0, 0.0, 500.0,  lc};


Point(11) = {410.0, 0.0, 0.0,  lc};
Point(12) = {370.0, 0.0, 100.0,  lc};
Point(13) = {340.0, 0.0, 140.0, lc};
Point(14) = {290.0, 0.0, 340.0,  lc};
Point(15) = {260.0, 0.0, 380.0, lc};
Point(16) = {210.0, 0.0, 500.0,  lc};


/////////////////



Point(17) = {800.0, 0.0, 0.0,  lc};
Point(18) = {760.0, 0.0, 100.0,  lc};
Point(19) = {730.0, 0.0, 140.0, lc};
Point(20) = {660.0, 0.0, 340.0,  lc};
Point(21) = {640.0, 0.0, 380.0, lc};
Point(22) = {600.0, 0.0, 500.0,  lc};

Point(23) = {810.0, 0.0, 0.0,  lc};
Point(24) = {800.0, 0.0, 40.0,  lc};
Point(25) = {750.0, 0.0, 140.0, lc};
Point(26) = {740.0, 0.0, 180.0,  lc};
Point(27) = {610.0, 0.0, 500.0,  lc};


/////////////////

Point(28) = {0.0, 0.0, 60.0,  lc};
Point(29) = {0.0, 0.0, 100.0, lc};
Point(30) = {0.0, 0.0, 300.0,  lc};
Point(31) = {0.0, 0.0, 340.0, lc};

Point(33) = {1000.0, 0.0, 40.0,  lc};
Point(34) = {1000.0, 0.0, 140.0, lc};
Point(35) = {1000.0, 0.0, 180.0,  lc};

////////////////////////



// Punto donde se desea refinar la malla
Point(50) = {200, 0, 200, lc}; //  inyector well coordinate 
Point(60) = {850, 0, 100, lc}; //  producer well coordinate


// --- Campo Distance ---
Field[1] = Distance;
Field[1].NodesList = {50,60};  // Referencia al punto donde quieres refinar

// --- Campo Threshold ---
Field[2] = Threshold;
Field[2].InField = 1;         // Se basa en el campo Distance
Field[2].SizeMin = lc_wells; // Tamaño mínimo de malla cerca del punto
Field[2].SizeMax = lc;       // Tamaño máximo lejos del punto
Field[2].DistMin = 50;      // Dentro de este radio se aplica SizeMin
Field[2].DistMax = 200;     // Después de este radio se aplica SizeMax

// Aplicar campo como campo de fondo
Background Field = 2;




/////////////////

Line(1) = {1, 28};
Line(2) = {28, 6};
Line(3) = {6, 5};
Line(4) = {5, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};


Line(5) = {28, 29};
Line(6) = {29, 7};
Line(7) = {7, 6};
Line(8) = {6, 28};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};

Line(9) = {29, 30};
Line(10) = {30, 8};
Line(11) = {8, 7};
Line(12) = {7, 29};
Curve Loop(3) = {9, 10, 11, 12};
Plane Surface(3) = {3};


Line(13) = {30, 31};
Line(14) = {31, 9};
Line(15) = {9, 8};
Line(16) = {8, 30};
Curve Loop(4) = {13, 14, 15, 16};
Plane Surface(4) = {4};


Line(17) = {31, 2};
Line(18) = {2, 10};
Line(19) = {10, 9};
Line(20) = {9, 31};
Curve Loop(5) = {17, 18, 19, 20};
Plane Surface(5) = {5};

/////////////////


Line(21) = {10, 9};
Line(22) = {9, 8};
Line(23) = {8, 7};
Line(24) = {7, 6};
Line(25) = {6, 5};
Line(26) = {5, 11};
Line(27) = {11, 12};
Line(28) = {12, 13};
Line(29) = {13, 14};
Line(30) = {14, 15};
Line(31) = {15, 16};
Line(32) = {16, 10};
Curve Loop(6) = {21,22,23,24,25,26,27,28,29,30,31,32};
Plane Surface(6) = {6};

/////////////////

Line(33) = {11, 12};
Line(34) = {12, 18};
Line(35) = {18, 17};
Line(36) = {17, 11};
Curve Loop(7) = {33,34,35,36};
Plane Surface(7) = {7};

Line(37) = {12, 13};
Line(38) = {13, 19};
Line(39) = {19, 18};
Line(40) = {18, 12};
Curve Loop(8) = {37,38,39,40};
Plane Surface(8) = {8};

Line(41) = {13, 14};
Line(42) = {14, 20};
Line(43) = {20, 19};
Line(44) = {19, 13};
Curve Loop(9) = {41,42,43,44};
Plane Surface(9) = {9};

Line(45) = {14, 15};
Line(46) = {15, 21};
Line(47) = {21, 20};
Line(48) = {20, 14};
Curve Loop(10) = {45,46,47,48};
Plane Surface(10) = {10};

Line(49) = {15, 16};
Line(50) = {16, 22};
Line(51) = {22, 21};
Line(52) = {21, 15};
Curve Loop(11) = {49,50,51,52};
Plane Surface(11) = {11};


/////////////////

Line(53) = {22, 27};
Line(54) = {27, 26};
Line(55) = {26, 25};
Line(56) = {25, 24};
Line(57) = {24, 23};
Line(58) = {23, 17};
Line(59) = {17, 18};
Line(60) = {18, 19};
Line(61) = {19, 20};
Line(62) = {20, 21};
Line(63) = {21, 22};
Curve Loop(12) = {53,54,55,56,57,58,59,60,61,62,63};
Plane Surface(12) = {12};


/////////////////

Line(64) = {23, 24};
Line(65) = {24, 33};
Line(66) = {33, 4};
Line(67) = {4, 23};
Curve Loop(13) = {64,65,66,67};
Plane Surface(13) = {13};

Line(68) = {24, 25};
Line(69) = {25, 34};
Line(70) = {34, 33};
Line(71) = {33, 24};
Curve Loop(14) = {68,69,70,71};
Plane Surface(14) = {14};

Line(72) = {25, 26};
Line(73) = {26, 35};
Line(74) = {35, 34};
Line(75) = {34, 25};
Curve Loop(15) = {72,73,74,75};
Plane Surface(15) = {15};


Line(76) = {26, 27};
Line(77) = {27, 3};
Line(78) = {3, 35};
Line(79) = {35, 26};
Curve Loop(16) = {76,77,78,79};
Plane Surface(16) = {16};


/////////////////////////////////

// Altura de cada capa
h_layer = height_res;

// Extrude surface with embedded features

// --- Extrusión  ---

sr1[] = Extrude {0, h_layer, 0 }{ Surface {1}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer1", 1) = {1};

sr2[] = Extrude {0, h_layer, 0 }{ Surface {2}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer2", 2) = {2};

sr3[] = Extrude {0, h_layer, 0 }{ Surface {3}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer3", 3) = {3};

sr4[] = Extrude {0, h_layer, 0 }{ Surface {4}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer4", 4) = {4};

sr5[] = Extrude {0, h_layer, 0 }{ Surface {5}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer5", 5) = {5};


sr6[] = Extrude {0, h_layer, 0 }{ Surface {6}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer6", 6) = {6};

sr7[] = Extrude {0, h_layer, 0 }{ Surface {7}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer7", 7) = {7};


sr8[] = Extrude {0, h_layer, 0 }{ Surface {8}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer8", 8) = {8};

sr9[] = Extrude {0, h_layer, 0 }{ Surface {9}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer9", 9) = {9};

sr10[] = Extrude {0, h_layer, 0 }{ Surface {10}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer10", 10) = {10};

sr11[] = Extrude {0, h_layer, 0 }{ Surface {11}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer11", 11) = {11};

sr12[] = Extrude {0, h_layer, 0 }{ Surface {12}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer12", 12) = {12};

sr13[] = Extrude {0, h_layer, 0 }{ Surface {13}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer13", 13) = {13};


sr14[] = Extrude {0, h_layer, 0 }{ Surface {14}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer14", 14) = {14};

sr15[] = Extrude {0, h_layer, 0 }{ Surface {15}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer15", 15) = {15};

sr16[] = Extrude {0, h_layer, 0 }{ Surface {16}; Layers{rsv_layers}; Recombine;};
Physical Volume("matrix_layer16", 16) = {16};


// Mallado
Mesh 3;
Coherence Mesh;
// Geo file which meshes the input mesh from act_frac_sys.
// Change mesh-elements size by varying "lc" below.

lc = 25.000;
lc_box = 25.000;
lc_well = 25.000;
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
Point(1) = {21.00000, 2632.00000,  0.00000, lc};
Point(50) = {773.00000, 2920.00000,  0.00000, lc};
Line(1) = {1, 50};

Point(2) = {31.00000, 1029.00000,  0.00000, lc};
Point(11) = {167.00000, 1399.00000,  0.00000, lc};
Line(2) = {2, 11};

Point(3) = {47.00000, 1333.00000,  0.00000, lc};
Point(52) = {801.00000, 1096.00000,  0.00000, lc};
Line(3) = {3, 52};

Point(4) = {56.00000, 1275.00000,  0.00000, lc};
Point(16) = {208.00000, 1651.00000,  0.00000, lc};
Line(4) = {4, 16};

Point(5) = {81.00000, 1599.00000,  0.00000, lc};
Point(56) = {859.00000, 1770.00000,  0.00000, lc};
Line(5) = {5, 56};

Point(6) = {93.00000, 2272.00000,  0.00000, lc};
Point(17) = {212.00000, 2652.00000,  0.00000, lc};
Line(6) = {6, 17};

Point(7) = {95.00000, 2672.00000,  0.00000, lc};
Point(12) = {195.00000, 2283.00000,  0.00000, lc};
Line(7) = {7, 12};

Point(8) = {104.00000, 1290.00000,  0.00000, lc};
Point(21) = {259.00000, 1655.00000,  0.00000, lc};
Line(8) = {8, 21};

Point(9) = {124.00000, 1392.00000,  0.00000, lc};
Point(23) = {289.00000, 1038.00000,  0.00000, lc};
Line(9) = {9, 23};

Point(10) = {156.00000, 1665.00000,  0.00000, lc};
Point(62) = {900.00000, 1983.00000,  0.00000, lc};
Line(10) = {10, 62};

Point(13) = {200.00000, 2085.00000,  0.00000, lc};
Point(25) = {313.00000, 2470.00000,  0.00000, lc};
Line(11) = {13, 25};

Point(14) = {201.00000, 1520.00000,  0.00000, lc};
Point(26) = {330.00000, 1145.00000,  0.00000, lc};
Line(12) = {14, 26};

Point(15) = {205.00000, 1481.00000,  0.00000, lc};
Point(66) = {990.00000, 1601.00000,  0.00000, lc};
Line(13) = {15, 66};

Point(18) = {214.00000, 2092.00000,  0.00000, lc};
Point(63) = {952.00000, 1799.00000,  0.00000, lc};
Line(14) = {18, 63};

Point(19) = {239.00000, 1865.00000,  0.00000, lc};
Point(73) = {1026.00000, 1999.00000,  0.00000, lc};
Line(15) = {19, 73};

Point(20) = {242.00000, 1021.00000,  0.00000, lc};
Point(71) = {1022.00000, 1158.00000,  0.00000, lc};
Line(16) = {20, 71};

Point(22) = {270.00000, 1368.00000,  0.00000, lc};
Point(72) = {1026.00000, 1629.00000,  0.00000, lc};
Line(17) = {22, 72};

Point(24) = {296.00000, 2846.00000,  0.00000, lc};
Point(28) = {354.00000, 2443.00000,  0.00000, lc};
Line(18) = {24, 28};

Point(27) = {342.00000, 1947.00000,  0.00000, lc};
Point(31) = {449.00000, 1568.00000,  0.00000, lc};
Line(19) = {27, 31};

Point(29) = {385.00000, 1934.00000,  0.00000, lc};
Point(80) = {1171.00000, 1855.00000,  0.00000, lc};
Line(20) = {29, 80};

Point(30) = {444.00000, 2142.00000,  0.00000, lc};
Point(35) = {529.00000, 2534.00000,  0.00000, lc};
Line(21) = {30, 35};

Point(32) = {489.00000, 2072.00000,  0.00000, lc};
Point(39) = {582.00000, 2465.00000,  0.00000, lc};
Line(22) = {32, 39};

Point(33) = {493.00000, 1779.00000,  0.00000, lc};
Point(36) = {540.00000, 1376.00000,  0.00000, lc};
Line(23) = {33, 36};

Point(34) = {493.00000, 1789.00000,  0.00000, lc};
Point(83) = {1240.00000, 1504.00000,  0.00000, lc};
Line(24) = {34, 83};

Point(37) = {544.00000, 2292.00000,  0.00000, lc};
Point(41) = {632.00000, 2676.00000,  0.00000, lc};
Line(25) = {37, 41};

Point(38) = {548.00000, 1626.00000,  0.00000, lc};
Point(46) = {692.00000, 2008.00000,  0.00000, lc};
Line(26) = {38, 46};

Point(40) = {599.00000, 1151.00000,  0.00000, lc};
Point(92) = {1389.00000, 1305.00000,  0.00000, lc};
Line(27) = {40, 92};

Point(42) = {651.00000, 2862.00000,  0.00000, lc};
Point(91) = {1385.00000, 2557.00000,  0.00000, lc};
Line(28) = {42, 91};

Point(43) = {660.00000, 2567.00000,  0.00000, lc};
Point(48) = {725.00000, 2970.00000,  0.00000, lc};
Line(29) = {43, 48};

Point(44) = {661.00000, 1831.00000,  0.00000, lc};
Point(95) = {1455.00000, 1907.00000,  0.00000, lc};
Line(30) = {44, 95};

Point(45) = {661.00000, 1902.00000,  0.00000, lc};
Point(53) = {821.00000, 1540.00000,  0.00000, lc};
Line(31) = {45, 53};

Point(47) = {717.00000, 2811.00000,  0.00000, lc};
Point(59) = {880.00000, 2441.00000,  0.00000, lc};
Line(32) = {47, 59};

Point(49) = {751.00000, 1413.00000,  0.00000, lc};
Point(102) = {1509.00000, 1164.00000,  0.00000, lc};
Line(33) = {49, 102};

Point(51) = {785.00000, 2328.00000,  0.00000, lc};
Point(105) = {1563.00000, 2104.00000,  0.00000, lc};
Line(34) = {51, 105};

Point(54) = {826.00000, 2993.00000,  0.00000, lc};
Point(106) = {1572.00000, 2709.00000,  0.00000, lc};
Line(35) = {54, 106};

Point(55) = {853.00000, 2667.00000,  0.00000, lc};
Point(107) = {1617.00000, 2403.00000,  0.00000, lc};
Line(36) = {55, 107};

Point(57) = {862.00000, 2108.00000,  0.00000, lc};
Point(67) = {993.00000, 1730.00000,  0.00000, lc};
Line(37) = {57, 67};

Point(58) = {870.00000, 2212.00000,  0.00000, lc};
Point(111) = {1652.00000, 2048.00000,  0.00000, lc};
Line(38) = {58, 111};

Point(60) = {883.00000, 2382.00000,  0.00000, lc};
Point(65) = {982.00000, 1995.00000,  0.00000, lc};
Line(39) = {60, 65};

Point(61) = {899.00000, 1065.00000,  0.00000, lc};
Point(68) = {999.00000, 1444.00000,  0.00000, lc};
Line(40) = {61, 68};

Point(64) = {965.00000, 2499.00000,  0.00000, lc};
Point(114) = {1732.00000, 2299.00000,  0.00000, lc};
Line(41) = {64, 114};

Point(69) = {1001.00000, 2190.00000,  0.00000, lc};
Point(78) = {1152.00000, 1813.00000,  0.00000, lc};
Line(42) = {69, 78};

Point(70) = {1013.00000, 2238.00000,  0.00000, lc};
Point(74) = {1102.00000, 2619.00000,  0.00000, lc};
Line(43) = {70, 74};

Point(75) = {1115.00000, 1831.00000,  0.00000, lc};
Point(119) = {1894.00000, 1625.00000,  0.00000, lc};
Line(44) = {75, 119};

Point(76) = {1117.00000, 2373.00000,  0.00000, lc};
Point(85) = {1260.00000, 2753.00000,  0.00000, lc};
Line(45) = {76, 85};

Point(77) = {1126.00000, 2984.00000,  0.00000, lc};
Point(82) = {1238.00000, 2591.00000,  0.00000, lc};
Line(46) = {77, 82};

Point(79) = {1153.00000, 2569.00000,  0.00000, lc};
Point(118) = {1876.00000, 2904.00000,  0.00000, lc};
Line(47) = {79, 118};

Point(81) = {1196.00000, 1298.00000,  0.00000, lc};
Point(87) = {1307.00000, 1686.00000,  0.00000, lc};
Line(48) = {81, 87};

Point(84) = {1243.00000, 2140.00000,  0.00000, lc};
Point(121) = {1978.00000, 1806.00000,  0.00000, lc};
Line(49) = {84, 121};

Point(86) = {1302.00000, 1182.00000,  0.00000, lc};
Point(94) = {1420.00000, 1553.00000,  0.00000, lc};
Line(50) = {86, 94};

Point(88) = {1358.00000, 2684.00000,  0.00000, lc};
Point(99) = {1467.00000, 2302.00000,  0.00000, lc};
Line(51) = {88, 99};

Point(89) = {1360.00000, 1399.00000,  0.00000, lc};
Point(93) = {1405.00000, 1796.00000,  0.00000, lc};
Line(52) = {89, 93};

Point(90) = {1376.00000, 2171.00000,  0.00000, lc};
Point(98) = {1465.00000, 1774.00000,  0.00000, lc};
Line(53) = {90, 98};

Point(96) = {1457.00000, 2272.00000,  0.00000, lc};
Point(141) = {2246.00000, 2124.00000,  0.00000, lc};
Line(54) = {96, 141};

Point(97) = {1463.00000, 1137.00000,  0.00000, lc};
Point(137) = {2195.00000, 1473.00000,  0.00000, lc};
Line(55) = {97, 137};

Point(100) = {1490.00000, 1629.00000,  0.00000, lc};
Point(104) = {1526.00000, 2033.00000,  0.00000, lc};
Line(56) = {100, 104};

Point(101) = {1495.00000, 1640.00000,  0.00000, lc};
Point(139) = {2239.00000, 1342.00000,  0.00000, lc};
Line(57) = {101, 139};

Point(103) = {1524.00000, 2041.00000,  0.00000, lc};
Point(144) = {2321.00000, 1967.00000,  0.00000, lc};
Line(58) = {103, 144};

Point(108) = {1633.00000, 2836.00000,  0.00000, lc};
Point(153) = {2435.00000, 2760.00000,  0.00000, lc};
Line(59) = {108, 153};

Point(109) = {1636.00000, 2247.00000,  0.00000, lc};
Point(149) = {2416.00000, 2107.00000,  0.00000, lc};
Line(60) = {109, 149};

Point(110) = {1646.00000, 2789.00000,  0.00000, lc};
Point(148) = {2375.00000, 2451.00000,  0.00000, lc};
Line(61) = {110, 148};

Point(112) = {1673.00000, 2211.00000,  0.00000, lc};
Point(158) = {2459.00000, 2313.00000,  0.00000, lc};
Line(62) = {112, 158};

Point(113) = {1692.00000, 2793.00000,  0.00000, lc};
Point(159) = {2463.00000, 3000.00000,  0.00000, lc};
Line(63) = {113, 159};

Point(115) = {1770.00000, 2194.00000,  0.00000, lc};
Point(161) = {2509.00000, 2492.00000,  0.00000, lc};
Line(64) = {115, 161};

Point(116) = {1775.00000, 1861.00000,  0.00000, lc};
Point(168) = {2575.00000, 1731.00000,  0.00000, lc};
Line(65) = {116, 168};

Point(117) = {1837.00000, 1586.00000,  0.00000, lc};
Point(167) = {2571.00000, 1886.00000,  0.00000, lc};
Line(66) = {117, 167};

Point(120) = {1939.00000, 1649.00000,  0.00000, lc};
Point(178) = {2721.00000, 1774.00000,  0.00000, lc};
Line(67) = {120, 178};

Point(122) = {1993.00000, 2895.00000,  0.00000, lc};
Point(127) = {2093.00000, 2510.00000,  0.00000, lc};
Line(68) = {122, 127};

Point(123) = {1997.00000, 1762.00000,  0.00000, lc};
Point(180) = {2755.00000, 1515.00000,  0.00000, lc};
Line(69) = {123, 180};

Point(124) = {2038.00000, 2313.00000,  0.00000, lc};
Point(128) = {2126.00000, 1926.00000,  0.00000, lc};
Line(70) = {124, 128};

Point(125) = {2080.00000, 2294.00000,  0.00000, lc};
Point(134) = {2170.00000, 2675.00000,  0.00000, lc};
Line(71) = {125, 134};

Point(126) = {2088.00000, 1718.00000,  0.00000, lc};
Point(135) = {2173.00000, 2108.00000,  0.00000, lc};
Line(72) = {126, 135};

Point(129) = {2126.00000, 2591.00000,  0.00000, lc};
Point(138) = {2222.00000, 2201.00000,  0.00000, lc};
Line(73) = {129, 138};

Point(130) = {2128.00000, 2494.00000,  0.00000, lc};
Point(193) = {2914.00000, 2334.00000,  0.00000, lc};
Line(74) = {130, 193};

Point(131) = {2135.00000, 1343.00000,  0.00000, lc};
Point(190) = {2890.00000, 1617.00000,  0.00000, lc};
Line(75) = {131, 190};

Point(132) = {2136.00000, 1329.00000,  0.00000, lc};
Point(192) = {2913.00000, 1519.00000,  0.00000, lc};
Line(76) = {132, 192};

Point(133) = {2151.00000, 1780.00000,  0.00000, lc};
Point(140) = {2242.00000, 1394.00000,  0.00000, lc};
Line(77) = {133, 140};

Point(136) = {2178.00000, 1070.00000,  0.00000, lc};
Point(142) = {2254.00000, 1469.00000,  0.00000, lc};
Line(78) = {136, 142};

Point(143) = {2282.00000, 1497.00000,  0.00000, lc};
Point(147) = {2363.00000, 1113.00000,  0.00000, lc};
Line(79) = {143, 147};

Point(145) = {2325.00000, 2857.00000,  0.00000, lc};
Point(199) = {3084.00000, 2636.00000,  0.00000, lc};
Line(80) = {145, 199};

Point(146) = {2348.00000, 1300.00000,  0.00000, lc};
Point(163) = {2518.00000, 1673.00000,  0.00000, lc};
Line(81) = {146, 163};

Point(150) = {2423.00000, 2207.00000,  0.00000, lc};
Point(170) = {2579.00000, 2573.00000,  0.00000, lc};
Line(82) = {150, 170};

Point(151) = {2428.00000, 2605.00000,  0.00000, lc};
Point(205) = {3198.00000, 2374.00000,  0.00000, lc};
Line(83) = {151, 205};

Point(152) = {2430.00000, 2957.00000,  0.00000, lc};
Point(164) = {2559.00000, 2574.00000,  0.00000, lc};
Line(84) = {152, 164};

Point(154) = {2437.00000, 2294.00000,  0.00000, lc};
Point(160) = {2499.00000, 2683.00000,  0.00000, lc};
Line(85) = {154, 160};

Point(155) = {2444.00000, 2306.00000,  0.00000, lc};
Point(169) = {2578.00000, 1930.00000,  0.00000, lc};
Line(86) = {155, 169};

Point(156) = {2449.00000, 1935.00000,  0.00000, lc};
Point(165) = {2565.00000, 1543.00000,  0.00000, lc};
Line(87) = {156, 165};

Point(157) = {2454.00000, 1711.00000,  0.00000, lc};
Point(206) = {3237.00000, 1912.00000,  0.00000, lc};
Line(88) = {157, 206};

Point(162) = {2512.00000, 1396.00000,  0.00000, lc};
Point(166) = {2571.00000, 1788.00000,  0.00000, lc};
Line(89) = {162, 166};

Point(171) = {2592.00000, 1973.00000,  0.00000, lc};
Point(211) = {3389.00000, 2087.00000,  0.00000, lc};
Line(90) = {171, 211};

Point(172) = {2619.00000, 2614.00000,  0.00000, lc};
Point(174) = {2674.00000, 2214.00000,  0.00000, lc};
Line(91) = {172, 174};

Point(173) = {2645.00000, 2550.00000,  0.00000, lc};
Point(212) = {3419.00000, 2387.00000,  0.00000, lc};
Line(92) = {173, 212};

Point(175) = {2685.00000, 1253.00000,  0.00000, lc};
Point(213) = {3465.00000, 1069.00000,  0.00000, lc};
Line(93) = {175, 213};

Point(176) = {2685.00000, 1578.00000,  0.00000, lc};
Point(215) = {3477.00000, 1741.00000,  0.00000, lc};
Line(94) = {176, 215};

Point(177) = {2705.00000, 2735.00000,  0.00000, lc};
Point(216) = {3482.00000, 2570.00000,  0.00000, lc};
Line(95) = {177, 216};

Point(179) = {2725.00000, 2750.00000,  0.00000, lc};
Point(217) = {3501.00000, 2521.00000,  0.00000, lc};
Line(96) = {179, 217};

Point(181) = {2763.00000, 1928.00000,  0.00000, lc};
Point(183) = {2802.00000, 1523.00000,  0.00000, lc};
Line(97) = {181, 183};

Point(182) = {2787.00000, 1567.00000,  0.00000, lc};
Point(220) = {3534.00000, 1260.00000,  0.00000, lc};
Line(98) = {182, 220};

Point(184) = {2808.00000, 2877.00000,  0.00000, lc};
Point(221) = {3571.00000, 2606.00000,  0.00000, lc};
Line(99) = {184, 221};

Point(185) = {2814.00000, 2332.00000,  0.00000, lc};
Point(191) = {2900.00000, 1950.00000,  0.00000, lc};
Line(100) = {185, 191};

Point(186) = {2833.00000, 1976.00000,  0.00000, lc};
Point(225) = {3640.00000, 1904.00000,  0.00000, lc};
Line(101) = {186, 225};

Point(187) = {2862.00000, 1504.00000,  0.00000, lc};
Point(222) = {3592.00000, 1176.00000,  0.00000, lc};
Line(102) = {187, 222};

Point(188) = {2862.00000, 2487.00000,  0.00000, lc};
Point(194) = {2916.00000, 2096.00000,  0.00000, lc};
Line(103) = {188, 194};

Point(189) = {2863.00000, 2699.00000,  0.00000, lc};
Point(198) = {2940.00000, 2315.00000,  0.00000, lc};
Line(104) = {189, 198};

Point(195) = {2923.00000, 1479.00000,  0.00000, lc};
Point(227) = {3692.00000, 1277.00000,  0.00000, lc};
Line(105) = {195, 227};

Point(196) = {2927.00000, 2732.00000,  0.00000, lc};
Point(228) = {3705.00000, 2920.00000,  0.00000, lc};
Line(106) = {196, 228};

Point(197) = {2936.00000, 2166.00000,  0.00000, lc};
Point(229) = {3729.00000, 2073.00000,  0.00000, lc};
Line(107) = {197, 229};

Point(200) = {3104.00000, 1195.00000,  0.00000, lc};
Point(232) = {3822.00000, 1526.00000,  0.00000, lc};
Line(108) = {200, 232};

Point(201) = {3107.00000, 1754.00000,  0.00000, lc};
Point(204) = {3180.00000, 2137.00000,  0.00000, lc};
Line(109) = {201, 204};

Point(202) = {3109.00000, 1983.00000,  0.00000, lc};
Point(239) = {3910.00000, 1888.00000,  0.00000, lc};
Line(110) = {202, 239};

Point(203) = {3134.00000, 1565.00000,  0.00000, lc};
Point(238) = {3909.00000, 1348.00000,  0.00000, lc};
Line(111) = {203, 238};

Point(207) = {3250.00000, 1455.00000,  0.00000, lc};
Point(240) = {3991.00000, 1766.00000,  0.00000, lc};
Line(112) = {207, 240};

Point(208) = {3342.00000, 1805.00000,  0.00000, lc};
Point(210) = {3376.00000, 2197.00000,  0.00000, lc};
Line(113) = {208, 210};

Point(209) = {3345.00000, 2691.00000,  0.00000, lc};
Point(218) = {3506.00000, 2319.00000,  0.00000, lc};
Line(114) = {209, 218};

Point(214) = {3468.00000, 2484.00000,  0.00000, lc};
Point(223) = {3597.00000, 2853.00000,  0.00000, lc};
Line(115) = {214, 223};

Point(219) = {3520.00000, 2295.00000,  0.00000, lc};
Point(224) = {3609.00000, 1901.00000,  0.00000, lc};
Line(116) = {219, 224};

Point(226) = {3681.00000, 1674.00000,  0.00000, lc};
Point(233) = {3822.00000, 2047.00000,  0.00000, lc};
Line(117) = {226, 233};

Point(230) = {3777.00000, 2837.00000,  0.00000, lc};
Point(236) = {3849.00000, 2438.00000,  0.00000, lc};
Line(118) = {230, 236};

Point(231) = {3783.00000, 2254.00000,  0.00000, lc};
Point(235) = {3842.00000, 1858.00000,  0.00000, lc};
Line(119) = {231, 235};

Point(234) = {3840.00000, 1265.00000,  0.00000, lc};
Point(237) = {3902.00000, 1659.00000,  0.00000, lc};
Line(120) = {234, 237};

num_points_frac = newp - 1;
num_lines_frac = newl - 1;

// Extra points for boundary of domain:
Point(241) = { 0.00000, 900.00000,  0.00000, lc_box};
Point(242) = {4000.00000, 900.00000,  0.00000, lc_box};
Point(243) = {4000.00000, 3100.00000,  0.00000, lc_box};
Point(244) = { 0.00000, 3100.00000,  0.00000, lc_box};

// Extra lines for boundary of domain:
Line(121) = {241, 242};
Line(122) = {242, 243};
Line(123) = {243, 244};
Line(124) = {244, 241};

// Create line loop for boundary surface:
Curve Loop(1) = {121, 122, 123, 124};
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

// Fracture {59}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {59}; Layers{rsv_layers}; Recombine;};
Physical Surface(90058) = {news - 1};

// Fracture {60}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {60}; Layers{rsv_layers}; Recombine;};
Physical Surface(90059) = {news - 1};

// Fracture {61}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {61}; Layers{rsv_layers}; Recombine;};
Physical Surface(90060) = {news - 1};

// Fracture {62}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {62}; Layers{rsv_layers}; Recombine;};
Physical Surface(90061) = {news - 1};

// Fracture {63}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {63}; Layers{rsv_layers}; Recombine;};
Physical Surface(90062) = {news - 1};

// Fracture {64}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {64}; Layers{rsv_layers}; Recombine;};
Physical Surface(90063) = {news - 1};

// Fracture {65}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {65}; Layers{rsv_layers}; Recombine;};
Physical Surface(90064) = {news - 1};

// Fracture {66}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {66}; Layers{rsv_layers}; Recombine;};
Physical Surface(90065) = {news - 1};

// Fracture {67}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {67}; Layers{rsv_layers}; Recombine;};
Physical Surface(90066) = {news - 1};

// Fracture {68}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {68}; Layers{rsv_layers}; Recombine;};
Physical Surface(90067) = {news - 1};

// Fracture {69}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {69}; Layers{rsv_layers}; Recombine;};
Physical Surface(90068) = {news - 1};

// Fracture {70}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {70}; Layers{rsv_layers}; Recombine;};
Physical Surface(90069) = {news - 1};

// Fracture {71}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {71}; Layers{rsv_layers}; Recombine;};
Physical Surface(90070) = {news - 1};

// Fracture {72}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {72}; Layers{rsv_layers}; Recombine;};
Physical Surface(90071) = {news - 1};

// Fracture {73}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {73}; Layers{rsv_layers}; Recombine;};
Physical Surface(90072) = {news - 1};

// Fracture {74}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {74}; Layers{rsv_layers}; Recombine;};
Physical Surface(90073) = {news - 1};

// Fracture {75}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {75}; Layers{rsv_layers}; Recombine;};
Physical Surface(90074) = {news - 1};

// Fracture {76}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {76}; Layers{rsv_layers}; Recombine;};
Physical Surface(90075) = {news - 1};

// Fracture {77}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {77}; Layers{rsv_layers}; Recombine;};
Physical Surface(90076) = {news - 1};

// Fracture {78}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {78}; Layers{rsv_layers}; Recombine;};
Physical Surface(90077) = {news - 1};

// Fracture {79}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {79}; Layers{rsv_layers}; Recombine;};
Physical Surface(90078) = {news - 1};

// Fracture {80}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {80}; Layers{rsv_layers}; Recombine;};
Physical Surface(90079) = {news - 1};

// Fracture {81}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {81}; Layers{rsv_layers}; Recombine;};
Physical Surface(90080) = {news - 1};

// Fracture {82}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {82}; Layers{rsv_layers}; Recombine;};
Physical Surface(90081) = {news - 1};

// Fracture {83}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {83}; Layers{rsv_layers}; Recombine;};
Physical Surface(90082) = {news - 1};

// Fracture {84}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {84}; Layers{rsv_layers}; Recombine;};
Physical Surface(90083) = {news - 1};

// Fracture {85}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {85}; Layers{rsv_layers}; Recombine;};
Physical Surface(90084) = {news - 1};

// Fracture {86}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {86}; Layers{rsv_layers}; Recombine;};
Physical Surface(90085) = {news - 1};

// Fracture {87}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {87}; Layers{rsv_layers}; Recombine;};
Physical Surface(90086) = {news - 1};

// Fracture {88}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {88}; Layers{rsv_layers}; Recombine;};
Physical Surface(90087) = {news - 1};

// Fracture {89}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {89}; Layers{rsv_layers}; Recombine;};
Physical Surface(90088) = {news - 1};

// Fracture {90}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {90}; Layers{rsv_layers}; Recombine;};
Physical Surface(90089) = {news - 1};

// Fracture {91}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {91}; Layers{rsv_layers}; Recombine;};
Physical Surface(90090) = {news - 1};

// Fracture {92}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {92}; Layers{rsv_layers}; Recombine;};
Physical Surface(90091) = {news - 1};

// Fracture {93}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {93}; Layers{rsv_layers}; Recombine;};
Physical Surface(90092) = {news - 1};

// Fracture {94}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {94}; Layers{rsv_layers}; Recombine;};
Physical Surface(90093) = {news - 1};

// Fracture {95}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {95}; Layers{rsv_layers}; Recombine;};
Physical Surface(90094) = {news - 1};

// Fracture {96}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {96}; Layers{rsv_layers}; Recombine;};
Physical Surface(90095) = {news - 1};

// Fracture {97}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {97}; Layers{rsv_layers}; Recombine;};
Physical Surface(90096) = {news - 1};

// Fracture {98}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {98}; Layers{rsv_layers}; Recombine;};
Physical Surface(90097) = {news - 1};

// Fracture {99}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {99}; Layers{rsv_layers}; Recombine;};
Physical Surface(90098) = {news - 1};

// Fracture {100}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {100}; Layers{rsv_layers}; Recombine;};
Physical Surface(90099) = {news - 1};

// Fracture {101}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {101}; Layers{rsv_layers}; Recombine;};
Physical Surface(90100) = {news - 1};

// Fracture {102}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {102}; Layers{rsv_layers}; Recombine;};
Physical Surface(90101) = {news - 1};

// Fracture {103}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {103}; Layers{rsv_layers}; Recombine;};
Physical Surface(90102) = {news - 1};

// Fracture {104}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {104}; Layers{rsv_layers}; Recombine;};
Physical Surface(90103) = {news - 1};

// Fracture {105}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {105}; Layers{rsv_layers}; Recombine;};
Physical Surface(90104) = {news - 1};

// Fracture {106}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {106}; Layers{rsv_layers}; Recombine;};
Physical Surface(90105) = {news - 1};

// Fracture {107}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {107}; Layers{rsv_layers}; Recombine;};
Physical Surface(90106) = {news - 1};

// Fracture {108}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {108}; Layers{rsv_layers}; Recombine;};
Physical Surface(90107) = {news - 1};

// Fracture {109}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {109}; Layers{rsv_layers}; Recombine;};
Physical Surface(90108) = {news - 1};

// Fracture {110}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {110}; Layers{rsv_layers}; Recombine;};
Physical Surface(90109) = {news - 1};

// Fracture {111}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {111}; Layers{rsv_layers}; Recombine;};
Physical Surface(90110) = {news - 1};

// Fracture {112}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {112}; Layers{rsv_layers}; Recombine;};
Physical Surface(90111) = {news - 1};

// Fracture {113}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {113}; Layers{rsv_layers}; Recombine;};
Physical Surface(90112) = {news - 1};

// Fracture {114}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {114}; Layers{rsv_layers}; Recombine;};
Physical Surface(90113) = {news - 1};

// Fracture {115}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {115}; Layers{rsv_layers}; Recombine;};
Physical Surface(90114) = {news - 1};

// Fracture {116}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {116}; Layers{rsv_layers}; Recombine;};
Physical Surface(90115) = {news - 1};

// Fracture {117}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {117}; Layers{rsv_layers}; Recombine;};
Physical Surface(90116) = {news - 1};

// Fracture {118}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {118}; Layers{rsv_layers}; Recombine;};
Physical Surface(90117) = {news - 1};

// Fracture {119}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {119}; Layers{rsv_layers}; Recombine;};
Physical Surface(90118) = {news - 1};

// Fracture {120}
// Reservoir layers
fr[] = Extrude {0, 0, height_res}{ Line {120}; Layers{rsv_layers}; Recombine;};
Physical Surface(90119) = {news - 1};

num_surfaces_before = news;
num_surfaces_after = news - 1;
num_surfaces_fracs = num_surfaces_after - num_surfaces_before;

//Reservoir
Physical Volume("matrix", 9991) = {1};


Mesh 3;  // Generate 3D mesh
Coherence Mesh;  // Remove duplicate entities
//Mesh.MshFileVersion = 2.1;

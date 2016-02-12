from pandas import DataFrame, Series, pivot_table
import numpy as np
import pickle, copy, datetime
import pandas as pd

date = datetime.date(year=2015, month=8, day=29)

hr = [76, 76, 76, 79, 83, 86, 89, 95, 100, 103, 108, 111, 114, 117, 120, 117, 120, 117, 114, 115, 112, 115, 118, 122, 122, 126, 130, 127, 124, 121, 127, 130, 134, 137, 139, 143, 145, 148, 151, 152, 112, 113, 110, 107, 107, 108, 111, 112, 112, 113, 112, 114, 113, 115, 118, 117, 114, 111, 108, 104, 99, 96, 91, 87, 81, 80, 77, 82, 83, 84, 88, 91, 95, 98, 103, 105, 108, 116, 119, 116, 113, 115, 118, 123, 126, 129, 133, 134, 135, 130, 130, 130, 129, 129, 129, 131, 131, 128, 125, 128, 128, 129, 131, 133, 129, 127, 119, 119, 120, 120, 120, 115, 116, 117, 114, 114, 117, 120, 121, 119, 117, 119, 120, 121, 120, 118, 120, 119, 119, 119, 121, 122, 125, 122, 118, 115, 114, 114, 117, 114, 114, 118, 121, 125, 128, 127, 128, 131, 131, 134, 131, 130, 128, 129, 129, 126, 123, 122, 120, 118, 115, 116, 113, 113, 112, 110, 114, 115, 115, 113, 113, 116, 115, 115, 119, 118, 117, 114, 114, 110, 107, 104, 101, 98, 89, 85, 80, 81, 82, 85, 88, 94, 97, 102, 108, 108, 111, 116, 119, 121, 125, 128, 131, 133, 135, 137, 133, 130, 124, 120, 120, 127, 127, 127, 127, 130, 130, 130, 130, 132, 132, 133, 132, 132, 134, 134, 127, 123, 123, 120, 117, 114, 111, 105, 102, 99, 102, 105, 106, 110, 113, 115, 117, 116, 113, 116, 118, 120, 123, 123, 124, 126, 126, 126, 126, 126, 127, 129, 130, 130, 128, 128, 127, 127, 126, 127, 128, 129, 130, 131, 133, 132, 133, 130, 130, 128, 127, 126, 128, 130, 132, 133, 132, 131, 132, 129, 126, 129, 132, 133, 135, 135, 135, 136, 135, 136, 138, 138, 138, 138, 139, 140, 141, 139, 140, 137, 137, 137, 137, 139, 142, 141, 139, 136, 133, 133, 132, 131, 130, 128, 132, 134, 134, 135, 134, 131, 129, 128, 129, 126, 123, 120, 117, 114, 111, 106, 103, 100, 108, 111, 108, 105, 101, 98, 100, 103, 106, 108, 112, 114, 112, 112, 112, 113, 122, 124, 125, 125, 125, 128, 130, 129, 131, 133, 136, 136, 134, 135, 132, 135, 134, 131, 128, 127, 129, 130, 129, 129, 130, 131, 132, 132, 133, 136, 135, 135, 133, 132, 134, 134, 131, 131, 132, 131, 131, 130, 131, 133, 133, 134, 137, 136, 133, 128, 128, 127, 126, 126, 133, 136, 134, 131, 130, 128, 126, 123, 124, 122, 119, 118, 115, 112, 109, 105, 98, 94, 94, 91, 88, 83, 80, 84, 87, 90, 95, 98, 101, 105, 108, 113, 123, 123, 125, 125, 125, 126, 127, 128, 128, 128, 131, 134, 137, 142, 146, 149, 148, 147, 147, 145, 145, 142, 141, 141, 137, 136, 135, 137, 140, 139, 139, 140, 143, 143, 140, 140, 137, 137, 138, 137, 135, 133, 127, 127, 126, 129, 131, 133, 133, 132, 132, 133, 133, 130, 130, 129, 129, 128, 128, 128, 129, 133, 134, 134, 133, 133, 134, 131, 128, 125, 122, 120, 122, 123, 123, 118, 118, 118, 116, 118, 122, 125, 128, 131, 132, 133, 133, 131, 135, 136, 137, 137, 136, 136, 133, 134, 136, 137, 136, 135, 134, 137, 138, 141, 141, 141, 142, 142, 142, 141, 141, 140, 140, 140, 139, 138, 139, 140, 141, 142, 145, 145, 147, 140, 140, 135, 135, 132, 131, 133, 133, 133, 137, 141, 141, 141, 145, 148, 149, 150, 153, 157, 158, 159, 160, 161, 161, 161, 163, 164, 164, 165, 166, 167, 167, 167, 167, 167, 168, 167, 167, 168, 173, 173, 173, 173, 173, 173, 173, 172, 173, 173, 173, 173, 176, 176, 173, 172, 172, 173, 174, 176, 176, 176, 177, 177, 177, 177, 176, 176, 176, 175, 176, 182, 182, 182, 182, 181, 170, 170, 170, 166, 164, 161, 158, 157, 155, 153, 153, 152, 149, 146, 143, 142, 142, 141, 143, 141, 140, 139, 139, 139, 140, 142, 144, 145, 145, 144, 145, 142, 140, 140, 140, 139, 141, 143, 143, 144, 145, 145, 140, 139, 142, 140, 140, 138, 139, 141, 144, 144, 145, 148, 148, 146, 146, 145, 142, 141, 141, 141, 140, 137, 138, 139, 138, 134, 132, 135, 136, 135, 135, 137, 139, 140, 140, 140, 142, 143, 143, 143, 143, 142, 141, 138, 135, 132, 127, 126, 122, 119, 116, 115, 119, 122, 124, 125, 126, 125, 124, 125, 126, 126, 124, 121, 119, 115, 114, 111, 110, 108, 104, 100, 97, 100, 100, 100, 98, 101, 102, 104, 107, 110, 115, 118, 120, 123, 126, 122, 122, 122, 122, 121, 121, 120, 123, 124, 127, 130, 130, 132, 132, 132, 129, 128, 129, 130, 131, 134, 131, 130, 127, 126, 123, 121, 118, 115, 111, 110]

distance = [0.0, 0.0, 0.0, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 12.5, 15.4, 15.4, 15.4, 20.2, 33.3, 41.1, 42.7, 50.6, 71.8, 76.6, 91.1, 108.4, 111.3, 114.1, 119.9, 134.8, 153.6, 153.6, 153.6, 173.6, 186.5, 193.5, 215.5, 238.1, 257.8, 277.8, 287.8, 310.4, 330.5, 348.7, 351.6, 357.5, 366.4, 387.2, 407.7, 423.5, 445.2, 466.7, 488.4, 509.9, 530.7, 552.4, 573.6, 594.9, 615.6, 637.8, 648.0, 651.1, 651.1, 651.1, 651.1, 651.1, 651.1, 651.1, 652.5, 662.6, 666.8, 669.8, 678.5, 703.8, 703.8, 714.2, 723.7, 735.6, 741.8, 752.7, 773.5, 786.5, 796.1, 809.0, 832.7, 855.3, 858.7, 880.9, 899.6, 906.0, 928.0, 950.3, 956.2, 959.3, 981.0, 1002.9, 1005.7, 1026.2, 1048.2, 1053.5, 1057.3, 1076.4, 1088.6, 1110.1, 1133.3, 1155.8, 1177.0, 1192.4, 1213.1, 1216.3, 1219.6, 1234.0, 1247.9, 1269.9, 1280.8, 1286.8, 1289.7, 1294.5, 1296.6, 1303.7, 1321.4, 1342.0, 1363.8, 1385.1, 1407.3, 1419.7, 1443.7, 1466.7, 1488.4, 1509.1, 1530.2, 1550.4, 1573.8, 1597.0, 1612.0, 1630.2, 1633.9, 1653.5, 1661.3, 1684.8, 1707.8, 1718.5, 1725.3, 1747.3, 1765.4, 1780.6, 1784.3, 1791.8, 1814.6, 1837.8, 1848.9, 1870.2, 1881.7, 1897.2, 1920.1, 1943.3, 1966.3, 1973.1, 1984.2, 2004.7, 2027.0, 2048.2, 2070.8, 2077.6, 2101.8, 2124.2, 2128.0, 2148.2, 2170.7, 2174.1, 2196.0, 2218.0, 2239.1, 2262.9, 2273.6, 2294.8, 2301.9, 2303.9, 2325.0, 2345.5, 2352.5, 2355.3, 2355.9, 2356.3, 2357.2, 2357.3, 2358.3, 2358.5, 2359.3, 2360.0, 2380.3, 2392.1, 2402.8, 2420.3, 2423.8, 2436.9, 2444.4, 2450.2, 2452.9, 2470.3, 2480.4, 2490.7, 2510.9, 2523.6, 2539.1, 2556.2, 2578.7, 2598.8, 2621.5, 2633.6, 2638.5, 2650.0, 2656.7, 2678.9, 2690.0, 2693.4, 2714.6, 2735.3, 2743.4, 2765.0, 2789.3, 2811.5, 2831.7, 2854.2, 2869.7, 2890.5, 2911.0, 2931.8, 2944.4, 2949.4, 2955.2, 2956.2, 2959.8, 2964.2, 2965.3, 2968.8, 2969.2, 2969.2, 2971.9, 2983.6, 3004.6, 3025.0, 3040.4, 3056.8, 3077.4, 3099.5, 3122.0, 3132.8, 3140.6, 3163.2, 3185.8, 3204.8, 3219.8, 3240.1, 3263.2, 3284.0, 3305.0, 3316.5, 3338.3, 3361.6, 3382.4, 3403.5, 3426.0, 3449.5, 3471.0, 3493.9, 3517.3, 3539.4, 3559.5, 3582.2, 3604.7, 3625.1, 3646.6, 3669.3, 3690.3, 3712.3, 3727.9, 3749.0, 3770.1, 3793.6, 3813.7, 3835.3, 3856.0, 3879.4, 3900.5, 3921.1, 3942.0, 3965.7, 3988.9, 4011.0, 4029.2, 4032.8, 4054.4, 4076.1, 4098.0, 4122.0, 4143.9, 4167.7, 4188.2, 4211.5, 4231.9, 4254.4, 4277.0, 4300.1, 4323.6, 4335.9, 4356.2, 4377.5, 4396.9, 4420.4, 4442.3, 4462.4, 4485.6, 4493.6, 4514.0, 4537.4, 4541.1, 4564.1, 4585.6, 4606.2, 4629.3, 4649.5, 4673.1, 4695.0, 4717.8, 4740.1, 4761.1, 4781.4, 4798.3, 4819.1, 4829.4, 4837.3, 4850.6, 4867.4, 4867.4, 4867.7, 4868.3, 4869.2, 4870.0, 4870.1, 4872.0, 4872.2, 4872.6, 4873.4, 4874.8, 4878.0, 4884.7, 4889.4, 4892.7, 4900.4, 4911.0, 4925.9, 4945.9, 4968.2, 4988.5, 5011.1, 5031.3, 5038.2, 5059.8, 5073.6, 5095.3, 5119.0, 5140.4, 5162.7, 5183.8, 5205.5, 5223.0, 5243.5, 5251.4, 5272.9, 5293.6, 5297.4, 5315.6, 5337.3, 5343.7, 5361.7, 5375.2, 5399.0, 5419.7, 5428.9, 5440.7, 5450.5, 5458.0, 5465.2, 5474.7, 5482.8, 5490.2, 5496.7, 5498.3, 5508.8, 5531.5, 5553.7, 5562.8, 5567.8, 5591.1, 5611.4, 5634.6, 5656.9, 5677.5, 5693.7, 5706.9, 5710.0, 5726.8, 5748.5, 5771.6, 5792.4, 5798.5, 5802.0, 5824.1, 5844.1, 5865.4, 5868.4, 5872.1, 5894.2, 5903.3, 5923.6, 5944.5, 5965.9, 5983.7, 6006.7, 6022.3, 6025.6, 6028.6, 6032.0, 6040.9, 6043.4, 6044.9, 6046.4, 6047.2, 6048.2, 6053.2, 6064.7, 6069.0, 6075.4, 6075.4, 6079.9, 6081.1, 6096.1, 6110.3, 6119.5, 6122.6, 6126.2, 6127.0, 6148.1, 6151.5, 6173.8, 6196.7, 6217.5, 6240.0, 6263.1, 6276.6, 6294.7, 6315.3, 6326.4, 6346.9, 6361.6, 6385.1, 6397.4, 6416.8, 6418.9, 6437.7, 6443.0, 6464.9, 6467.2, 6478.5, 6480.3, 6491.4, 6496.6, 6518.6, 6531.6, 6554.7, 6568.2, 6581.1, 6595.3, 6618.0, 6635.2, 6655.7, 6676.4, 6679.2, 6691.7, 6692.4, 6707.4, 6723.9, 6741.8, 6763.1, 6766.7, 6770.2, 6780.5, 6792.5, 6814.5, 6836.9, 6859.8, 6881.5, 6903.6, 6926.2, 6949.2, 6959.6, 6980.7, 7001.1, 7024.0, 7044.8, 7065.7, 7086.0, 7108.1, 7121.7, 7142.4, 7143.9, 7163.9, 7185.8, 7207.9, 7225.2, 7239.0, 7250.2, 7252.8, 7275.9, 7297.5, 7310.8, 7321.4, 7329.8, 7333.8, 7356.5, 7378.5, 7399.0, 7414.7, 7422.2, 7441.6, 7459.9, 7483.1, 7505.9, 7529.0, 7550.6, 7565.1, 7587.0, 7609.8, 7632.1, 7653.6, 7674.9, 7691.1, 7711.4, 7733.3, 7754.7, 7777.0, 7800.3, 7821.1, 7831.8, 7854.2, 7869.0, 7892.6, 7914.9, 7938.2, 7959.0, 7979.3, 8000.4, 8021.1, 8043.1, 8047.4, 8068.6, 8090.2, 8113.5, 8134.9, 8156.7, 8181.1, 8203.1, 8221.7, 8243.6, 8265.2, 8267.7, 8269.2, 8273.5, 8275.1, 8277.6, 8298.5, 8320.2, 8344.1, 8366.6, 8372.9, 8379.1, 8385.5, 8391.5, 8415.0, 8419.8, 8425.2, 8446.8, 8463.4, 8473.6, 8478.4, 8499.4, 8505.4, 8511.6, 8528.1, 8538.5, 8562.6, 8586.5, 8592.6, 8603.1, 8619.8, 8646.4, 8653.3, 8676.2, 8698.5, 8708.8, 8732.2, 8743.7, 8766.7, 8776.5, 8793.9, 8800.3, 8811.9, 8833.1, 8857.6, 8881.1, 8901.2, 8911.4, 8935.2, 8956.2, 8961.4, 8971.3, 8992.1, 9012.9, 9032.7, 9053.8, 9075.7, 9096.3, 9107.5, 9112.6, 9136.6, 9142.7, 9148.0, 9171.9, 9194.8, 9200.6, 9222.2, 9236.8, 9246.3, 9262.0, 9282.4, 9298.4, 9302.0, 9306.2, 9311.4, 9317.0, 9334.2, 9338.1, 9358.8, 9382.7, 9404.5, 9425.3, 9437.8, 9460.2, 9481.1, 9501.7, 9524.9, 9545.0, 9563.7, 9583.0, 9602.6, 9624.9, 9646.1, 9657.9, 9678.1, 9687.0, 9696.9, 9718.2, 9726.7, 9747.9, 9770.1, 9790.6, 9812.6, 9833.5, 9854.6, 9875.4, 9890.7, 9901.0, 9908.9, 9914.6, 9921.7, 9933.6, 9954.3, 9975.9, 9998.9, 10022.4, 10046.2, 10066.2, 10080.5, 10102.4, 10114.0, 10136.2, 10159.7, 10179.7, 10203.4, 10227.4, 10244.1, 10266.3, 10287.8, 10296.3, 10316.5, 10336.6, 10359.4, 10381.1, 10391.4, 10412.4, 10434.3, 10457.9, 10480.1, 10499.9, 10519.9, 10536.8, 10543.9, 10554.5, 10576.1, 10585.6, 10608.8, 10630.4, 10638.0, 10658.6, 10679.7, 10691.2, 10712.5, 10720.5, 10744.0, 10766.4, 10789.9, 10794.1, 10815.6, 10821.6, 10830.1, 10830.3, 10832.3, 10835.8, 10850.3, 10852.3, 10869.7, 10877.4, 10895.9, 10904.0, 10917.9, 10927.5, 10947.6, 10968.2, 10989.7, 11010.6, 11033.0, 11053.6, 11075.2, 11097.4, 11122.4, 11132.2, 11156.4, 11175.2, 11180.7, 11188.1, 11188.1, 11189.7, 11192.9, 11193.2, 11193.5, 11208.8, 11218.1, 11238.4, 11241.6, 11253.8, 11267.0, 11278.0, 11280.9, 11294.7, 11298.2, 11309.9, 11330.7, 11339.2, 11352.1, 11364.2, 11368.3, 11392.0, 11413.5, 11421.9, 11443.4, 11466.3, 11485.0, 11506.8, 11525.2, 11545.3, 11568.4, 11592.1, 11612.4, 11636.2, 11652.0, 11674.7, 11695.2, 11719.5, 11739.7, 11757.4, 11773.8, 11775.3, 11796.1, 11817.7, 11831.2, 11839.2, 11841.5, 11843.5, 11847.3, 11850.1]

time = [0, 7, 8, 16, 17, 20, 23, 26, 32, 34, 85, 87, 92, 96, 108, 140, 151, 165, 182, 195, 202, 214, 220, 221, 222, 224, 229, 243, 249, 262, 276, 280, 282, 289, 296, 302, 308, 311, 318, 325, 331, 332, 334, 337, 344, 351, 356, 363, 370, 377, 384, 391, 398, 405, 412, 419, 426, 434, 442, 444, 446, 448, 449, 450, 451, 452, 457, 459, 461, 465, 470, 471, 473, 475, 478, 480, 483, 489, 493, 496, 500, 507, 513, 514, 521, 527, 529, 536, 543, 545, 546, 553, 561, 562, 568, 576, 579, 582, 588, 592, 598, 605, 611, 617, 621, 627, 628, 629, 633, 637, 643, 646, 648, 651, 655, 656, 658, 662, 667, 672, 678, 684, 688, 695, 701, 706, 711, 717, 723, 729, 735, 739, 744, 745, 750, 752, 758, 764, 767, 769, 775, 780, 784, 785, 787, 793, 799, 802, 808, 811, 815, 821, 828, 834, 837, 843, 851, 857, 863, 870, 872, 879, 885, 886, 891, 897, 898, 904, 910, 915, 921, 924, 930, 933, 935, 942, 949, 952, 955, 960, 966, 972, 974, 976, 980, 983, 987, 1001, 1005, 1008, 1013, 1014, 1018, 1021, 1023, 1024, 1030, 1033, 1036, 1042, 1046, 1051, 1056, 1062, 1067, 1073, 1077, 1079, 1083, 1085, 1091, 1094, 1095, 1101, 1109, 1111, 1116, 1121, 1127, 1133, 1140, 1145, 1152, 1158, 1164, 1169, 1172, 1178, 1179, 1183, 1191, 1193, 1201, 1202, 1204, 1211, 1217, 1222, 1227, 1231, 1235, 1240, 1245, 1251, 1254, 1256, 1262, 1268, 1273, 1277, 1283, 1289, 1295, 1301, 1304, 1310, 1316, 1322, 1328, 1334, 1341, 1347, 1353, 1360, 1366, 1371, 1377, 1383, 1388, 1394, 1400, 1406, 1412, 1416, 1422, 1428, 1435, 1441, 1447, 1453, 1460, 1465, 1470, 1476, 1483, 1490, 1495, 1500, 1501, 1507, 1513, 1519, 1525, 1531, 1537, 1542, 1548, 1553, 1559, 1565, 1571, 1577, 1581, 1587, 1593, 1598, 1604, 1609, 1614, 1620, 1622, 1627, 1633, 1634, 1640, 1646, 1651, 1657, 1662, 1668, 1673, 1679, 1685, 1692, 1698, 1703, 1710, 1714, 1717, 1721, 1727, 1731, 1733, 1736, 1738, 1743, 1744, 1752, 1755, 1757, 1764, 1766, 1768, 1772, 1774, 1776, 1779, 1782, 1786, 1791, 1797, 1802, 1807, 1812, 1814, 1821, 1825, 1832, 1839, 1844, 1850, 1856, 1862, 1867, 1872, 1874, 1879, 1885, 1886, 1891, 1897, 1899, 1904, 1908, 1915, 1921, 1924, 1929, 1933, 1936, 1939, 1943, 1948, 1953, 1958, 1959, 1964, 1973, 1980, 1983, 1985, 1992, 1998, 2005, 2013, 2019, 2024, 2028, 2029, 2035, 2043, 2050, 2056, 2058, 2059, 2065, 2070, 2076, 2077, 2079, 2090, 2094, 2100, 2106, 2113, 2118, 2123, 2128, 2129, 2130, 2131, 2135, 2138, 2139, 2140, 2141, 2142, 2148, 2162, 2167, 2191, 2195, 2228, 2238, 2247, 2252, 2256, 2259, 2264, 2265, 2273, 2274, 2280, 2286, 2292, 2298, 2304, 2307, 2312, 2318, 2323, 2328, 2331, 2336, 2339, 2344, 2345, 2353, 2355, 2364, 2365, 2369, 2370, 2374, 2376, 2384, 2388, 2397, 2402, 2407, 2412, 2420, 2425, 2431, 2438, 2439, 2445, 2446, 2455, 2463, 2470, 2477, 2478, 2479, 2482, 2486, 2494, 2501, 2507, 2513, 2519, 2525, 2531, 2534, 2540, 2546, 2553, 2559, 2566, 2573, 2580, 2584, 2592, 2593, 2599, 2605, 2612, 2619, 2628, 2636, 2638, 2647, 2653, 2659, 2663, 2665, 2666, 2672, 2678, 2683, 2687, 2689, 2694, 2699, 2705, 2711, 2717, 2723, 2727, 2733, 2739, 2745, 2751, 2757, 2761, 2766, 2772, 2777, 2783, 2790, 2796, 2799, 2806, 2810, 2816, 2822, 2828, 2833, 2838, 2843, 2848, 2853, 2854, 2859, 2864, 2870, 2876, 2882, 2888, 2892, 2896, 2902, 2912, 2914, 2915, 2918, 2919, 2921, 2926, 2930, 2934, 2938, 2939, 2940, 2941, 2942, 2946, 2947, 2948, 2952, 2955, 2957, 2958, 2962, 2963, 2964, 2967, 2969, 2973, 2977, 2978, 2980, 2983, 2987, 2988, 2992, 2996, 2998, 3003, 3005, 3009, 3011, 3014, 3015, 3017, 3021, 3025, 3029, 3033, 3035, 3039, 3043, 3044, 3046, 3050, 3054, 3058, 3062, 3066, 3070, 3072, 3073, 3078, 3079, 3080, 3084, 3088, 3089, 3093, 3096, 3098, 3101, 3105, 3108, 3109, 3110, 3112, 3116, 3122, 3123, 3128, 3133, 3139, 3145, 3149, 3156, 3163, 3169, 3175, 3182, 3188, 3193, 3200, 3208, 3215, 3219, 3226, 3229, 3233, 3244, 3247, 3254, 3263, 3271, 3279, 3287, 3294, 3304, 3313, 3324, 3332, 3336, 3341, 3349, 3361, 3368, 3374, 3380, 3386, 3391, 3395, 3401, 3404, 3410, 3416, 3421, 3427, 3433, 3437, 3442, 3447, 3449, 3454, 3459, 3465, 3471, 3474, 3480, 3487, 3495, 3500, 3505, 3511, 3517, 3522, 3528, 3535, 3538, 3545, 3551, 3553, 3559, 3565, 3568, 3573, 3575, 3581, 3587, 3591, 3592, 3599, 3601, 3605, 3606, 3608, 3611, 3626, 3628, 3633, 3635, 3640, 3643, 3648, 3651, 3655, 3659, 3663, 3668, 3674, 3679, 3685, 3690, 3695, 3697, 3702, 3707, 3709, 3712, 3713, 3716, 3723, 3724, 3727, 3739, 3743, 3752, 3757, 3761, 3765, 3769, 3770, 3774, 3775, 3778, 3783, 3785, 3788, 3791, 3792, 3798, 3806, 3810, 3816, 3822, 3827, 3833, 3838, 3844, 3850, 3856, 3861, 3867, 3871, 3877, 3882, 3887, 3892, 3897, 3903, 3904, 3911, 3919, 3926, 3933, 3936, 3940, 3949, 3962]

altitude = [33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3, 33.4, 33.7, 33.8, 34.0, 34.5, 36.8, 34.8, 35.4, 35.9, 36.2, 36.6, 37.8, 38.6, 39.2, 39.8, 40.4, 41.8, 43.1, 43.3, 44.6, 45.7, 46.0, 47.3, 48.6, 49.0, 49.1, 50.3, 51.6, 51.7, 52.6, 52.7, 52.6, 52.8, 53.1, 52.7, 52.8, 54.6, 54.1, 53.3, 52.0, 51.8, 51.9, 52.0, 51.5, 50.9, 50.3, 50.0, 49.9, 49.9, 49.7, 49.6, 49.3, 48.8, 48.4, 47.8, 47.4, 47.0, 47.0, 46.4, 45.8, 45.6, 39.4, 30.7, 27.4, 16.9, 13.5, 13.2, 13.1, 13.0, 12.1, 11.9, 8.0, 7.0, 7.9, 9.0, 14.2, 15.1, 17.7, 18.0, 19.9, 22.5, 24.9, 26.0, 28.4, 33.2, 41.3, 45.3, 45.6, 45.9, 45.8, 44.3, 38.0, 34.9, 33.1, 30.9, 30.1, 27.4, 25.0, 24.7, 22.4, 20.1, 20.0, 19.3, 18.6, 18.3, 17.9, 17.6, 16.6, 15.9, 15.8, 17.2, 18.0, 18.1, 18.0, 18.0, 17.9, 17.9, 17.9, 17.9, 17.9, 17.9, 17.9, 18.3, 18.4, 18.7, 19.2, 19.4, 19.9, 20.1, 20.2, 20.4, 21.1, 20.8, 20.2, 20.0, 19.4, 16.1, 13.6, 13.2, 13.3, 13.8, 14.7, 14.5, 15.0, 15.1, 13.3, 12.9, 13.2, 15.1, 15.2, 16.1, 14.8, 10.7, 11.3, 7.5, 7.5, 7.9, 7.6, 7.7, 7.6, 7.6, 7.8, 10.3, 11.0, 12.5, 13.2, 13.4, 13.5, 13.6, 13.7, 14.3, 12.8, 11.6, 11.4, 11.0, 10.8, 10.1, 10.1, 8.8, 8.8, 8.8, 8.6, 8.7, 8.8, 8.9, 8.8, 9.0, 9.7, 10.8, 11.0, 12.3, 12.5, 12.3, 12.0, 11.5, 10.8, 10.9, 11.1, 11.2, 11.3, 11.2, 11.0, 11.4, 10.4, 10.6, 11.0, 9.7, 9.7, 9.6, 9.7, 9.9, 10.0, 9.6, 9.5, 12.9, 14.5, 14.6, 14.8, 15.0, 15.3, 15.1, 15.1, 15.1, 15.1, 15.0, 15.3, 15.7, 16.3, 17.0, 17.4, 17.9, 18.3, 19.0, 19.4, 19.9, 20.2, 20.1, 20.1, 19.9, 19.7, 19.5, 19.2, 19.0, 18.8, 18.4, 18.2, 18.1, 17.7, 18.1, 23.6, 22.5, 21.5, 16.9, 18.0, 17.2, 14.6, 14.5, 14.7, 14.9, 15.4, 15.7, 15.6, 15.9, 16.0, 16.4, 12.6, 12.6, 12.6, 12.6, 12.6, 12.6, 12.6, 12.5, 12.5, 12.5, 12.6, 12.6, 13.2, 15.4, 15.4, 15.4, 15.7, 17.1, 18.1, 19.0, 19.1, 18.2, 16.8, 16.7, 16.4, 17.1, 20.9, 25.8, 24.5, 20.8, 17.8, 15.8, 15.8, 16.0, 16.4, 16.5, 16.9, 17.4, 17.6, 17.6, 18.1, 18.1, 17.8, 18.1, 16.9, 16.1, 16.2, 18.4, 20.1, 21.3, 23.7, 24.5, 25.4, 28.4, 30.7, 31.1, 34.1, 36.1, 36.0, 36.9, 37.3, 37.0, 35.3, 34.7, 32.6, 32.7, 31.7, 36.2, 36.7, 39.2, 39.6, 38.0, 36.1, 36.0, 36.2, 33.6, 32.7, 31.1, 31.6, 31.3, 23.6, 21.5, 18.3, 18.1, 18.5, 19.1, 20.0, 20.2, 20.2, 19.8, 19.6, 19.8, 19.6, 19.6, 19.5, 19.4, 19.4, 19.2, 18.8, 18.7, 18.6, 18.6, 18.7, 18.7, 19.5, 19.9, 20.1, 20.1, 20.2, 20.2, 19.6, 19.4, 18.7, 18.3, 18.2, 17.8, 18.0, 19.7, 23.7, 27.8, 30.1, 33.3, 33.5, 35.4, 36.0, 37.3, 37.2, 38.2, 38.6, 40.1, 39.5, 36.6, 36.2, 34.9, 34.2, 33.6, 32.8, 35.9, 35.3, 33.7, 35.7, 37.0, 34.2, 31.7, 32.1, 32.4, 30.9, 30.5, 26.0, 22.3, 18.5, 16.0, 16.1, 16.1, 16.2, 16.6, 17.0, 17.4, 17.2, 16.9, 16.8, 16.4, 16.0, 16.0, 15.9, 15.8, 16.0, 17.5, 18.7, 18.4, 16.5, 15.9, 14.9, 15.1, 15.2, 16.1, 17.1, 18.4, 18.6, 18.5, 18.4, 17.2, 16.6, 16.7, 13.9, 12.6, 12.8, 16.1, 15.5, 15.6, 15.6, 15.4, 14.9, 14.7, 14.5, 14.5, 14.6, 14.8, 15.0, 15.4, 14.9, 15.1, 16.8, 17.8, 18.1, 18.4, 19.2, 20.5, 22.6, 19.6, 19.8, 19.9, 20.1, 20.2, 20.6, 21.1, 22.3, 21.8, 21.6, 22.4, 27.5, 23.4, 22.2, 20.8, 20.1, 18.2, 17.1, 16.5, 15.8, 15.4, 15.1, 15.1, 15.2, 15.1, 15.1, 15.1, 15.1, 15.0, 15.1, 15.2, 15.2, 14.9, 14.9, 14.9, 15.0, 15.0, 15.1, 15.1, 15.2, 15.0, 15.0, 14.9, 14.8, 14.7, 14.7, 14.7, 14.4, 14.4, 13.7, 10.5, 10.5, 11.0, 11.6, 12.6, 12.1, 12.0, 12.1, 12.0, 12.0, 12.2, 12.9, 13.5, 15.1, 13.1, 12.1, 12.1, 12.1, 12.6, 12.3, 11.6, 10.7, 11.0, 11.1, 11.1, 10.9, 10.7, 10.9, 10.6, 10.8, 11.7, 11.8, 11.9, 12.2, 12.4, 12.7, 14.3, 13.4, 12.5, 12.7, 13.0, 13.1, 13.5, 13.6, 14.8, 15.4, 15.6, 17.1, 14.4, 9.1, 7.9, 7.5, 8.0, 11.4, 10.9, 10.8, 11.2, 11.0, 10.9, 11.4, 12.0, 12.0, 10.9, 11.4, 12.1, 13.0, 13.7, 16.4, 18.1, 20.4, 21.9, 22.0, 24.4, 26.4, 28.4, 30.7, 32.8, 34.6, 37.0, 40.1, 42.5, 44.5, 45.3, 45.3, 45.4, 45.4, 43.6, 32.7, 26.4, 24.0, 22.5, 20.9, 16.3, 14.4, 13.7, 7.4, 7.1, 9.2, 12.3, 13.4, 15.9, 13.7, 14.3, 21.5, 28.0, 35.2, 45.4, 45.5, 46.0, 46.5, 46.9, 47.3, 46.8, 48.4, 48.6, 48.6, 48.7, 49.2, 49.4, 50.3, 51.3, 51.6, 52.1, 51.0, 51.6, 52.3, 53.9, 53.4, 52.8, 52.9, 52.7, 52.9, 52.8, 52.8, 52.8, 52.7, 52.6, 52.7, 52.7, 52.8, 52.8, 52.1, 51.6, 50.7, 50.2, 49.1, 48.0, 46.9, 45.9, 44.9, 43.9, 42.7, 42.0, 40.9, 39.8, 37.4, 36.1, 35.8, 35.5, 35.4, 35.4, 35.2, 35.2, 35.2, 34.4, 34.0, 33.2, 33.2, 32.8, 32.3, 32.0, 32.0, 31.7, 31.7, 31.5, 31.2, 31.1, 30.9, 30.7, 30.7, 30.2, 29.9, 29.9, 29.6, 29.4, 29.3, 29.0, 28.9, 28.7, 28.5, 28.4, 28.1, 28.0, 27.9, 27.6, 27.4, 27.3, 27.2, 27.2, 27.1, 27.1, 27.8, 27.2, 27.7, 27.7, 27.6, 27.6, 27.5, 27.4]

# assembling data into usable df
def assemble(date, activityId, hrList, distanceList, timeList, altitudeList, latlngList):
    speeds = []
    timeDeltas = []
    distDeltas = []
    altDeltas = []
    for i in range(1,len(timeList)):
        timeSegment = float(timeList[i] - timeList[i-1])
        distanceCovered = float(distanceList[i] - distanceList[i-1])
        speedEntry = distanceCovered / timeSegment
        altDeltaEntry = float(altitudeList[i] - altitudeList[i-1])       
        altDeltas.append(altDeltaEntry)
        speeds.append(speedEntry)
        timeDeltas.append(timeSegment)
        distDeltas.append(distanceCovered)            
    hrL = hrList[1:len(hrList)]     
    latlngL = latlngList[1:len(latlngList)]  
    timeL = timeList[1:len(timeList)]
    date = [date]*len(timeL)

    # assembling
    df = DataFrame({'date':date, "latlng":latlngL, "activityId":activityId, 'hr': hrL, 'speeds':speeds, 'time': timeL, 'timeDeltas':timeDeltas, 'distDeltas':distDeltas, 'altDeltas':altDeltas}) 
     
    # adding smoothed speeds
    df['speedsSmoothed2'] = pd.rolling_mean(df.speeds,2)
    df['speedsSmoothed3'] = pd.rolling_mean(df.speeds,3)
    
    # adding percentange AND absolute speed changes
    df['speedDelta2'] = df.speeds.pct_change(periods=2)    
    df['speedDelta3'] = df.speeds.pct_change(periods=3)  
    s = [np.nan]*3 + list(df.speeds[:-3])
    df['speedDelta3Abs'] = df.speeds - s   
    df['speedDelta5'] = df.speeds.pct_change(periods=5)
    df['speedDelta8'] = df.speeds.pct_change(periods=8)
    
    # Calculating a normalizer column (a scale, i.e. hr 145 = 1.0) to yield load / Real Miles calculation. 
    easyHr = 145
    minHr = 80
    threshhold = 170     
    # Defining a function to be mapped onto df.hr
    def normalizer(x):
        if (x <= easyHr) and (x >= minHr):
            slope = 1.0 / (easyHr - minHr)
            a = -slope*minHr
            normalizer = a + slope*x # a number centered on 1
            x = normalizer
            return x
        if (x > easyHr):
            slope = 1.0 / (threshhold - easyHr) 
            a = -slope*easyHr
            normalizer = a + slope*x + 1
            x = normalizer
            return x      
    # mapping      
    normalizer = df.hr.map(normalizer)
    df['normalizer'] = normalizer

    # Real Dist / normalized load 
    df['realDistInd'] = df.normalizer * df.distDeltas
    df['realDistCum'] = np.cumsum(df.realDistInd)    
    df['distCum'] = np.cumsum(df.distDeltas)    
    return df

#df = assemble(date, hr, distance, time, altitude) 
#print(df)

# takes in client and outputs master df with all activities appended end to end. It checks to see which activities have already been uploaded, only appending new entries
def masterAssemble(client):
    activities = list(client.get_activities())
    athlete = client.get_athlete()
    # add in name of run

    # add 'if' statement to account for new users
    inFile = open("stravaChimp/master_dfs/"+str(athlete.id)+"masterDf.txt", 'r')
    df = pickle.load(inFile)
    inFile.close()

    #df = DataFrame({})
    
    for i in range(len(activities)):
        if float(activities[i].id) not in list(df.activityId):
            activityId = activities[i].id
            run = client.get_activity_streams(activityId, types=['time','latlng','distance','heartrate','altitude',])
            latlng = run['latlng'].data
            time = run['time'].data
            distance = run['distance'].data
            heartrate = run['heartrate'].data
            altitude = run['altitude'].data
            date = activities[i].start_date_local 
            activity = activityId   
            dfi = assemble(date, activityId, heartrate, distance, time, altitude, latlng)
            df = df.append(dfi)    
            print(dfi)          
    return df
     

# making summary df
def getSummaryDf(df):
    sumDf = pivot_table(df, ['timeDeltas','realDistInd', 'distDeltas'], 'date', aggfunc=np.sum) 
    dates = sumDf.index
    summaryDf = DataFrame({})
    for i in range(len(dates)):
        run = df[df.date == dates[i]]
        climb = getClimb(run)
        recovery = getRecovery(run)
        #easySpeed = np.round(getEasySpeed(run),2)# fitness proxy. fix this
        totalDist = np.sum(run.distDeltas)
        realMiles = np.round(np.sum(run.realDistInd), 0)
        totalTime = np.round(np.sum(run.timeDeltas), 0)
        dateTime = dates[i]
        date = datetime.datetime(dateTime.year, dateTime.month, dateTime.day)
        variation = np.round(getHrVar(run), 0)
        avgHr = np.round(getHrAvg(run), 0)    
        activityId = run.activityId[0]
        summary = DataFrame({'date':date,'activityId':activityId, 'avgHr':avgHr,'realMiles':realMiles, 'totalDist':totalDist, 'totalTime':totalTime, 'variation': variation, 'recovery':recovery, 'climb':climb, 'impulse':getImpulse(run), 'stamina':getStamina(run), 'easy':getEasy(run), 'recovery':getRecovery(run)}, index=[date])
        summaryDf = summaryDf.append(summary)
        
    return(summaryDf)

# making df for rolling figures
def getRollingSummaryDf(summaryDf):
    start = min(summaryDf.index)
    end = datetime.datetime.now()
    dates = DataFrame({'date':pd.date_range(start, end)})
    fullSummary = pd.merge(dates, summaryDf, how='outer',on=['date'])    
        
    rollingDistL = []   
    rollingImpulseL = []
    rollingRecL = [] 
    rollingEasyL = []
    rollingStaminaL = []  
    for i in range(len(fullSummary)):
        aWeekPrevious = fullSummary.date[i] - datetime.timedelta(days=7)
        previous7Days = fullSummary[(fullSummary.date > aWeekPrevious) & (fullSummary.date <= fullSummary.date[i])]
        
        rollingDist = np.sum(previous7Days.totalDist)
        rollingDistL.append(rollingDist)

        rollingImpulse = np.sum(previous7Days.impulse)
        rollingImpulseL.append(rollingImpulse)
        
        rollingRec = np.sum(previous7Days.recovery)
        rollingRecL.append(rollingRec)
        
        rollingEasy = np.sum(previous7Days.easy)
        rollingEasyL.append(rollingEasy)
        
        rollingStamina = np.sum(previous7Days.stamina)
        rollingStaminaL.append(rollingStamina)

    rolling = DataFrame({'date':fullSummary.date, 'rollDist':rollingDistL, 'rollImp':rollingImpulseL, 'rollRec':rollingRecL, 'rollEasy':rollingEasyL, 'rollStam':rollingStaminaL})
    
    return rolling

# making json for testing purposes
#testJson = df.loc[:,['hr','speeds', 'altDeltas', 'speedDelta3', 'speedDelta3Abs', 'speedsSmoothed2', 'realDistCum', 'time']].to_json(orient="records")
#print(testJson)

def histogram(df): # returns histogram of time spent at each hr
     hist = pivot_table(df, ['timeDeltas'], 'hr', aggfunc=np.sum)
     hist['hr'] = hist.index
     hist['timeSmoothed'] = pd.rolling_mean(hist.timeDeltas, 2)
     return hist
     
#print(histogram(df).to_json(orient="records"))     
#fitLine = pivot_table(df2, ['speeds'], 'hr', aggfunc=np.mean) 

#hrTime_json = df.hr.to_json()

#openFile = open("hrTime_json.txt", 'w')
#pickle.dump(hrTime_json, openFile)
#openFile.close()


def getFitLine(df): # returns one-column df with avg speed by hr (index)
    df2 = copy.deepcopy(df)
    counter = 1
    while counter < len(df2)-6:
        diff = np.sqrt((df2.speeds[counter] - df2.speeds[counter-1])**2) 
        percentDiff = diff / (df2.speeds[counter-1] + .00000001)    
        if percentDiff > .3:
            df2.speeds[counter] = np.nan
            df2.speeds[counter-1] = np.nan
            df2.speeds[counter+1] = np.nan
            df2.speeds[counter+2] = np.nan
            df2.speeds[counter+3] = np.nan
            df2.speeds[counter+4] = np.nan
            counter += 6
        else: counter += 1
    # only keeping entries where altDelta less than 1, speed greater than 1, distDelta greater than 1, and change in speed less than threshhold   
    df2 = df2[(np.sqrt(df2.altDeltas**2) < 1.0) & (df2.speeds > 1.0)]    
    # making pivot table to consolidate by hr
    fitLine = pivot_table(df2, ['speeds'], 'hr', aggfunc=np.mean) 
    fitLine = DataFrame({'hr':fitLine.index, 'avgSpeed':fitLine.speeds})   
    return fitLine
    
def getFitDf(df):  # returns df w fit-cleaned data
    df2 = copy.deepcopy(df)
    counter = 1
    while counter < len(df2)-6:
        diff = np.sqrt((df2.speeds[counter] - df2.speeds[counter-1])**2) 
        percentDiff = diff / (df2.speeds[counter-1] + .00000001)    
        if percentDiff > .3:
            df2.speeds[counter] = np.nan
            df2.speeds[counter-1] = np.nan
            df2.speeds[counter+1] = np.nan
            df2.speeds[counter+2] = np.nan
            df2.speeds[counter+3] = np.nan
            df2.speeds[counter+4] = np.nan
            counter += 6
        else: counter += 1
    # only keeping entries where altDelta less than 1, speed greater than 1, distDelta greater than 1, and change in speed less than threshhold        
    df2 = df2[(np.sqrt(df2.altDeltas**2) < 1.0) & (df2.speeds > 1.0)]    
    return df2

"""
#print(getFitLine(df))
df.loc[:,['hr','time']].to_csv("/home/rudebeans/jan10_project/calculator/static/calculator/hrTime.csv", index=False, header=True)

df.loc[:,['speeds','time']].to_csv("/home/rudebeans/jan10_project/calculator/static/calculator/speedsTime.csv", index=False, header=True)

getFitLine(df).to_csv("/home/rudebeans/jan10_project/calculator/static/calculator/fitLine.csv", index=False, header=True)

df.loc[:,['hr','speeds']].to_csv("/home/rudebeans/jan10_project/calculator/static/calculator/scatter.csv", index=False, header=True)

getFitDf(df).loc[:,['hr','speeds']].to_csv("/home/rudebeans/jan10_project/calculator/static/calculator/scatterFitCleaned.csv", index=False, header=True)
 """
 
def getHrVar(df):
    minHr = 80
    df = df[df.hr > minHr]
    hrVar = np.std(df.hr)
    return hrVar
    
def getHrAvg(df):
    minHr = 80
    df = df[df.hr > minHr]
    avg = np.mean(df.hr)
    return avg
    
def getClimb(df):
    df = df[df.altDeltas > 0.0]
    climb = np.sum(df.altDeltas)
    return climb
    
def getTotalDistance(df):
    totalDistance = np.sum(df.distDeltas)
    return totalDistance

#print(getTotalDistance(df))


minRecHr = 90
maxRecHr = 125
minLoadAcc = 126
minImpulseAcc = 165
maxHr = 190

def getLoad(df): # simple version 
    loadDf = df[df.hr >= minLoadAcc]
    totalLoad = np.sum(loadDf.hr * loadDf.timeDeltas)
    return totalLoad
  
# getting distance spent in each zone
impulse = 165
stamina = 145
easy = 120
recovery = 80    
def getImpulse(df):
    impulseDf = df[df.hr >= impulse]
    totImpulse = np.sum(impulseDf.distDeltas)
    return totImpulse
def getStamina(df):
    staminaDf = df[(df.hr >= stamina) & (df.hr < impulse)]
    totStamina = np.sum(staminaDf.distDeltas)
    return totStamina   
def getEasy(df):
    easyDf = df[(df.hr >= easy) & (df.hr < stamina)]
    totEasy = np.sum(easyDf.distDeltas)
    return totEasy
def getRecovery(df):
    recoveryDf = df[(df.hr >= recovery) & (df.hr < easy)]
    totRecovery = np.sum(recoveryDf.distDeltas)
    return totRecovery
def getZones(df):
    zones = {'impulse':getImpulse(df), 'stamina':getStamina(df), 'easy':getEasy(df), 'recovery':getRecovery(df)}
    return zones


sampleMin = 135 # simple fit score, ie speed at easy pace
sampleMax = 145    
def getEasySpeed(df): 
    fitLine = getFitLine(df)
    segment = fitLine[(fitLine.index >= sampleMin) & (fitLine.index <= sampleMax)]
    avgSpeed = np.mean(segment.avgSpeed)
    return avgSpeed


  

easyHr = 145 # stamina
minHr = 80 # recovery
threshhold = 170 # impulse
  
def getRealMiles(df): # Returns normalized distance, ie real load

    slope = 1.0 / (stamina - recovery)
    a = -slope*recovery
    easy = df[(df.hr <= stamina) & (df.hr >= recovery)]
    easy['normalizer'] = a + slope*(easy.hr)
    easy['normalizedDist'] = easy.normalizer * easy.distDeltas
    easyDist = np.sum(easy.normalizedDist)
    
    slope = 1.0 / (impulse - stamina) 
    a = -slope*stamina
    thresh = df[df.hr > stamina] # thresh = impulse
    thresh['normalizer'] = a + slope*(thresh.hr) + 1
    thresh['normalizedDist'] = thresh.normalizer * thresh.distDeltas
    threshDist = np.sum(thresh.normalizedDist)
    
    normalizedDist = threshDist + easyDist
    return normalizedDist
    





#print(getRecovery(df))
    
#print(df.hr)
    
"""
# TODO: change this to simpler absolute scale. Impulse calculations: Everything above minImpulseAcc counts as impulse. Time at max HR counted as 3 times more impulse accumulation than at minImpulseAcc. Normalized impulse is multiplied by time, then summed up for a total impulse figure

# Projecting onto the scale described directly above. Note magic number 2 in normalizedImpulse

def getImpulse(df):
    normalizerDomain = maxHr - minImpulseAcc
    impulseDf = df[df.hr >= minImpulseAcc]
    impulseDf['rawImpulse'] = impulseDf['hr'] - minImpulseAcc
    impulseDf['normalizedImpulse'] = 1.0 + (2.0 * impulseDf.rawImpulse) / normalizerDomain
    totalImpulse = np.sum(impulseDf.normalizedImpulse * impulseDf.timeDeltas)    
    return totalImpulse
"""

def getSpeed(distance, time):
    speed = list()
    for i in range(1,len(time)):
        timeSegment = time[i] - time[i-1]
        distanceCovered = distance[i] - distance[i-1]
        speedEntry = distanceCovered / timeSegment
        speed.append(speedEntry)
        
        
#print(getSpeed(distance, time))



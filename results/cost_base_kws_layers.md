| # | op | groups x M,K,N | WS tile (mt,nt,kt) | WS cycles (DB) | WS util | WS bound | OS tile | OS cycles (DB) | OS util | OS bound |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| 0 | QuantizeLinear | - | - | 308 | - | transfer | - | 308 | - | transfer |
| 1 | QLinearConv | 1 x 64,40,125 | 16,16,40 | 1666 | 75.0% | transfer | 16,125,40 | 1840 | 67.9% | transfer |
| 2 | QLinearConv | 64 x 1,9,125 | 1,16,9 | 4992 | 5.6% | compute | 1,16,9 | 4992 | 5.6% | compute |
| 3 | QLinearConv | 1 x 64,64,125 | 64,16,64 | 2824 | 70.8% | transfer | 64,125,1 | 3104 | 64.4% | transfer |
| 4 | QLinearConv | 64 x 1,9,125 | 1,16,9 | 4992 | 5.6% | compute | 1,16,9 | 4992 | 5.6% | compute |
| 5 | QLinearConv | 1 x 64,64,125 | 64,16,64 | 2824 | 70.8% | transfer | 64,125,1 | 3104 | 64.4% | transfer |
| 6 | QLinearConv | 64 x 1,9,125 | 1,16,9 | 4992 | 5.6% | compute | 1,16,9 | 4992 | 5.6% | compute |
| 7 | QLinearConv | 1 x 64,64,125 | 64,16,64 | 2824 | 70.8% | transfer | 64,125,1 | 3104 | 64.4% | transfer |
| 8 | QLinearConv | 64 x 1,9,125 | 1,16,9 | 4992 | 5.6% | compute | 1,16,9 | 4992 | 5.6% | compute |
| 9 | QLinearConv | 1 x 64,64,125 | 64,16,64 | 2824 | 70.8% | transfer | 64,125,1 | 3104 | 64.4% | transfer |
| 10 | DequantizeLinear | - | - | 5000 | - | transfer | - | 5000 | - | transfer |
| 11 | ReduceMean | - | - | 4047 | - | transfer | - | 4047 | - | transfer |
| 12 | QuantizeLinear | - | - | 40 | - | transfer | - | 40 | - | transfer |
| 13 | QGemm | 1 x 12,64,1 | 12,1,32 | 150 | 2.0% | transfer | 12,1,8 | 120 | 2.5% | transfer |
| 14 | DequantizeLinear | - | - | 8 | - | transfer | - | 8 | - | transfer |

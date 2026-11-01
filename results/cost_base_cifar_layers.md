| # | op | groups x M,K,N | WS tile (mt,nt,kt) | WS cycles (DB) | WS util | WS bound | OS tile | OS cycles (DB) | OS util | OS bound |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| 0 | QuantizeLinear | - | - | 1920 | - | transfer | - | 1920 | - | transfer |
| 1 | QLinearConv | 1 x 32,27,1024 | 32,16,27 | 4652 | 74.3% | transfer | 16,256,27 | 5634 | 61.3% | transfer |
| 2 | QLinearConv | 1 x 32,288,1024 | 16,64,288 | 37832 | 97.4% | compute | 16,128,64 | 37370 | 98.6% | compute |
| 3 | MaxPool | - | - | 5128 | - | transfer | - | 5128 | - | transfer |
| 4 | QLinearConv | 1 x 64,288,256 | 16,64,288 | 19400 | 95.0% | compute | 16,128,64 | 18938 | 97.3% | compute |
| 5 | QLinearConv | 1 x 64,576,256 | 16,64,576 | 38664 | 95.3% | compute | 16,128,64 | 37370 | 98.6% | compute |
| 6 | MaxPool | - | - | 2568 | - | transfer | - | 2568 | - | transfer |
| 7 | QLinearGlobalAveragePool | - | - | 544 | - | transfer | - | 544 | - | transfer |
| 8 | Flatten | - | - | 0 | - | none | - | 0 | - | none |
| 9 | QGemm | 1 x 10,64,1 | 10,1,32 | 132 | 1.9% | transfer | 10,1,8 | 103 | 2.4% | transfer |
| 10 | DequantizeLinear | - | - | 8 | - | transfer | - | 8 | - | transfer |

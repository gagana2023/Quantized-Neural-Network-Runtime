# Results

**Draft – for my review.** Everything in the cost-model sections is *modelled* accelerator cost for a generic, configurable
accelerator (see DESIGN D25-D30). It is not measured, not host CPU time, and not a statement about numerical correctness
(those are `results/parity_*.json` and `results/accuracy_*.json`). The interpretation text is a first draft by Claude Code; the
tables are copied unchanged from `results/cost_base_<model>_layers.md` and `_summary.md`, which `tools/run_cost_model.py` writes.

Reproduce: `python -m tools.run_cost_model --model cifar` and `--model kws` (base config `costmodel/configs/base.yaml`:
16x16 MACs per cycle, 32 KiB weight SRAM, 64 KiB activation SRAM, 8 B/cycle DMA, 16 vector lanes, 500 MHz).
Files: `results/cost_base_<model>.csv` (per layer, both loop orders), `_sweeps.csv`, `.json` (environment, config, totals),
`.png` (per-layer compute vs transfer), `_sweeps.png`.

WS = weight stationary, OS = output stationary, DB = double buffered. "Bound" compares a layer's total compute cycles with its
total transfer cycles. Utilisation = useful MACs / (R x C x total cycles).

## CIFAR CNN

### Per layer (base config)

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

### Totals and sweeps

| scheme | compute cycles | transfer cycles | no overlap | double buffered | DMA stalls cut | utilisation (DB) | DRAM KiB | compute-bound / transfer-bound layers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| weight_stationary | 99217 | 51660 | 150877 | 110848 | 77.49% | 86.3% | 403.5 | 3 / 7 |
| output_stationary | 99217 | 67598 | 166815 | 109583 | 84.67% | 87.3% | 527.9 | 3 / 7 |
| best_per_layer | 99217 | 66842 | 166059 | 108601 | 85.96% | 88.0% | 522.0 | 3 / 7 |

Sweep `sram` (whole-network cycles, double buffered / no overlap):

| point | WS | OS | WS DMA stalls cut | OS DMA stalls cut |
| --- | ---: | ---: | ---: | ---: |
| 8+16 KiB | 217051 / 315237 | 115306 / 191951 | 45.47% | 82.71% |
| 16+32 KiB | 124651 / 183741 | 109958 / 167343 | 69.92% | 84.25% |
| 32+64 KiB | 110848 / 150877 | 109583 / 166815 | 77.49% | 84.67% |
| 64+128 KiB | 110856 / 150877 | 109577 / 166431 | 77.47% | 84.59% |
| 128+256 KiB | 110856 / 150877 | 109577 / 166431 | 77.47% | 84.59% |
| 256+512 KiB | 110856 / 150877 | 109577 / 166431 | 77.47% | 84.59% |

Sweep `bandwidth` (whole-network cycles, double buffered / no overlap):

| point | WS | OS | WS DMA stalls cut | OS DMA stalls cut |
| --- | ---: | ---: | ---: | ---: |
| 1 B/cycle | 364622 / 463238 | 447625 / 546790 | 27.09% | 22.16% |
| 2 B/cycle | 189693 / 281259 | 223965 / 322995 | 50.3% | 44.26% |
| 4 B/cycle | 133943 / 202519 | 135587 / 211067 | 66.38% | 67.48% |
| 8 B/cycle | 110848 / 150877 | 109583 / 166815 | 77.49% | 84.67% |
| 16 B/cycle | 102404 / 125261 | 101255 / 145976 | 87.76% | 95.64% |
| 32 B/cycle | 100039 / 128821 | 99646 / 136214 | 97.2% | 98.82% |
| 64 B/cycle | 99377 / 147085 | 99323 / 138148 | 99.65% | 99.71% |

Sweep `array` (whole-network cycles, double buffered / no overlap):

| point | WS | OS | WS DMA stalls cut | OS DMA stalls cut |
| --- | ---: | ---: | ---: | ---: |
| 4x4 | 1540326 / 2332414 | 1540259 / 2135728 | 99.16% | 98.89% |
| 8x8 | 394039 / 525041 | 393002 / 557423 | 94.31% | 95.99% |
| 16x16 | 110848 / 150877 | 109583 / 166815 | 77.49% | 84.67% |
| 32x32 | 76451 / 101709 | 57847 / 84991 | 34.04% | 47.22% |
| 64x64 | 136815 / 148757 | 68519 / 80415 | 8.74% | 17.41% |
| 8x32 | 109953 / 171441 | 109617 / 166919 | 85.21% | 84.72% |
| 32x8 | 123202 / 173437 | 110857 / 156667 | 67.68% | 79.74% |

## KWS DS-CNN

### Per layer (base config)

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

### Totals and sweeps

| scheme | compute cycles | transfer cycles | no overlap | double buffered | DMA stalls cut | utilisation (DB) | DRAM KiB | compute-bound / transfer-bound layers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| weight_stationary | 29018 | 29996 | 59014 | 42483 | 55.11% | 24.4% | 232.0 | 4 / 11 |
| output_stationary | 29018 | 33660 | 62678 | 43747 | 56.24% | 23.7% | 254.7 | 4 / 11 |
| best_per_layer | 29018 | 29984 | 59002 | 42453 | 55.19% | 24.4% | 231.9 | 4 / 11 |

Sweep `sram` (whole-network cycles, double buffered / no overlap):

| point | WS | OS | WS DMA stalls cut | OS DMA stalls cut |
| --- | ---: | ---: | ---: | ---: |
| 8+16 KiB | 45619 / 63014 | 52349 / 73462 | 51.17% | 47.5% |
| 16+32 KiB | 42487 / 59019 | 45767 / 66779 | 55.11% | 55.65% |
| 32+64 KiB | 42483 / 59014 | 43747 / 62678 | 55.11% | 56.24% |
| 64+128 KiB | 42483 / 59014 | 43747 / 62678 | 55.11% | 56.24% |
| 128+256 KiB | 42483 / 59014 | 43747 / 62678 | 55.11% | 56.24% |
| 256+512 KiB | 42483 / 59014 | 43747 / 62678 | 55.11% | 56.24% |

Sweep `bandwidth` (whole-network cycles, double buffered / no overlap):

| point | WS | OS | WS DMA stalls cut | OS DMA stalls cut |
| --- | ---: | ---: | ---: | ---: |
| 1 B/cycle | 236187 / 265040 | 238519 / 265324 | 12.22% | 11.34% |
| 2 B/cycle | 119102 / 147466 | 121286 / 148984 | 23.95% | 23.09% |
| 4 B/cycle | 63877 / 88626 | 67115 / 91603 | 41.52% | 39.13% |
| 8 B/cycle | 42483 / 59014 | 43747 / 62678 | 55.11% | 56.24% |
| 16 B/cycle | 34900 / 46031 | 35318 / 51125 | 65.42% | 71.5% |
| 32 B/cycle | 31360 / 41517 | 31201 / 41448 | 81.33% | 82.51% |
| 64 B/cycle | 29902 / 35329 | 29781 / 37734 | 86.14% | 91.36% |

Sweep `array` (whole-network cycles, double buffered / no overlap):

| point | WS | OS | WS DMA stalls cut | OS DMA stalls cut |
| --- | ---: | ---: | ---: | ---: |
| 4x4 | 236229 / 317323 | 235471 / 340887 | 89.31% | 92.18% |
| 8x8 | 87058 / 118202 | 85893 / 141694 | 73.68% | 84.85% |
| 16x16 | 42483 / 59014 | 43747 / 62678 | 55.11% | 56.24% |
| 32x32 | 32981 / 42566 | 33555 / 44142 | 32.09% | 33.67% |
| 64x64 | 30369 / 36120 | 30665 / 36396 | 19.29% | 19.05% |
| 8x32 | 35379 / 49886 | 36085 / 51502 | 48.32% | 48.73% |
| 32x8 | 59877 / 77320 | 61321 / 93626 | 58.4% | 69.96% |

## Draft interpretation

**Double buffering.** Overlapping DMA with compute removes most of the modelled DMA stall time (total minus compute cycles):
CIFAR 77.5% (WS) and 84.7% (OS), KWS 55.1% and 56.2%. In total cycles that is 26.5% / 34.3% (CIFAR) and 28.0% / 30.2% (KWS).
CIFAR with DB ends 11.7% above its pure-compute floor (110,848 vs 99,217 cycles, WS). The no-overlap figure reuses the tiling
that was chosen to minimise the DB cycles (DESIGN D27), so a design built without double buffering and tiled for itself could do
somewhat better than the no-overlap column; the percentages above are therefore an upper estimate of the benefit. The visible
symptom is that the no-overlap WS total does not fall monotonically with bandwidth (CIFAR 128,821 cycles at 32 B/cycle, 147,085
at 64 B/cycle).

**Which layers are transfer-bound (base config).** CIFAR: 7 of 10 data-moving layers. These are the first conv (K = 27, so each
loaded input byte feeds few MACs), both max-pools, global average pool, the FC layer and the quantise / dequantise ends. The
three convs with K >= 288 are compute-bound at 95-98.6% utilisation. KWS: 11 of 15 layers are transfer-bound: the stem conv,
all four pointwise convs (70.8% WS utilisation), the vector-unit layers, and the FC. The four depthwise convs are compute-bound.
In KWS the float32 `DequantizeLinear` + `ReduceMean` pair at the end (8,000 float32 elements moved each way) costs 9,047 of
42,483 DB cycles (21%). That is an artefact of where this quantised graph puts the quantisation boundary, and an accelerator
runtime would more likely fuse or reorder it; I did not model that.

**Depthwise convolution.** Each KWS depthwise conv has M = 1 per group, so only one of the 16 array rows is ever busy: 72,000
MACs in 4,992 DB cycles is 5.6% utilisation, against a compute-only ceiling of 1/16 = 6.25% for this mapping. Those four layers
take 19,968 of 42,483 DB cycles (47%) while doing 288,000 of the network's 2,656,768 MACs (10.8%); the whole network's utilisation is 24.4%
(WS). Mapping several channels across the rows would fix this but needs a different array dataflow that I did not model. With
a 4x4 array the same network reaches 70.3% utilisation; with 64x64 it falls to 2.1%.

**Loop orders.** WS reads far fewer DRAM bytes (CIFAR 403.5 KiB vs 527.9 KiB for OS; KWS 232.0 vs 254.7 KiB) and wins on the
no-overlap totals, but with DB the two are within 1.1% (CIFAR) and 3.0% (KWS) of each other, with OS ahead on CIFAR and WS ahead
on KWS. I did not isolate why. Taking the better scheme per layer gives 108,601 (CIFAR) and 42,453 (KWS) DB cycles, i.e. 0.9%
and 0.1% below the best single scheme.

**Sweeps.**
- *SRAM:* beyond 32+64 KiB nothing improves (the biggest layer's tile already fits). Shrinking to 8+16 KiB costs CIFAR WS 96%
  (217,051 vs 110,848 cycles) because K must be cut and INT32 partial sums spill to DRAM; OS keeps accumulators on chip and
  loses only 5%.
- *Bandwidth:* strongest lever below 16 B/cycle. 8 -> 16 B/cycle saves 7.6% (CIFAR WS) and 17.9% (KWS WS). Beyond that both
  models approach their compute floor (CIFAR 99,377 at 64 B/cycle vs 99,217 compute cycles).
- *Array shape:* more MACs help until layers are too narrow for the array. CIFAR WS goes 110,848 (16x16) -> 76,451 (32x32) ->
  136,815 (64x64); its layers have only 32-64 output channels, so a 64-row array is at most half used and transfer then dominates.
  Non-square shapes are close to 16x16 (8x32: 109,953) except 32x8 (123,202).

**What would help most?** Two candidates, and they answer different questions. For the transfer-bound layers, doubling DMA
bandwidth (8 -> 16 B/cycle) is the lever that targets them: -7.6% cycles on CIFAR and -17.8% / -19.3% on KWS (WS / OS), at no
utilisation cost. Going from a 16x16 to a 32x32 array cuts more cycles (CIFAR WS -31%, OS -47%; KWS WS -22%) but utilisation
drops from 86-87% to 31-41% on CIFAR and from 24% to 8% on KWS, because the layers have only 32-64 output channels and the
extra rows sit idle; it shortens the compute-bound layers, not the transfer-bound ones. So the evidence here favours bandwidth
for transfer-bound layers and a different mapping (not a bigger array) for depthwise. Keeping activations in SRAM between
layers would probably help more than either, but that is exactly what this model does not capture.

## What the cost model ignores (and how it could change the conclusions)

- **Layers are isolated:** every layer reads its input from DRAM and writes its output back. Real accelerators keep small
  activations on chip and fuse conv + pool, which would make many "transfer-bound" layers compute-bound and shrink the DRAM
  totals. This is the largest simplification; it probably exaggerates transfer-boundness.
- No DRAM latency, burst, bank or refresh effects; DMA is one engine at a flat bytes-per-cycle rate.
- No halo re-fetch between neighbouring conv tiles (input reuse is assumed ideal inside the DRAM traffic formula) and no
  on-chip im2col cost.
- No array fill / drain cycles and no requantisation cycles (assumed free when accumulators drain).
- INT32 accumulators are charged to the activation SRAM; weights and activations are INT8 only; scales and zero points are free.
- The vector unit and the array never run at the same time; groups of a grouped convolution run one after another with no
  overlap across the group boundary.
- Tile search is restricted to unit x 2^j sizes (powers of two times R, C or 1) plus the full dimension, not exhaustive.
- No energy model, so SRAM vs DRAM bytes are reported but never traded off.

All of these favour or penalise different layers, so absolute cycle counts should not be quoted as hardware predictions; the
trends (which layers are narrow, where bandwidth matters, why depthwise is hard) are the defensible part.

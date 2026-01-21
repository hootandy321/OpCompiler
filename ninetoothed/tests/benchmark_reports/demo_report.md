# Demo Fusion Performance (Synthetic Data) - Benchmark Report

**Generated:** 2026-01-20 11:37:26
## Test Environment
- **device:** Unknown- **note:** Error detecting GPU: 'function' object has no attribute 'is_available'
## AddMM Small
### Test Parameters
- **M:** 512- **N:** 512- **K:** 512
### Performance Results
| Implementation | Host Time (ms) | Device Time (ms) | Host Speedup | Device Speedup |
|----------------|----------------|------------------|--------------|----------------|
| PyTorch Native | 5.360 | 4.824 | - | - |
| Separate Ops | 5.628 | 4.920 | 0.95x | 0.98x |
| Manual Fusion | 3.970 | 3.446 | 1.35x | 1.40x |

## AddMM Medium
### Test Parameters
- **M:** 1024- **N:** 1024- **K:** 1024
### Performance Results
| Implementation | Host Time (ms) | Device Time (ms) | Host Speedup | Device Speedup |
|----------------|----------------|------------------|--------------|----------------|
| PyTorch Native | 19.433 | 17.489 | - | - |
| Separate Ops | 20.404 | 17.839 | 0.95x | 0.98x |
| Manual Fusion | 14.395 | 12.492 | 1.35x | 1.40x |

## AddMM Large
### Test Parameters
- **M:** 2048- **N:** 2048- **K:** 2048
### Performance Results
| Implementation | Host Time (ms) | Device Time (ms) | Host Speedup | Device Speedup |
|----------------|----------------|------------------|--------------|----------------|
| PyTorch Native | 79.412 | 71.470 | - | - |
| Separate Ops | 83.382 | 72.900 | 0.95x | 0.98x |
| Manual Fusion | 58.823 | 51.050 | 1.35x | 1.40x |

## Chain Small
### Test Parameters
- **size:** 1024
### Performance Results
| Implementation | Host Time (ms) | Device Time (ms) | Host Speedup | Device Speedup |
|----------------|----------------|------------------|--------------|----------------|
| PyTorch Native | 4.630 | 4.167 | - | - |
| Separate Ops | 4.861 | 4.250 | 0.95x | 0.98x |
| Manual Fusion | 3.429 | 2.976 | 1.35x | 1.40x |

## Chain Medium
### Test Parameters
- **size:** 4096
### Performance Results
| Implementation | Host Time (ms) | Device Time (ms) | Host Speedup | Device Speedup |
|----------------|----------------|------------------|--------------|----------------|
| PyTorch Native | 20.627 | 18.565 | - | - |
| Separate Ops | 21.659 | 18.936 | 0.95x | 0.98x |
| Manual Fusion | 15.280 | 13.260 | 1.35x | 1.40x |

## Chain Large
### Test Parameters
- **size:** 16384
### Performance Results
| Implementation | Host Time (ms) | Device Time (ms) | Host Speedup | Device Speedup |
|----------------|----------------|------------------|--------------|----------------|
| PyTorch Native | 77.211 | 69.490 | - | - |
| Separate Ops | 81.071 | 70.880 | 0.95x | 0.98x |
| Manual Fusion | 57.193 | 49.636 | 1.35x | 1.40x |


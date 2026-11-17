<div align="center">
<img src="portfolio-assets/cover.svg" width="100%" alt="GPU-Accelerated Satellite Simulation project cover">

# GPU-Accelerated Satellite Simulation

**Compare serial, OpenMP and OpenMP plus OpenCL implementations of a gravitational simulation.**

C · OpenMP · OpenCL · SDL2 · CMake

[Overview](#overview) · [Design](portfolio/DESIGN.md) · [Setup](portfolio/SETUP.md) · [Validation](portfolio/VALIDATION.md) · [Evidence](portfolio/EVIDENCE.md)
</div>

## Overview

| Item | Record |
| :-- | :-- |
| Project type | Team course project |
| Field | Engineering |
| Material in this package | Existing repository inspected |
| Documentation review | 17 November 2026 |

## Scope

Compare serial, OpenMP and OpenMP plus OpenCL implementations of a gravitational simulation. The documentation separates the stated project scope, inspected artifacts and future work. A listed feature is not a new test result.

### Main elements

- Satellite state
- Physics update
- Pixel computation
- SDL2 display

## Start here

1. Read the [design record](portfolio/DESIGN.md) for the system elements and boundaries.
2. Follow the [setup record](portfolio/SETUP.md) for available entry points.
3. Use the [validation plan](portfolio/VALIDATION.md) to record reproducible results.
4. Check the [evidence register](portfolio/EVIDENCE.md) before reusing claims.

## Technical limits

The inspected CPU source calculates gravity from one black-hole center. It does not sum gravity between every satellite pair. The physics work scales with satellite count and the graphics work includes pixel-by-satellite loops. The previous generic all-pairs O(N²) description does not describe this source. No speedup value is asserted here without its run conditions.

## Project preview

![Project artifact](images/Demo.gif)

This image is an existing repository artifact. It is not evidence of a new test.

## Available material

| Path | Type |
| :-- | :-- |
| [LICENSE](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/LICENSE) | file |
| [README.md](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/README.md) | .md |
| [images/Demo.gif](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/images/Demo.gif) | .gif |
| [images/Optimizer.png](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/images/Optimizer.png) | .png |
| [results/Part1-OpenMP.pdf](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/results/Part1-OpenMP.pdf) | .pdf |
| [results/Part2-OpenCL.pdf](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/results/Part2-OpenCL.pdf) | .pdf |
| [src/SDL2/SDL2-devel-2.32.10-VC.zip](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/src/SDL2/SDL2-devel-2.32.10-VC.zip) | .zip |
| [src/VS-CMakeSetting/CMakeSettings.json](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/src/VS-CMakeSetting/CMakeSettings.json) | .json |
| [src/cpu/CMakeLists.txt](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/src/cpu/CMakeLists.txt) | .txt |
| [src/cpu/parallel.c](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/src/cpu/parallel.c) | .c |
| [src/opencl/CMakeLists.txt](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/src/opencl/CMakeLists.txt) | .txt |
| [src/opencl/parallel.c](https://github.com/Ace-Alch/GPU-Accelerated-Satellite-Sim/blob/main/src/opencl/parallel.c) | .c |

See the [complete source inventory](portfolio/SOURCE-INVENTORY.md) for all inspected file paths.

## Next work

- [ ] Raw benchmark timings
- [ ] Exact build and hardware settings
- [ ] Per-author contribution
- [ ] Repeatable speedup calculation

## Credits and reuse

This is a team project. The exact division of work must be recorded before individual authorship is claimed.

Existing license terms remain in effect. This package does not grant a new license over source code, course material, models or datasets. See [reuse and contribution notes](portfolio/CONTRIBUTIONS.md).

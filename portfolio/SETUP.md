# GPU-Accelerated Satellite Simulation: setup

## Current entry point

1. Select the serial, OpenMP or OpenCL project under `src/`.
2. Inspect that folder’s `CMakeLists.txt`.
3. Configure a matching C toolchain and SDL2. For the GPU version, configure the OpenCL headers and runtime.
4. Build all compared versions with the same release settings.
5. Use the same seed, satellite count and viewport for timing comparisons.
6. Separate physics time, pixel computation time and total frame time.

No build or benchmark was run during this update.

## Required records

- Raw benchmark timings
- Exact build and hardware settings
- Per-author contribution
- Repeatable speedup calculation

## Reproducibility record

Record the source commit, operating system, tool versions, dependencies, input files, configuration and command. Store the output separately from the expected result. Mark any manual step.

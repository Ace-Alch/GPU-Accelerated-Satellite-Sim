# Parallel Satellite Simulation

**One orbital workload, three implementations.** This Tampere University project follows a C reference implementation through CPU threading and OpenCL compute.

[Replay the CPU reference run](https://ace-alch.github.io/apps/exhibits/satellite/) · [OpenMP report](results/Part1-OpenMP.pdf) · [OpenCL report](results/Part2-OpenCL.pdf)

![Recording of the original SDL2 satellite simulation](images/Demo.gif)

*Original desktop demonstration. In the SDL2 app, the mouse moves the gravity center.*

## Trace the compute path

1. **[C reference](src/cpu/parallel.c)** — The reference solver updates satellite positions and velocities under central gravity. SDL2 displays the frame.
2. **[OpenMP implementation](src/openmp/parallel.c)** — Inspect the CPU parallel loops and compare them with the reference implementation.
3. **[OpenCL host](src/opencl/parallel.c) and [kernels](src/opencl/parallel.cl)** — Follow the device setup, work dispatch, and kernels.

The workloads include both motion updates and per-pixel rendering. Use the two reports above for their original performance measurements and test conditions.

## Browser replay

The replay contains **181 frames of 64 satellites** from a native run of the sequential C reference solver with a fixed seed. The browser draws recorded coordinates with trails. This run checks the reference trajectory. It does not measure GPU performance.

## Build on your machine

Start with the [setup notes](portfolio/SETUP.md) and the CMake files for [reference](src/cpu/CMakeLists.txt), [OpenMP](src/openmp/CMakeLists.txt), or [OpenCL](src/opencl/CMakeLists.txt). The desktop build uses SDL2. The OpenCL variant also needs an OpenCL development environment and a compatible device runtime.

**Authors:** Mehdi Moallemkolaei and Md Ashfak Haider Nehal. See the [license and source attribution](LICENSE).

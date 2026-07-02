# knok Demo

Interactive `egui` demo for static `knok` tensor graphs.

## Run

From a shell with the `knok` build dependencies available:

```sh
cargo run
```

The app builds `512x512` fixed-shape graph variants for:

- Mandelbrot
- Heat diffusion
- Wave simulation
- Conway's Game of Life
- Particle interaction

CPU is enabled by default. Optional backend builds:

```sh
cargo run --features vulkan
cargo run --features cuda
```

On macOS, Metal graph variants are compiled by target `cfg`.

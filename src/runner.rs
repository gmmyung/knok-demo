use knok::Engine;

use crate::{
    backend::BackendChoice,
    cpu_graphs,
    presets::{DiffusionPreset, MandelbrotIterations, ParticlePreset, WavePreset},
    types::{Field, ParticleVec},
};

#[cfg(feature = "cuda")]
use crate::cuda_graphs;

#[cfg(target_os = "macos")]
use crate::metal_graphs;

#[cfg(feature = "vulkan")]
use crate::vulkan_graphs;

pub(crate) fn run_mandelbrot(
    backend: BackendChoice,
    preset: MandelbrotIterations,
    engine: &Engine,
    x: Field,
    y: Field,
) -> knok::Result<Field> {
    match backend {
        BackendChoice::Cpu => match preset {
            MandelbrotIterations::Low => cpu_graphs::mandelbrot_24_cpu::run(engine, x, y),
            MandelbrotIterations::Medium => cpu_graphs::mandelbrot_48_cpu::run(engine, x, y),
            MandelbrotIterations::High => cpu_graphs::mandelbrot_72_cpu::run(engine, x, y),
        },
        #[cfg(feature = "vulkan")]
        BackendChoice::Vulkan => match preset {
            MandelbrotIterations::Low => vulkan_graphs::mandelbrot_24_vulkan::run(engine, x, y),
            MandelbrotIterations::Medium => vulkan_graphs::mandelbrot_48_vulkan::run(engine, x, y),
            MandelbrotIterations::High => vulkan_graphs::mandelbrot_72_vulkan::run(engine, x, y),
        },
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => cuda_graphs::mandelbrot_48_cuda::run(engine, x, y),
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => metal_graphs::mandelbrot_48_metal::run(engine, x, y),
    }
}

pub(crate) fn run_heat(
    backend: BackendChoice,
    preset: DiffusionPreset,
    engine: &Engine,
    field: Field,
) -> knok::Result<Field> {
    match backend {
        BackendChoice::Cpu => match preset {
            DiffusionPreset::Low => cpu_graphs::heat_low_cpu::run(engine, field),
            DiffusionPreset::Medium => cpu_graphs::heat_medium_cpu::run(engine, field),
            DiffusionPreset::High => cpu_graphs::heat_high_cpu::run(engine, field),
        },
        #[cfg(feature = "vulkan")]
        BackendChoice::Vulkan => match preset {
            DiffusionPreset::Low => vulkan_graphs::heat_low_vulkan::run(engine, field),
            DiffusionPreset::Medium => vulkan_graphs::heat_medium_vulkan::run(engine, field),
            DiffusionPreset::High => vulkan_graphs::heat_high_vulkan::run(engine, field),
        },
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => cuda_graphs::heat_medium_cuda::run(engine, field),
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => metal_graphs::heat_medium_metal::run(engine, field),
    }
}

pub(crate) fn run_wave(
    backend: BackendChoice,
    preset: WavePreset,
    engine: &Engine,
    height: Field,
    velocity: Field,
) -> knok::Result<(Field, Field)> {
    match backend {
        BackendChoice::Cpu => match preset {
            WavePreset::Slow => cpu_graphs::wave_slow_cpu::run(engine, height, velocity),
            WavePreset::Medium => cpu_graphs::wave_medium_cpu::run(engine, height, velocity),
            WavePreset::Fast => cpu_graphs::wave_fast_cpu::run(engine, height, velocity),
        },
        #[cfg(feature = "vulkan")]
        BackendChoice::Vulkan => match preset {
            WavePreset::Slow => vulkan_graphs::wave_slow_vulkan::run(engine, height, velocity),
            WavePreset::Medium => vulkan_graphs::wave_medium_vulkan::run(engine, height, velocity),
            WavePreset::Fast => vulkan_graphs::wave_fast_vulkan::run(engine, height, velocity),
        },
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => cuda_graphs::wave_medium_cuda::run(engine, height, velocity),
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => metal_graphs::wave_medium_metal::run(engine, height, velocity),
    }
}

pub(crate) fn run_life(
    backend: BackendChoice,
    engine: &Engine,
    field: Field,
) -> knok::Result<Field> {
    match backend {
        BackendChoice::Cpu => cpu_graphs::life_cpu::run(engine, field),
        #[cfg(feature = "vulkan")]
        BackendChoice::Vulkan => vulkan_graphs::life_vulkan::run(engine, field),
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => cuda_graphs::life_cuda::run(engine, field),
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => metal_graphs::life_metal::run(engine, field),
    }
}

pub(crate) fn run_particles(
    backend: BackendChoice,
    preset: ParticlePreset,
    engine: &Engine,
    x: ParticleVec,
    y: ParticleVec,
    vx: ParticleVec,
    vy: ParticleVec,
) -> knok::Result<(ParticleVec, ParticleVec, ParticleVec, ParticleVec)> {
    match backend {
        BackendChoice::Cpu => match preset {
            ParticlePreset::Gentle => cpu_graphs::particles_gentle_cpu::run(engine, x, y, vx, vy),
            ParticlePreset::Strong => cpu_graphs::particles_strong_cpu::run(engine, x, y, vx, vy),
        },
        #[cfg(feature = "vulkan")]
        BackendChoice::Vulkan => match preset {
            ParticlePreset::Gentle => {
                vulkan_graphs::particles_gentle_vulkan::run(engine, x, y, vx, vy)
            }
            ParticlePreset::Strong => {
                vulkan_graphs::particles_strong_vulkan::run(engine, x, y, vx, vy)
            }
        },
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => cuda_graphs::particles_gentle_cuda::run(engine, x, y, vx, vy),
        #[cfg(target_os = "macos")]
        BackendChoice::Metal => metal_graphs::particles_gentle_metal::run(engine, x, y, vx, vy),
    }
}

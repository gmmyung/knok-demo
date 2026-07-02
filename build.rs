use knok_build::prelude::*;

const SIZE: usize = 512;
const PARTICLES: usize = 256;

type Field = T2<f32, SIZE, SIZE>;
type Particles = T1<f32, PARTICLES>;

fn mandelbrot_core<const ITERATIONS: usize>(x0: Field, y0: Field) -> Field {
    let mut zx = x0.clone() * 0.0;
    let mut zy = y0.clone() * 0.0;
    let mut count = x0.clone() * 0.0;

    for _ in 0..ITERATIONS {
        let zx2 = square(zx.clone());
        let zy2 = square(zy.clone());
        let active = less_equal(zx2.clone() + zy2.clone(), 4.0);
        let next_zx = zx2 - zy2 + x0.clone();
        let next_zy = zx.clone() * zy.clone() * 2.0 + y0.clone();
        count = r#where(active.clone(), count.clone() + 1.0, count);
        zx = r#where(active.clone(), next_zx, zx);
        zy = r#where(active, next_zy, zy);
    }

    count / (ITERATIONS as f32)
}

fn heat_core<const SUBSTEPS: usize>(mut field: Field, coeff: f32) -> Field {
    for _ in 0..SUBSTEPS {
        let up = roll(field.clone(), 0, 1);
        let down = roll(field.clone(), 0, SIZE - 1);
        let left = roll(field.clone(), 1, 1);
        let right = roll(field.clone(), 1, SIZE - 1);
        let laplacian = up + down + left + right - field.clone() * 4.0;
        field = clip(field + laplacian * coeff, 0.0, 1.0);
    }
    field
}

fn wave_core(height: Field, velocity: Field, speed: f32, damping: f32) -> (Field, Field) {
    let up = roll(height.clone(), 0, 1);
    let down = roll(height.clone(), 0, SIZE - 1);
    let left = roll(height.clone(), 1, 1);
    let right = roll(height.clone(), 1, SIZE - 1);
    let laplacian = up + down + left + right - height.clone() * 4.0;
    let next_velocity = velocity * damping + laplacian * speed;
    let next_height = clip(height + next_velocity.clone(), -1.0, 1.0);
    (next_height, next_velocity)
}

fn particles_core(
    x: Particles,
    y: Particles,
    vx: Particles,
    vy: Particles,
    gravity: f32,
    softening: f32,
) -> (Particles, Particles, Particles, Particles) {
    let xi: T2<f32, PARTICLES, 1> = unsqueeze(x.clone());
    let yi: T2<f32, PARTICLES, 1> = unsqueeze(y.clone());
    let xj_row: T2<f32, 1, PARTICLES> = unsqueeze(x.clone());
    let yj_row: T2<f32, 1, PARTICLES> = unsqueeze(y.clone());
    let xj: T2<f32, PARTICLES, PARTICLES> = broadcast(xj_row);
    let yj: T2<f32, PARTICLES, PARTICLES> = broadcast(yj_row);
    let xi: T2<f32, PARTICLES, PARTICLES> = broadcast(xi);
    let yi: T2<f32, PARTICLES, PARTICLES> = broadcast(yi);

    let dx = xj - xi;
    let dy = yj - yi;
    let dist2 = square(dx.clone()) + square(dy.clone()) + softening;
    let inv_dist = reciprocal(sqrt(dist2));
    let inv_dist3 = inv_dist.clone() * inv_dist.clone() * inv_dist;
    let ax: Particles = sum_axis(dx * inv_dist3.clone() * gravity, 1);
    let ay: Particles = sum_axis(dy * inv_dist3 * gravity, 1);

    let next_vx = vx + ax * 0.016;
    let next_vy = vy + ay * 0.016;
    let next_x = x + next_vx.clone() * 0.016;
    let next_y = y + next_vy.clone() * 0.016;
    (next_x, next_y, next_vx, next_vy)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn mandelbrot_24_cpu(x: Field, y: Field) -> Field {
    mandelbrot_core::<24>(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn mandelbrot_48_cpu(x: Field, y: Field) -> Field {
    mandelbrot_core::<48>(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn mandelbrot_72_cpu(x: Field, y: Field) -> Field {
    mandelbrot_core::<72>(x, y)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn heat_low_cpu(field: Field) -> Field {
    heat_core::<3>(field, 0.12)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn heat_medium_cpu(field: Field) -> Field {
    heat_core::<4>(field, 0.18)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn heat_high_cpu(field: Field) -> Field {
    heat_core::<5>(field, 0.22)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn wave_slow_cpu(height: Field, velocity: Field) -> (Field, Field) {
    wave_core(height, velocity, 0.08, 0.992)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn wave_medium_cpu(height: Field, velocity: Field) -> (Field, Field) {
    wave_core(height, velocity, 0.12, 0.988)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn wave_fast_cpu(height: Field, velocity: Field) -> (Field, Field) {
    wave_core(height, velocity, 0.18, 0.982)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn particles_gentle_cpu(
    x: Particles,
    y: Particles,
    vx: Particles,
    vy: Particles,
) -> (Particles, Particles, Particles, Particles) {
    particles_core(x, y, vx, vy, 0.00018, 0.0025)
}

#[knok_build::graph(backend = Backend::LlvmCpu)]
fn particles_strong_cpu(
    x: Particles,
    y: Particles,
    vx: Particles,
    vy: Particles,
) -> (Particles, Particles, Particles, Particles) {
    particles_core(x, y, vx, vy, 0.00034, 0.0035)
}

#[cfg(feature = "vulkan")]
mod vulkan_graphs {
    use super::*;

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn mandelbrot_24_vulkan(x: Field, y: Field) -> Field {
        mandelbrot_core::<24>(x, y)
    }

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn mandelbrot_48_vulkan(x: Field, y: Field) -> Field {
        mandelbrot_core::<48>(x, y)
    }

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn mandelbrot_72_vulkan(x: Field, y: Field) -> Field {
        mandelbrot_core::<72>(x, y)
    }

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn heat_low_vulkan(field: Field) -> Field {
        heat_core::<3>(field, 0.12)
    }

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn heat_medium_vulkan(field: Field) -> Field {
        heat_core::<4>(field, 0.18)
    }

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn heat_high_vulkan(field: Field) -> Field {
        heat_core::<5>(field, 0.22)
    }

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn wave_slow_vulkan(height: Field, velocity: Field) -> (Field, Field) {
        wave_core(height, velocity, 0.08, 0.992)
    }

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn wave_medium_vulkan(height: Field, velocity: Field) -> (Field, Field) {
        wave_core(height, velocity, 0.12, 0.988)
    }

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn wave_fast_vulkan(height: Field, velocity: Field) -> (Field, Field) {
        wave_core(height, velocity, 0.18, 0.982)
    }

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn particles_gentle_vulkan(
        x: Particles,
        y: Particles,
        vx: Particles,
        vy: Particles,
    ) -> (Particles, Particles, Particles, Particles) {
        particles_core(x, y, vx, vy, 0.00018, 0.0025)
    }

    #[knok_build::graph(backend = Backend::VulkanSpirv)]
    fn particles_strong_vulkan(
        x: Particles,
        y: Particles,
        vx: Particles,
        vy: Particles,
    ) -> (Particles, Particles, Particles, Particles) {
        particles_core(x, y, vx, vy, 0.00034, 0.0035)
    }

    pub fn compile() {
        knok_build::compile_graphs_with_options!(
            BuildOptions::default().output_file("knok_vulkan_graphs.rs");
            mandelbrot_24_vulkan,
            mandelbrot_48_vulkan,
            mandelbrot_72_vulkan,
            heat_low_vulkan,
            heat_medium_vulkan,
            heat_high_vulkan,
            wave_slow_vulkan,
            wave_medium_vulkan,
            wave_fast_vulkan,
            particles_gentle_vulkan,
            particles_strong_vulkan
        );
    }
}

#[cfg(feature = "cuda")]
mod cuda_graphs {
    use super::*;

    #[knok_build::graph(backend = Backend::Cuda)]
    fn mandelbrot_48_cuda(x: Field, y: Field) -> Field {
        mandelbrot_core::<48>(x, y)
    }

    #[knok_build::graph(backend = Backend::Cuda)]
    fn heat_medium_cuda(field: Field) -> Field {
        heat_core::<4>(field, 0.18)
    }

    #[knok_build::graph(backend = Backend::Cuda)]
    fn wave_medium_cuda(height: Field, velocity: Field) -> (Field, Field) {
        wave_core(height, velocity, 0.12, 0.988)
    }

    #[knok_build::graph(backend = Backend::Cuda)]
    fn particles_gentle_cuda(
        x: Particles,
        y: Particles,
        vx: Particles,
        vy: Particles,
    ) -> (Particles, Particles, Particles, Particles) {
        particles_core(x, y, vx, vy, 0.00018, 0.0025)
    }

    pub fn compile() {
        knok_build::compile_graphs_with_options!(
            BuildOptions::default().output_file("knok_cuda_graphs.rs");
            mandelbrot_48_cuda,
            heat_medium_cuda,
            wave_medium_cuda,
            particles_gentle_cuda
        );
    }
}

#[cfg(target_os = "macos")]
mod metal_graphs {
    use super::*;

    #[knok_build::graph(backend = Backend::MetalSpirv)]
    fn mandelbrot_48_metal(x: Field, y: Field) -> Field {
        mandelbrot_core::<48>(x, y)
    }

    #[knok_build::graph(backend = Backend::MetalSpirv)]
    fn heat_medium_metal(field: Field) -> Field {
        heat_core::<4>(field, 0.18)
    }

    #[knok_build::graph(backend = Backend::MetalSpirv)]
    fn wave_medium_metal(height: Field, velocity: Field) -> (Field, Field) {
        wave_core(height, velocity, 0.12, 0.988)
    }

    #[knok_build::graph(backend = Backend::MetalSpirv)]
    fn particles_gentle_metal(
        x: Particles,
        y: Particles,
        vx: Particles,
        vy: Particles,
    ) -> (Particles, Particles, Particles, Particles) {
        particles_core(x, y, vx, vy, 0.00018, 0.0025)
    }

    pub fn compile() {
        knok_build::compile_graphs_with_options!(
            BuildOptions::default().output_file("knok_metal_graphs.rs");
            mandelbrot_48_metal,
            heat_medium_metal,
            wave_medium_metal,
            particles_gentle_metal
        );
    }
}

fn main() {
    knok_build::compile_graphs_with_options!(
        BuildOptions::default().output_file("knok_cpu_graphs.rs");
        mandelbrot_24_cpu,
        mandelbrot_48_cpu,
        mandelbrot_72_cpu,
        heat_low_cpu,
        heat_medium_cpu,
        heat_high_cpu,
        wave_slow_cpu,
        wave_medium_cpu,
        wave_fast_cpu,
        particles_gentle_cpu,
        particles_strong_cpu
    );

    #[cfg(feature = "vulkan")]
    vulkan_graphs::compile();
    #[cfg(feature = "cuda")]
    cuda_graphs::compile();
    #[cfg(target_os = "macos")]
    metal_graphs::compile();
}

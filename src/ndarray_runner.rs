use ndarray::{Array1, Array2, Axis};

use crate::{
    presets::{DiffusionPreset, MandelbrotIterations, ParticlePreset, WavePreset},
    types::{AppResult, Field, IntoAppResult, ParticleVec, SIZE},
};

pub(crate) fn run_mandelbrot(preset: MandelbrotIterations, x: Field, y: Field) -> AppResult<Field> {
    let iterations = match preset {
        MandelbrotIterations::Low => 24,
        MandelbrotIterations::Medium => 48,
        MandelbrotIterations::High => 72,
    };
    let x = array2_from_field(x)?;
    let y = array2_from_field(y)?;
    let output = Array2::from_shape_fn((SIZE, SIZE), |index| {
        let x0 = x[index];
        let y0 = y[index];
        let mut zx = 0.0;
        let mut zy = 0.0;
        let mut count = 0.0;
        for _ in 0..iterations {
            let zx2 = zx * zx;
            let zy2 = zy * zy;
            if zx2 + zy2 <= 4.0 {
                count += 1.0;
                let next_zx = zx2 - zy2 + x0;
                let next_zy = zx * zy * 2.0 + y0;
                zx = next_zx;
                zy = next_zy;
            }
        }
        count / iterations as f32
    });
    field_from_array2(output)
}

pub(crate) fn run_heat(preset: DiffusionPreset, field: Field) -> AppResult<Field> {
    let (substeps, coeff) = match preset {
        DiffusionPreset::Low => (3, 0.12),
        DiffusionPreset::Medium => (4, 0.18),
        DiffusionPreset::High => (5, 0.22),
    };
    let mut field = array2_from_field(field)?;
    for _ in 0..substeps {
        field = Array2::from_shape_fn((SIZE, SIZE), |(y, x)| {
            let up = field[((y + SIZE - 1) % SIZE, x)];
            let down = field[((y + 1) % SIZE, x)];
            let left = field[(y, (x + SIZE - 1) % SIZE)];
            let right = field[(y, (x + 1) % SIZE)];
            let center = field[(y, x)];
            (center + (up + down + left + right - center * 4.0) * coeff).clamp(0.0, 1.0)
        });
    }
    field_from_array2(field)
}

pub(crate) fn run_wave(
    preset: WavePreset,
    height: Field,
    velocity: Field,
) -> AppResult<(Field, Field)> {
    let (speed, damping) = match preset {
        WavePreset::Slow => (0.08, 0.992),
        WavePreset::Medium => (0.12, 0.988),
        WavePreset::Fast => (0.18, 0.982),
    };
    let height = array2_from_field(height)?;
    let velocity = array2_from_field(velocity)?;
    let next_velocity = Array2::from_shape_fn((SIZE, SIZE), |(y, x)| {
        let up = height[((y + SIZE - 1) % SIZE, x)];
        let down = height[((y + 1) % SIZE, x)];
        let left = height[(y, (x + SIZE - 1) % SIZE)];
        let right = height[(y, (x + 1) % SIZE)];
        let center = height[(y, x)];
        velocity[(y, x)] * damping + (up + down + left + right - center * 4.0) * speed
    });
    let next_height = Array2::from_shape_fn((SIZE, SIZE), |index| {
        (height[index] + next_velocity[index]).clamp(-1.0, 1.0)
    });
    Ok((
        field_from_array2(next_height)?,
        field_from_array2(next_velocity)?,
    ))
}

pub(crate) fn run_life(field: Field) -> AppResult<Field> {
    let field = array2_from_field(field)?;
    let output = Array2::from_shape_fn((SIZE, SIZE), |(y, x)| {
        let up = (y + SIZE - 1) % SIZE;
        let down = (y + 1) % SIZE;
        let left = (x + SIZE - 1) % SIZE;
        let right = (x + 1) % SIZE;
        let neighbors = field[(up, x)]
            + field[(down, x)]
            + field[(y, left)]
            + field[(y, right)]
            + field[(up, left)]
            + field[(up, right)]
            + field[(down, left)]
            + field[(down, right)];
        let alive = field[(y, x)] > 0.5;
        if (alive && neighbors == 2.0) || neighbors == 3.0 {
            1.0
        } else {
            0.0
        }
    });
    field_from_array2(output)
}

pub(crate) fn run_particles(
    preset: ParticlePreset,
    x: ParticleVec,
    y: ParticleVec,
    vx: ParticleVec,
    vy: ParticleVec,
) -> AppResult<(ParticleVec, ParticleVec, ParticleVec, ParticleVec)> {
    let (gravity, softening) = match preset {
        ParticlePreset::Gentle => (0.00018, 0.0025),
        ParticlePreset::Strong => (0.00034, 0.0035),
    };
    let x = Array1::from_vec(x.into_vec());
    let y = Array1::from_vec(y.into_vec());
    let vx = Array1::from_vec(vx.into_vec());
    let vy = Array1::from_vec(vy.into_vec());

    let xi = x.view().insert_axis(Axis(1));
    let yi = y.view().insert_axis(Axis(1));
    let xj = x.view().insert_axis(Axis(0));
    let yj = y.view().insert_axis(Axis(0));
    let dx = &xj - &xi;
    let dy = &yj - &yi;
    let dist2 = dx.mapv(|value| value * value) + dy.mapv(|value| value * value) + softening;
    let inv_dist = dist2.mapv(|value| value.sqrt().recip());
    let inv_dist3 = &inv_dist * &inv_dist * &inv_dist;
    let ax = (&dx * &inv_dist3 * gravity).sum_axis(Axis(1));
    let ay = (&dy * &inv_dist3 * gravity).sum_axis(Axis(1));

    let next_vx = vx + ax * 0.016;
    let next_vy = vy + ay * 0.016;
    let next_x = x + &next_vx * 0.016;
    let next_y = y + &next_vy * 0.016;

    Ok((
        particle_vec_from_array1(next_x)?,
        particle_vec_from_array1(next_y)?,
        particle_vec_from_array1(next_vx)?,
        particle_vec_from_array1(next_vy)?,
    ))
}

fn array2_from_field(field: Field) -> AppResult<Array2<f32>> {
    Array2::from_shape_vec((SIZE, SIZE), field.into_vec()).into_app_result()
}

fn field_from_array2(array: Array2<f32>) -> AppResult<Field> {
    let data = array.iter().copied().collect();
    Field::from_vec(data).into_app_result()
}

fn particle_vec_from_array1(array: Array1<f32>) -> AppResult<ParticleVec> {
    let data = array.iter().copied().collect();
    ParticleVec::from_vec(data).into_app_result()
}

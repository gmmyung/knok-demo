use crate::{
    field_ops::{stamp_disc, stamp_life_cells},
    presets::{ColorMap, DiffusionPreset, MandelbrotIterations, ParticlePreset, WavePreset},
    types::{AppResult, Field, IntoAppResult, ParticleVec, PARTICLES, SIZE},
};

pub(crate) struct MandelbrotState {
    pub(crate) center: [f32; 2],
    pub(crate) scale: f32,
    pub(crate) iterations: MandelbrotIterations,
    pub(crate) color_map: ColorMap,
    pub(crate) image: Vec<u8>,
    pub(crate) dirty: bool,
}

impl MandelbrotState {
    pub(crate) fn new() -> Self {
        Self {
            center: [-0.5, 0.0],
            scale: 3.0,
            iterations: MandelbrotIterations::Medium,
            color_map: ColorMap::Fire,
            image: vec![0; SIZE * SIZE * 4],
            dirty: true,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.center = [-0.5, 0.0];
        self.scale = 3.0;
        self.dirty = true;
    }

    pub(crate) fn grids(&self) -> AppResult<(Field, Field)> {
        let mut x = Vec::with_capacity(SIZE * SIZE);
        let mut y = Vec::with_capacity(SIZE * SIZE);
        let denom = (SIZE - 1) as f32;
        for row in 0..SIZE {
            let py = row as f32 / denom - 0.5;
            for col in 0..SIZE {
                let px = col as f32 / denom - 0.5;
                x.push(self.center[0] + px * self.scale);
                y.push(self.center[1] + py * self.scale);
            }
        }
        Ok((
            Field::from_vec(x).into_app_result()?,
            Field::from_vec(y).into_app_result()?,
        ))
    }
}

pub(crate) struct HeatState {
    pub(crate) field: Field,
    pub(crate) running: bool,
    pub(crate) preset: DiffusionPreset,
    pub(crate) image: Vec<u8>,
}

impl HeatState {
    pub(crate) fn new() -> Self {
        let mut state = Self {
            field: Field::filled(0.0),
            running: true,
            preset: DiffusionPreset::Medium,
            image: vec![0; SIZE * SIZE * 4],
        };
        state.reset();
        state
    }

    pub(crate) fn reset(&mut self) {
        let data = self.field.as_mut_slice();
        data.fill(0.0);
        stamp_disc(data, SIZE / 2, SIZE / 2, 34, 1.0);
        stamp_disc(data, SIZE / 3, SIZE / 3, 18, 0.75);
        stamp_disc(data, SIZE * 2 / 3, SIZE / 3, 22, 0.55);
    }
}

pub(crate) struct WaveState {
    pub(crate) height: Field,
    pub(crate) velocity: Field,
    pub(crate) running: bool,
    pub(crate) preset: WavePreset,
    pub(crate) image: Vec<u8>,
}

impl WaveState {
    pub(crate) fn new() -> Self {
        let mut state = Self {
            height: Field::filled(0.0),
            velocity: Field::filled(0.0),
            running: true,
            preset: WavePreset::Medium,
            image: vec![0; SIZE * SIZE * 4],
        };
        state.reset();
        state
    }

    pub(crate) fn reset(&mut self) {
        self.height.as_mut_slice().fill(0.0);
        self.velocity.as_mut_slice().fill(0.0);
        stamp_disc(self.height.as_mut_slice(), SIZE / 2, SIZE / 2, 22, 0.8);
        stamp_disc(
            self.height.as_mut_slice(),
            SIZE / 3,
            SIZE * 2 / 3,
            14,
            -0.55,
        );
    }
}

pub(crate) struct LifeState {
    pub(crate) field: Field,
    pub(crate) running: bool,
    pub(crate) image: Vec<u8>,
}

impl LifeState {
    pub(crate) fn new() -> Self {
        let mut state = Self {
            field: Field::filled(0.0),
            running: true,
            image: vec![0; SIZE * SIZE * 4],
        };
        state.reset();
        state
    }

    pub(crate) fn reset(&mut self) {
        let data = self.field.as_mut_slice();
        data.fill(0.0);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let hash = (x * 73 + y * 151 + x * y * 17) % 997;
                if hash < 118 {
                    data[y * SIZE + x] = 1.0;
                }
            }
        }
        stamp_life_cells(data, 64, 64, 2, 1.0);
        stamp_life_cells(data, SIZE / 2, SIZE / 2, 3, 1.0);
    }

    pub(crate) fn clear(&mut self) {
        self.field.as_mut_slice().fill(0.0);
    }
}

pub(crate) struct ParticleState {
    pub(crate) x: ParticleVec,
    pub(crate) y: ParticleVec,
    pub(crate) vx: ParticleVec,
    pub(crate) vy: ParticleVec,
    pub(crate) running: bool,
    pub(crate) trails: bool,
    pub(crate) preset: ParticlePreset,
    pub(crate) trail: Vec<f32>,
    pub(crate) image: Vec<u8>,
}

impl ParticleState {
    pub(crate) fn new() -> AppResult<Self> {
        let mut state = Self {
            x: ParticleVec::filled(0.0),
            y: ParticleVec::filled(0.0),
            vx: ParticleVec::filled(0.0),
            vy: ParticleVec::filled(0.0),
            running: true,
            trails: true,
            preset: ParticlePreset::Gentle,
            trail: vec![0.0; SIZE * SIZE],
            image: vec![0; SIZE * SIZE * 4],
        };
        state.reset()?;
        Ok(state)
    }

    pub(crate) fn reset(&mut self) -> AppResult<()> {
        let mut x = Vec::with_capacity(PARTICLES);
        let mut y = Vec::with_capacity(PARTICLES);
        let mut vx = Vec::with_capacity(PARTICLES);
        let mut vy = Vec::with_capacity(PARTICLES);
        let golden = 2.3999631_f32;
        for i in 0..PARTICLES {
            let t = i as f32 / PARTICLES as f32;
            let radius = 0.08 + 0.72 * t.sqrt();
            let angle = i as f32 * golden;
            let px = radius * angle.cos();
            let py = radius * angle.sin();
            x.push(px);
            y.push(py);
            vx.push(-py * 0.18);
            vy.push(px * 0.18);
        }
        self.x = ParticleVec::from_vec(x).into_app_result()?;
        self.y = ParticleVec::from_vec(y).into_app_result()?;
        self.vx = ParticleVec::from_vec(vx).into_app_result()?;
        self.vy = ParticleVec::from_vec(vy).into_app_result()?;
        self.trail.fill(0.0);
        Ok(())
    }

    pub(crate) fn normalize(&mut self) -> AppResult<()> {
        let mut x = self.x.clone().into_vec();
        let mut y = self.y.clone().into_vec();
        let mut vx = self.vx.clone().into_vec();
        let mut vy = self.vy.clone().into_vec();
        for i in 0..PARTICLES {
            if !x[i].is_finite() || !y[i].is_finite() || !vx[i].is_finite() || !vy[i].is_finite() {
                self.reset()?;
                return Ok(());
            }
            if x[i] > 1.0 {
                x[i] -= 2.0;
            } else if x[i] < -1.0 {
                x[i] += 2.0;
            }
            if y[i] > 1.0 {
                y[i] -= 2.0;
            } else if y[i] < -1.0 {
                y[i] += 2.0;
            }
            vx[i] = vx[i].clamp(-2.0, 2.0);
            vy[i] = vy[i].clamp(-2.0, 2.0);
        }
        self.x = ParticleVec::from_vec(x).into_app_result()?;
        self.y = ParticleVec::from_vec(y).into_app_result()?;
        self.vx = ParticleVec::from_vec(vx).into_app_result()?;
        self.vy = ParticleVec::from_vec(vy).into_app_result()?;
        Ok(())
    }
}

pub(crate) struct Hud {
    pub(crate) fps_ema: f32,
    pub(crate) graph_ms: f32,
    pub(crate) frame_ms: f32,
    pub(crate) error: Option<String>,
}

impl Hud {
    pub(crate) fn new() -> Self {
        Self {
            fps_ema: 0.0,
            graph_ms: 0.0,
            frame_ms: 0.0,
            error: None,
        }
    }
}

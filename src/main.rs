use std::time::Instant;

use eframe::egui::{
    self, ColorImage, ComboBox, Context, Response, Sense, TextureHandle, TextureOptions,
};
use knok::{
    tensor::{Tensor1, Tensor2},
    Backend, Engine,
};

const SIZE: usize = 1024;
const PARTICLES: usize = 256;

type Field = Tensor2<f32, SIZE, SIZE>;
type ParticleVec = Tensor1<f32, PARTICLES>;
type AppResult<T> = std::result::Result<T, String>;

trait IntoAppResult<T> {
    fn into_app_result(self) -> AppResult<T>;
}

impl<T, E: ToString> IntoAppResult<T> for std::result::Result<T, E> {
    fn into_app_result(self) -> AppResult<T> {
        self.map_err(|error| error.to_string())
    }
}

knok::generated_graphs!(pub mod cpu_graphs, "knok_cpu_graphs.rs");

#[cfg(feature = "vulkan")]
knok::generated_graphs!(pub mod vulkan_graphs, "knok_vulkan_graphs.rs");

#[cfg(feature = "cuda")]
knok::generated_graphs!(pub mod cuda_graphs, "knok_cuda_graphs.rs");

#[cfg(target_os = "macos")]
knok::generated_graphs!(pub mod metal_graphs, "knok_metal_graphs.rs");

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1180.0, 820.0]),
        ..Default::default()
    };
    eframe::run_native(
        "knok tensor demo",
        options,
        Box::new(|creation| Ok(Box::new(DemoApp::new(creation)))),
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Tab {
    Mandelbrot,
    Heat,
    Wave,
    Life,
    Particles,
}

impl Tab {
    const ALL: [Self; 5] = [
        Self::Mandelbrot,
        Self::Heat,
        Self::Wave,
        Self::Life,
        Self::Particles,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Mandelbrot => "Mandelbrot",
            Self::Heat => "Heat",
            Self::Wave => "Wave",
            Self::Life => "Life",
            Self::Particles => "Particles",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackendChoice {
    Cpu,
    #[cfg(feature = "vulkan")]
    Vulkan,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(target_os = "macos")]
    Metal,
}

impl BackendChoice {
    fn available() -> Vec<Self> {
        let mut backends = vec![Self::Cpu];
        #[cfg(feature = "vulkan")]
        backends.push(Self::Vulkan);
        #[cfg(feature = "cuda")]
        backends.push(Self::Cuda);
        #[cfg(target_os = "macos")]
        backends.push(Self::Metal);
        backends
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            #[cfg(feature = "vulkan")]
            Self::Vulkan => "Vulkan",
            #[cfg(feature = "cuda")]
            Self::Cuda => "CUDA",
            #[cfg(target_os = "macos")]
            Self::Metal => "Metal",
        }
    }

    fn driver(self) -> &'static str {
        match self {
            Self::Cpu => "local-task",
            #[cfg(feature = "vulkan")]
            Self::Vulkan => "vulkan",
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
            #[cfg(target_os = "macos")]
            Self::Metal => "metal",
        }
    }

    fn backend(self) -> Backend {
        match self {
            Self::Cpu => Backend::LlvmCpu,
            #[cfg(feature = "vulkan")]
            Self::Vulkan => Backend::VulkanSpirv,
            #[cfg(feature = "cuda")]
            Self::Cuda => Backend::Cuda,
            #[cfg(target_os = "macos")]
            Self::Metal => Backend::MetalSpirv,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MandelbrotIterations {
    Low,
    Medium,
    High,
}

impl MandelbrotIterations {
    const ALL: [Self; 3] = [Self::Low, Self::Medium, Self::High];

    fn name(self) -> &'static str {
        match self {
            Self::Low => "24",
            Self::Medium => "48",
            Self::High => "72",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DiffusionPreset {
    Low,
    Medium,
    High,
}

impl DiffusionPreset {
    const ALL: [Self; 3] = [Self::Low, Self::Medium, Self::High];

    fn name(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WavePreset {
    Slow,
    Medium,
    Fast,
}

impl WavePreset {
    const ALL: [Self; 3] = [Self::Slow, Self::Medium, Self::Fast];

    fn name(self) -> &'static str {
        match self {
            Self::Slow => "Slow",
            Self::Medium => "Medium",
            Self::Fast => "Fast",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParticlePreset {
    Gentle,
    Strong,
}

impl ParticlePreset {
    const ALL: [Self; 2] = [Self::Gentle, Self::Strong];

    fn name(self) -> &'static str {
        match self {
            Self::Gentle => "Gentle",
            Self::Strong => "Strong",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ColorMap {
    Fire,
    Viridis,
    Ice,
}

impl ColorMap {
    const ALL: [Self; 3] = [Self::Fire, Self::Viridis, Self::Ice];

    fn name(self) -> &'static str {
        match self {
            Self::Fire => "Fire",
            Self::Viridis => "Viridis",
            Self::Ice => "Ice",
        }
    }
}

struct EngineSlot {
    engine: Option<Engine>,
    error: Option<String>,
}

impl EngineSlot {
    fn new() -> Self {
        Self {
            engine: None,
            error: None,
        }
    }

    fn get(&mut self, backend: Backend) -> std::result::Result<&Engine, String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.engine.is_none() {
            match Engine::for_backend(backend) {
                Ok(engine) => self.engine = Some(engine),
                Err(error) => {
                    let message = error.to_string();
                    self.error = Some(message.clone());
                    return Err(message);
                }
            }
        }
        Ok(self.engine.as_ref().expect("engine was initialized"))
    }
}

struct EngineCache {
    cpu: EngineSlot,
    #[cfg(feature = "vulkan")]
    vulkan: EngineSlot,
    #[cfg(feature = "cuda")]
    cuda: EngineSlot,
    #[cfg(target_os = "macos")]
    metal: EngineSlot,
}

impl EngineCache {
    fn new() -> Self {
        Self {
            cpu: EngineSlot::new(),
            #[cfg(feature = "vulkan")]
            vulkan: EngineSlot::new(),
            #[cfg(feature = "cuda")]
            cuda: EngineSlot::new(),
            #[cfg(target_os = "macos")]
            metal: EngineSlot::new(),
        }
    }

    fn get(&mut self, backend: BackendChoice) -> std::result::Result<&Engine, String> {
        match backend {
            BackendChoice::Cpu => self.cpu.get(backend.backend()),
            #[cfg(feature = "vulkan")]
            BackendChoice::Vulkan => self.vulkan.get(backend.backend()),
            #[cfg(feature = "cuda")]
            BackendChoice::Cuda => self.cuda.get(backend.backend()),
            #[cfg(target_os = "macos")]
            BackendChoice::Metal => self.metal.get(backend.backend()),
        }
    }
}

struct MandelbrotState {
    center: [f32; 2],
    scale: f32,
    iterations: MandelbrotIterations,
    color_map: ColorMap,
    image: Vec<u8>,
    dirty: bool,
}

impl MandelbrotState {
    fn new() -> Self {
        Self {
            center: [-0.5, 0.0],
            scale: 3.0,
            iterations: MandelbrotIterations::Medium,
            color_map: ColorMap::Fire,
            image: vec![0; SIZE * SIZE * 4],
            dirty: true,
        }
    }

    fn reset(&mut self) {
        self.center = [-0.5, 0.0];
        self.scale = 3.0;
        self.dirty = true;
    }

    fn grids(&self) -> AppResult<(Field, Field)> {
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

struct HeatState {
    field: Field,
    running: bool,
    preset: DiffusionPreset,
    image: Vec<u8>,
}

impl HeatState {
    fn new() -> Self {
        let mut state = Self {
            field: Field::filled(0.0),
            running: true,
            preset: DiffusionPreset::Medium,
            image: vec![0; SIZE * SIZE * 4],
        };
        state.reset();
        state
    }

    fn reset(&mut self) {
        let data = self.field.as_mut_slice();
        data.fill(0.0);
        stamp_disc(data, SIZE / 2, SIZE / 2, 34, 1.0);
        stamp_disc(data, SIZE / 3, SIZE / 3, 18, 0.75);
        stamp_disc(data, SIZE * 2 / 3, SIZE / 3, 22, 0.55);
    }
}

struct WaveState {
    height: Field,
    velocity: Field,
    running: bool,
    preset: WavePreset,
    image: Vec<u8>,
}

impl WaveState {
    fn new() -> Self {
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

    fn reset(&mut self) {
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

struct ParticleState {
    x: ParticleVec,
    y: ParticleVec,
    vx: ParticleVec,
    vy: ParticleVec,
    running: bool,
    trails: bool,
    preset: ParticlePreset,
    trail: Vec<f32>,
    image: Vec<u8>,
}

struct LifeState {
    field: Field,
    running: bool,
    image: Vec<u8>,
}

impl LifeState {
    fn new() -> Self {
        let mut state = Self {
            field: Field::filled(0.0),
            running: true,
            image: vec![0; SIZE * SIZE * 4],
        };
        state.reset();
        state
    }

    fn reset(&mut self) {
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

    fn clear(&mut self) {
        self.field.as_mut_slice().fill(0.0);
    }
}

impl ParticleState {
    fn new() -> AppResult<Self> {
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

    fn reset(&mut self) -> AppResult<()> {
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
}

struct Hud {
    fps_ema: f32,
    graph_ms: f32,
    frame_ms: f32,
    error: Option<String>,
}

impl Hud {
    fn new() -> Self {
        Self {
            fps_ema: 0.0,
            graph_ms: 0.0,
            frame_ms: 0.0,
            error: None,
        }
    }
}

struct DemoApp {
    tab: Tab,
    backend: BackendChoice,
    engines: EngineCache,
    mandelbrot: MandelbrotState,
    heat: HeatState,
    wave: WaveState,
    life: LifeState,
    particles: ParticleState,
    texture: Option<TextureHandle>,
    hud: Hud,
    last_frame: Instant,
}

impl DemoApp {
    fn new(_: &eframe::CreationContext<'_>) -> Self {
        Self {
            tab: Tab::Mandelbrot,
            backend: BackendChoice::Cpu,
            engines: EngineCache::new(),
            mandelbrot: MandelbrotState::new(),
            heat: HeatState::new(),
            wave: WaveState::new(),
            life: LifeState::new(),
            particles: ParticleState::new().expect("initial particle state is valid"),
            texture: None,
            hud: Hud::new(),
            last_frame: Instant::now(),
        }
    }

    fn step(&mut self) {
        let engine = match self.engines.get(self.backend) {
            Ok(engine) => engine,
            Err(error) => {
                self.hud.error = Some(format!("{} unavailable: {error}", self.backend.name()));
                return;
            }
        };

        let graph_start = Instant::now();
        let result: AppResult<()> = (|| match self.tab {
            Tab::Mandelbrot if self.mandelbrot.dirty => {
                let (x, y) = self.mandelbrot.grids()?;
                let output = run_mandelbrot(self.backend, self.mandelbrot.iterations, engine, x, y)
                    .into_app_result()?;
                render_field(
                    output.as_slice(),
                    self.mandelbrot.color_map,
                    &mut self.mandelbrot.image,
                );
                self.mandelbrot.dirty = false;
                Ok(())
            }
            Tab::Mandelbrot => Ok(()),
            Tab::Heat if self.heat.running => {
                self.heat.field = run_heat(
                    self.backend,
                    self.heat.preset,
                    engine,
                    self.heat.field.clone(),
                )
                .into_app_result()?;
                render_field(
                    self.heat.field.as_slice(),
                    ColorMap::Fire,
                    &mut self.heat.image,
                );
                Ok(())
            }
            Tab::Heat => {
                render_field(
                    self.heat.field.as_slice(),
                    ColorMap::Fire,
                    &mut self.heat.image,
                );
                Ok(())
            }
            Tab::Wave if self.wave.running => {
                let (height, velocity) = run_wave(
                    self.backend,
                    self.wave.preset,
                    engine,
                    self.wave.height.clone(),
                    self.wave.velocity.clone(),
                )
                .into_app_result()?;
                self.wave.height = height;
                self.wave.velocity = velocity;
                render_signed_field(self.wave.height.as_slice(), &mut self.wave.image);
                Ok(())
            }
            Tab::Wave => {
                render_signed_field(self.wave.height.as_slice(), &mut self.wave.image);
                Ok(())
            }
            Tab::Life if self.life.running => {
                self.life.field =
                    run_life(self.backend, engine, self.life.field.clone()).into_app_result()?;
                render_life(self.life.field.as_slice(), &mut self.life.image);
                Ok(())
            }
            Tab::Life => {
                render_life(self.life.field.as_slice(), &mut self.life.image);
                Ok(())
            }
            Tab::Particles if self.particles.running => {
                let (x, y, vx, vy) = run_particles(
                    self.backend,
                    self.particles.preset,
                    engine,
                    self.particles.x.clone(),
                    self.particles.y.clone(),
                    self.particles.vx.clone(),
                    self.particles.vy.clone(),
                )
                .into_app_result()?;
                self.particles.x = x;
                self.particles.y = y;
                self.particles.vx = vx;
                self.particles.vy = vy;
                normalize_particles(&mut self.particles)?;
                render_particles(&mut self.particles);
                Ok(())
            }
            Tab::Particles => {
                render_particles(&mut self.particles);
                Ok(())
            }
        })();

        match result {
            Ok(()) => {
                self.hud.graph_ms = graph_start.elapsed().as_secs_f32() * 1000.0;
                self.hud.error = None;
            }
            Err(error) => {
                self.hud.error = Some(error.to_string());
            }
        }
    }

    fn top_bar(&mut self, ctx: &Context) {
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                for tab in Tab::ALL {
                    if ui.selectable_label(self.tab == tab, tab.name()).clicked() {
                        self.tab = tab;
                    }
                }
                ui.separator();
                ComboBox::from_id_salt("backend")
                    .selected_text(self.backend.name())
                    .show_ui(ui, |ui| {
                        for backend in BackendChoice::available() {
                            ui.selectable_value(&mut self.backend, backend, backend.name());
                        }
                    });
            });
        });
    }

    fn hud_panel(&self, ctx: &Context) {
        egui::SidePanel::right("hud")
            .resizable(false)
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("HUD");
                ui.separator();
                ui.label(format!("Demo: {}", self.tab.name()));
                ui.label(format!("Backend: {}", self.backend.name()));
                ui.label(format!("Driver: {}", self.backend.driver()));
                ui.label(format!("FPS: {:.1}", self.hud.fps_ema));
                ui.label(format!("Graph: {:.2} ms", self.hud.graph_ms));
                ui.label(format!("Frame: {:.2} ms", self.hud.frame_ms));
                ui.label(format!("Resolution: {}x{}", SIZE, SIZE));
                if let Some(error) = &self.hud.error {
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(210, 80, 70), error);
                }
            });
    }

    fn central_panel(&mut self, ctx: &Context) {
        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Mandelbrot => self.mandelbrot_ui(ui),
            Tab::Heat => self.heat_ui(ui),
            Tab::Wave => self.wave_ui(ui),
            Tab::Life => self.life_ui(ui),
            Tab::Particles => self.particles_ui(ui),
        });
    }

    fn mandelbrot_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Reset").clicked() {
                self.mandelbrot.reset();
            }
            ComboBox::from_id_salt("mandelbrot_iterations")
                .selected_text(self.mandelbrot.iterations.name())
                .show_ui(ui, |ui| {
                    for preset in MandelbrotIterations::ALL {
                        if ui
                            .selectable_value(
                                &mut self.mandelbrot.iterations,
                                preset,
                                preset.name(),
                            )
                            .clicked()
                        {
                            self.mandelbrot.dirty = true;
                        }
                    }
                });
            ComboBox::from_id_salt("mandelbrot_colormap")
                .selected_text(self.mandelbrot.color_map.name())
                .show_ui(ui, |ui| {
                    for color_map in ColorMap::ALL {
                        if ui
                            .selectable_value(
                                &mut self.mandelbrot.color_map,
                                color_map,
                                color_map.name(),
                            )
                            .clicked()
                        {
                            self.mandelbrot.dirty = true;
                        }
                    }
                });
            ui.label(format!("Scale {:.4}", self.mandelbrot.scale));
        });

        let response = show_texture(ui, &mut self.texture, &self.mandelbrot.image);
        if response.dragged() {
            let delta = ui.input(|input| input.pointer.delta());
            let side = response.rect.width().max(1.0);
            self.mandelbrot.center[0] -= delta.x / side * self.mandelbrot.scale;
            self.mandelbrot.center[1] -= delta.y / side * self.mandelbrot.scale;
            self.mandelbrot.dirty = true;
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll.abs() > f32::EPSILON {
                self.mandelbrot.scale *= if scroll > 0.0 { 0.88 } else { 1.14 };
                self.mandelbrot.scale = self.mandelbrot.scale.clamp(0.00002, 5.0);
                self.mandelbrot.dirty = true;
            }
        }
    }

    fn heat_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button(if self.heat.running { "Pause" } else { "Play" })
                .clicked()
            {
                self.heat.running = !self.heat.running;
            }
            if ui.button("Reset").clicked() {
                self.heat.reset();
            }
            ComboBox::from_id_salt("diffusion")
                .selected_text(self.heat.preset.name())
                .show_ui(ui, |ui| {
                    for preset in DiffusionPreset::ALL {
                        ui.selectable_value(&mut self.heat.preset, preset, preset.name());
                    }
                });
        });

        let response = show_texture(ui, &mut self.texture, &self.heat.image);
        if (response.dragged() || response.clicked())
            && ui.input(|input| input.pointer.primary_down())
        {
            if let Some((x, y)) = pointer_pixel(&response) {
                stamp_disc(self.heat.field.as_mut_slice(), x, y, 18, 1.0);
            }
        }
    }

    fn wave_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button(if self.wave.running { "Pause" } else { "Play" })
                .clicked()
            {
                self.wave.running = !self.wave.running;
            }
            if ui.button("Reset").clicked() {
                self.wave.reset();
            }
            ComboBox::from_id_salt("wave")
                .selected_text(self.wave.preset.name())
                .show_ui(ui, |ui| {
                    for preset in WavePreset::ALL {
                        ui.selectable_value(&mut self.wave.preset, preset, preset.name());
                    }
                });
        });

        let response = show_texture(ui, &mut self.texture, &self.wave.image);
        if response.clicked() || response.dragged() {
            if let Some((x, y)) = pointer_pixel(&response) {
                stamp_disc(self.wave.height.as_mut_slice(), x, y, 16, 0.75);
                stamp_disc(self.wave.velocity.as_mut_slice(), x, y, 16, 0.15);
            }
        }
    }

    fn particles_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button(if self.particles.running {
                    "Pause"
                } else {
                    "Play"
                })
                .clicked()
            {
                self.particles.running = !self.particles.running;
            }
            if ui.button("Reset").clicked() {
                if let Err(error) = self.particles.reset() {
                    self.hud.error = Some(error.to_string());
                }
            }
            ui.checkbox(&mut self.particles.trails, "Trails");
            ComboBox::from_id_salt("particles")
                .selected_text(self.particles.preset.name())
                .show_ui(ui, |ui| {
                    for preset in ParticlePreset::ALL {
                        ui.selectable_value(&mut self.particles.preset, preset, preset.name());
                    }
                });
        });

        show_texture(ui, &mut self.texture, &self.particles.image);
    }

    fn life_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button(if self.life.running { "Pause" } else { "Play" })
                .clicked()
            {
                self.life.running = !self.life.running;
            }
            if ui.button("Reset").clicked() {
                self.life.reset();
            }
            if ui.button("Clear").clicked() {
                self.life.clear();
            }
        });

        let response = show_texture(ui, &mut self.texture, &self.life.image);
        if response.dragged() || response.clicked() {
            if let Some((x, y)) = pointer_pixel(&response) {
                let erase = ui.input(|input| input.pointer.secondary_down());
                stamp_life_cells(
                    self.life.field.as_mut_slice(),
                    x,
                    y,
                    4,
                    if erase { 0.0 } else { 1.0 },
                );
            }
        }
    }
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        let frame_start = Instant::now();
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        if dt > 0.0 {
            let fps = 1.0 / dt;
            self.hud.fps_ema = if self.hud.fps_ema == 0.0 {
                fps
            } else {
                self.hud.fps_ema * 0.9 + fps * 0.1
            };
        }

        self.step();
        self.top_bar(ctx);
        self.hud_panel(ctx);
        self.central_panel(ctx);
        self.hud.frame_ms = frame_start.elapsed().as_secs_f32() * 1000.0;

        if matches!(self.tab, Tab::Heat | Tab::Wave | Tab::Life | Tab::Particles)
            || self.mandelbrot.dirty
        {
            ctx.request_repaint();
        }
    }
}

fn show_texture(ui: &mut egui::Ui, texture: &mut Option<TextureHandle>, rgba: &[u8]) -> Response {
    let image = ColorImage::from_rgba_unmultiplied([SIZE, SIZE], rgba);
    if let Some(texture) = texture {
        texture.set(image, TextureOptions::NEAREST);
    } else {
        *texture = Some(
            ui.ctx()
                .load_texture("knok-demo", image, TextureOptions::NEAREST),
        );
    }

    let texture = texture.as_ref().expect("texture was initialized");
    let available = ui.available_size();
    let side = available.x.min(available.y).max(128.0);
    ui.add(egui::Image::new((texture.id(), egui::vec2(side, side))).sense(Sense::click_and_drag()))
}

fn pointer_pixel(response: &Response) -> Option<(usize, usize)> {
    let pos = response.interact_pointer_pos()?;
    if !response.rect.contains(pos) {
        return None;
    }
    let offset = pos - response.rect.min;
    let width = response.rect.width().max(1.0);
    let height = response.rect.height().max(1.0);
    let x = ((offset.x / width) * SIZE as f32).floor() as usize;
    let y = ((offset.y / height) * SIZE as f32).floor() as usize;
    Some((x.min(SIZE - 1), y.min(SIZE - 1)))
}

fn stamp_disc(field: &mut [f32], center_x: usize, center_y: usize, radius: usize, value: f32) {
    let radius2 = (radius * radius) as isize;
    let min_y = center_y.saturating_sub(radius);
    let max_y = (center_y + radius).min(SIZE - 1);
    let min_x = center_x.saturating_sub(radius);
    let max_x = (center_x + radius).min(SIZE - 1);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = x as isize - center_x as isize;
            let dy = y as isize - center_y as isize;
            if dx * dx + dy * dy <= radius2 {
                let index = y * SIZE + x;
                field[index] = (field[index] + value).clamp(-1.0, 1.0);
            }
        }
    }
}

fn stamp_life_cells(
    field: &mut [f32],
    center_x: usize,
    center_y: usize,
    radius: usize,
    value: f32,
) {
    let min_y = center_y.saturating_sub(radius);
    let max_y = (center_y + radius).min(SIZE - 1);
    let min_x = center_x.saturating_sub(radius);
    let max_x = (center_x + radius).min(SIZE - 1);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            field[y * SIZE + x] = value;
        }
    }
}

fn render_field(values: &[f32], map: ColorMap, image: &mut [u8]) {
    for (value, px) in values.iter().zip(image.chunks_exact_mut(4)) {
        let [r, g, b] = color_map(value.clamp(0.0, 1.0), map);
        px.copy_from_slice(&[r, g, b, 255]);
    }
}

fn render_life(values: &[f32], image: &mut [u8]) {
    for (value, px) in values.iter().zip(image.chunks_exact_mut(4)) {
        if *value > 0.5 {
            px.copy_from_slice(&[218, 238, 174, 255]);
        } else {
            px.copy_from_slice(&[9, 12, 16, 255]);
        }
    }
}

fn render_signed_field(values: &[f32], image: &mut [u8]) {
    for (value, px) in values.iter().zip(image.chunks_exact_mut(4)) {
        let value = value.clamp(-1.0, 1.0);
        let positive = value.max(0.0);
        let negative = (-value).max(0.0);
        let r = (positive * 255.0) as u8;
        let g = ((1.0 - value.abs()) * 32.0) as u8;
        let b = (negative * 255.0) as u8;
        px.copy_from_slice(&[r, g, b, 255]);
    }
}

fn color_map(value: f32, map: ColorMap) -> [u8; 3] {
    match map {
        ColorMap::Fire => [
            (value.sqrt() * 255.0) as u8,
            ((value * value) * 190.0) as u8,
            ((value * value * value) * 80.0) as u8,
        ],
        ColorMap::Viridis => {
            let r = (68.0 + 185.0 * value.powf(1.7)) as u8;
            let g = (1.0 + 230.0 * (1.0 - (value - 0.55).abs() * 1.25).clamp(0.0, 1.0)) as u8;
            let b = (84.0 + 60.0 * (1.0 - value) + 35.0 * value) as u8;
            [r, g, b]
        }
        ColorMap::Ice => [
            (35.0 + 90.0 * value) as u8,
            (80.0 + 150.0 * value.sqrt()) as u8,
            (130.0 + 125.0 * value) as u8,
        ],
    }
}

fn render_particles(state: &mut ParticleState) {
    if state.trails {
        for value in &mut state.trail {
            *value *= 0.94;
        }
    } else {
        state.trail.fill(0.0);
    }

    for (&x, &y) in state.x.as_slice().iter().zip(state.y.as_slice()) {
        let px = (((x + 1.0) * 0.5) * (SIZE - 1) as f32) as isize;
        let py = (((y + 1.0) * 0.5) * (SIZE - 1) as f32) as isize;
        for oy in -2..=2 {
            for ox in -2..=2 {
                let tx = px + ox;
                let ty = py + oy;
                if tx >= 0 && ty >= 0 && tx < SIZE as isize && ty < SIZE as isize {
                    let index = ty as usize * SIZE + tx as usize;
                    state.trail[index] = 1.0;
                }
            }
        }
    }

    for (value, px) in state.trail.iter().zip(state.image.chunks_exact_mut(4)) {
        let value = value.clamp(0.0, 1.0);
        px.copy_from_slice(&[
            (40.0 + value * 215.0) as u8,
            (50.0 + value.sqrt() * 180.0) as u8,
            (64.0 + value * 70.0) as u8,
            255,
        ]);
    }
}

fn normalize_particles(state: &mut ParticleState) -> AppResult<()> {
    let mut x = state.x.clone().into_vec();
    let mut y = state.y.clone().into_vec();
    let mut vx = state.vx.clone().into_vec();
    let mut vy = state.vy.clone().into_vec();
    for i in 0..PARTICLES {
        if !x[i].is_finite() || !y[i].is_finite() || !vx[i].is_finite() || !vy[i].is_finite() {
            state.reset()?;
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
    state.x = ParticleVec::from_vec(x).into_app_result()?;
    state.y = ParticleVec::from_vec(y).into_app_result()?;
    state.vx = ParticleVec::from_vec(vx).into_app_result()?;
    state.vy = ParticleVec::from_vec(vy).into_app_result()?;
    Ok(())
}

fn run_mandelbrot(
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

fn run_heat(
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

fn run_wave(
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

fn run_life(backend: BackendChoice, engine: &Engine, field: Field) -> knok::Result<Field> {
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

fn run_particles(
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

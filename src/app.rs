use std::time::Instant;

use eframe::egui::{self, ComboBox, Context, TextureHandle};

use crate::{
    backend::{BackendChoice, EngineCache},
    field_ops::{stamp_disc, stamp_life_cells},
    ndarray_runner,
    presets::{ColorMap, DiffusionPreset, MandelbrotIterations, ParticlePreset, Tab, WavePreset},
    render::{
        pointer_pixel, render_field, render_life, render_particles, render_signed_field,
        show_texture,
    },
    runner::{run_heat, run_life, run_mandelbrot, run_particles, run_wave},
    state::{HeatState, Hud, LifeState, MandelbrotState, ParticleState, WaveState},
    types::{AppResult, IntoAppResult, SIZE},
};

pub(crate) struct DemoApp {
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
    pub(crate) fn new(_: &eframe::CreationContext<'_>) -> Self {
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
        if !self.backend.uses_knok_engine() {
            self.step_ndarray();
            return;
        }

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
                self.particles.normalize()?;
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

    fn step_ndarray(&mut self) {
        let graph_start = Instant::now();
        let result: AppResult<()> = (|| match self.tab {
            Tab::Mandelbrot if self.mandelbrot.dirty => {
                let (x, y) = self.mandelbrot.grids()?;
                let output = ndarray_runner::run_mandelbrot(self.mandelbrot.iterations, x, y)?;
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
                self.heat.field =
                    ndarray_runner::run_heat(self.heat.preset, self.heat.field.clone())?;
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
                let (height, velocity) = ndarray_runner::run_wave(
                    self.wave.preset,
                    self.wave.height.clone(),
                    self.wave.velocity.clone(),
                )?;
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
                self.life.field = ndarray_runner::run_life(self.life.field.clone())?;
                render_life(self.life.field.as_slice(), &mut self.life.image);
                Ok(())
            }
            Tab::Life => {
                render_life(self.life.field.as_slice(), &mut self.life.image);
                Ok(())
            }
            Tab::Particles if self.particles.running => {
                let (x, y, vx, vy) = ndarray_runner::run_particles(
                    self.particles.preset,
                    self.particles.x.clone(),
                    self.particles.y.clone(),
                    self.particles.vx.clone(),
                    self.particles.vy.clone(),
                )?;
                self.particles.x = x;
                self.particles.y = y;
                self.particles.vx = vx;
                self.particles.vy = vy;
                self.particles.normalize()?;
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
                            if ui
                                .selectable_value(&mut self.backend, backend, backend.name())
                                .clicked()
                            {
                                self.mandelbrot.dirty = true;
                            }
                        }
                    });
            });
        });
    }

    fn hud_overlay(&self, ui: &egui::Ui, response: &egui::Response) {
        let text = format!(
            "Demo: {}\nBackend: {}\nDriver: {}\nFPS: {:.1}\nCompute: {:.2} ms\nFrame: {:.2} ms\nResolution: {}x{}",
            self.tab.name(),
            self.backend.name(),
            self.backend.driver(),
            self.hud.fps_ema,
            self.hud.graph_ms,
            self.hud.frame_ms,
            SIZE,
            SIZE
        );

        let margin = 12.0;
        let padding = egui::vec2(10.0, 8.0);
        let max_width = (response.rect.width() - margin * 2.0 - padding.x * 2.0).max(96.0);
        let origin = response.rect.min + egui::vec2(margin, margin);
        let painter = ui.painter();
        let text_color = egui::Color32::from_rgb(235, 241, 247);
        let error_color = egui::Color32::from_rgb(255, 128, 118);
        let font = egui::FontId::monospace(12.0);
        let galley = painter.layout(text, font.clone(), text_color, max_width);
        let error_galley =
            self.hud.error.as_ref().map(|error| {
                painter.layout(format!("Error: {error}"), font, error_color, max_width)
            });

        let galley_size = galley.size();
        let error_size = error_galley
            .as_ref()
            .map(|galley| galley.size())
            .unwrap_or(egui::Vec2::ZERO);
        let error_gap = if error_galley.is_some() { 6.0 } else { 0.0 };
        let content_size = egui::vec2(
            galley_size.x.max(error_size.x),
            galley_size.y + error_gap + error_size.y,
        );
        let background = egui::Rect::from_min_size(origin, content_size + padding * 2.0);

        painter.rect_filled(background, 4.0, egui::Color32::from_black_alpha(165));
        painter.rect_stroke(
            background,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_white_alpha(35)),
        );
        painter.galley(origin + padding, galley, text_color);
        if let Some(error_galley) = error_galley {
            let error_pos = origin + egui::vec2(padding.x, padding.y + galley_size.y + error_gap);
            painter.galley(error_pos, error_galley, error_color);
        }
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
        self.hud_overlay(ui, &response);
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
        self.hud_overlay(ui, &response);
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
        self.hud_overlay(ui, &response);
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

        let response = show_texture(ui, &mut self.texture, &self.particles.image);
        self.hud_overlay(ui, &response);
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
        self.hud_overlay(ui, &response);
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
        self.central_panel(ctx);
        self.hud.frame_ms = frame_start.elapsed().as_secs_f32() * 1000.0;

        if matches!(self.tab, Tab::Heat | Tab::Wave | Tab::Life | Tab::Particles)
            || self.mandelbrot.dirty
        {
            ctx.request_repaint();
        }
    }
}

use eframe::egui::{self, ColorImage, Response, Sense, TextureHandle, TextureOptions};

use crate::{presets::ColorMap, state::ParticleState, types::SIZE};

pub(crate) fn show_texture(
    ui: &mut egui::Ui,
    texture: &mut Option<TextureHandle>,
    rgba: &[u8],
) -> Response {
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

pub(crate) fn pointer_pixel(response: &Response) -> Option<(usize, usize)> {
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

pub(crate) fn render_field(values: &[f32], map: ColorMap, image: &mut [u8]) {
    for (value, px) in values.iter().zip(image.chunks_exact_mut(4)) {
        let [r, g, b] = color_map(value.clamp(0.0, 1.0), map);
        px.copy_from_slice(&[r, g, b, 255]);
    }
}

pub(crate) fn render_life(values: &[f32], image: &mut [u8]) {
    for (value, px) in values.iter().zip(image.chunks_exact_mut(4)) {
        if *value > 0.5 {
            px.copy_from_slice(&[218, 238, 174, 255]);
        } else {
            px.copy_from_slice(&[9, 12, 16, 255]);
        }
    }
}

pub(crate) fn render_signed_field(values: &[f32], image: &mut [u8]) {
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

pub(crate) fn render_particles(state: &mut ParticleState) {
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

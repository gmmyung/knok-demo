mod app;
mod backend;
mod field_ops;
mod ndarray_runner;
mod presets;
mod render;
mod runner;
mod state;
mod types;

knok::generated_graphs!(pub mod cpu_graphs, "knok_cpu_graphs.rs");

#[cfg(feature = "vulkan")]
knok::generated_graphs!(pub mod vulkan_graphs, "knok_vulkan_graphs.rs");

#[cfg(feature = "cuda")]
knok::generated_graphs!(pub mod cuda_graphs, "knok_cuda_graphs.rs");

#[cfg(target_os = "macos")]
knok::generated_graphs!(pub mod metal_graphs, "knok_metal_graphs.rs");

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1180.0, 820.0]),
        ..Default::default()
    };
    eframe::run_native(
        "knok tensor demo",
        options,
        Box::new(|creation| Ok(Box::new(app::DemoApp::new(creation)))),
    )
}

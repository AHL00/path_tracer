use std::time::{Duration, Instant};

use egui_winit_vulkano::egui::{self};
use path_tracer::{
    graphics::VulkanContext,
    renderer::Renderer,
    scene::Scene,
};
use simple_logger::SimpleLogger;
use vulkano::{
    Validated, VulkanError,
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    image::{
        ImageAspects, ImageSubresourceRange, ImageUsage,
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    swapchain::{SwapchainPresentInfo, acquire_next_image},
    sync::GpuFuture,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};

fn main() {
    SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .without_timestamps()
        .with_module_level("vulkano", log::LevelFilter::Warn)
        .init()
        .unwrap();

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let mut app = App::new();

    event_loop.run_app(&mut app).unwrap();
}

pub struct AppStats {
    pub last_second_delta_time: Duration,

    _last_deltas: Vec<f32>,
    _last_second: Instant,
    _last_frame_start: Instant,
}

impl AppStats {
    pub fn new() -> Self {
        Self {
            last_second_delta_time: Duration::ZERO,
            _last_deltas: Vec::new(),
            _last_second: Instant::now(),
            _last_frame_start: Instant::now(),
        }
    }

    pub fn update_frame_start(&mut self) {
        let delta_time = self._last_frame_start.elapsed().as_secs_f32();
        self._last_deltas.push(delta_time);
        self._last_frame_start = Instant::now();

        if self._last_second.elapsed().as_secs_f32() > 1.0 {
            self.last_second_delta_time = Duration::from_secs_f32(
                self._last_deltas.iter().sum::<f32>() / self._last_deltas.len() as f32,
            );

            self._last_deltas.clear();
            self._last_second = Instant::now();
        }
    }
}

pub struct App {
    pub context: Option<VulkanContext>,
    pub renderer: Option<Renderer>,

    pub gui: Option<egui_winit_vulkano::Gui>,

    stats: AppStats,

    _queue_recreate_swapchain: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            context: None,
            renderer: None,
            gui: None,
            stats: AppStats::new(),
            _queue_recreate_swapchain: false,
        }
    }

    pub fn gui(&mut self) {
        self.gui.as_mut().unwrap().immediate_ui(|gui| {
            let ctx = gui.context();

            egui::Window::new("Settings")
                .resizable(true)
                .default_width(300.0)
                .show(&ctx, |ui| {
                    ui.label(format!(
                        "FPS: {:.2}",
                        1.0 / self.stats.last_second_delta_time.as_secs_f32()
                    ));

                    ui.label(format!(
                        "Delta Time: {:.2?}",
                        self.stats.last_second_delta_time
                    ));

                    ui.label(format!(
                        "Entities: {}",
                        self.renderer.as_ref().unwrap().scene.world.len()
                    ));

                    ui.label(format!(
                        "Meshes: {}",
                        self.renderer.as_ref().unwrap().scene.geometries_map.len()
                    ));

                    ui.label(format!(
                        "Textures: {}",
                        self.renderer.as_ref().unwrap().loaded_textures_map.len()
                    ));
                });
        });
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Vulkan Window")
                    .with_inner_size(LogicalSize::new(1280, 720))
                    .with_resizable(true),
            )
            .expect("Failed to create window");

        self.context = Some(VulkanContext::new(window));
        let context = self.context.as_ref().unwrap();
        self.gui = Some(egui_winit_vulkano::Gui::new(
            &event_loop,
            context.surface.clone(),
            context.queue.clone(),
            context.swapchain.image_format(),
            egui_winit_vulkano::GuiConfig {
                allow_srgb_render_target: true,
                is_overlay: true,
                ..Default::default()
            },
        ));

        let mut renderer = Renderer::new(&self.context.as_ref().unwrap());

        log::info!("Loading GLTF scene...");

        // Scene::import_gltf(
        //     &mut renderer,
        //     std::path::Path::new("./assets/sponza/Sponza.gltf"),
        //     &self.context.as_ref().unwrap(),
        // )
        // .unwrap();

        Scene::import_gltf(
            &mut renderer,
            std::path::Path::new("./assets/bistro/bistro.gltf"),
            &self.context.as_ref().unwrap(),
        )
        .unwrap();

        // Scene::import_gltf(
        //     &mut renderer,
        //     std::path::Path::new("./assets/cornell/cornell_box.gltf"),
        //     &self.context.as_ref().unwrap(),
        // )
        // .unwrap();

        log::info!("GLTF scene loaded");

        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if self.gui.as_mut().unwrap().update(&event) {
            return;
        };

        match event {
            WindowEvent::Resized(size) => {
                if let Some(context) = &mut self.context {
                    context.wait_for_previous_frame_end();
                    context
                        .recreate_swapchain(self.renderer.as_mut().unwrap(), size)
                        .unwrap();
                }
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.stats.update_frame_start();

                self.gui();

                let context = self.context.as_mut().unwrap();
                let renderer = self.renderer.as_mut().unwrap();

                context.wait_for_previous_frame_end();

                if self._queue_recreate_swapchain {
                    context
                        .recreate_swapchain(renderer, context.swapchain.image_extent().into())
                        .unwrap();
                    self._queue_recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(context.swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            self._queue_recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    self._queue_recreate_swapchain = true;
                }

                // Consolidate this into the renderer struct?
                let mut builder = AutoCommandBufferBuilder::primary(
                    context.command_buffer_allocator.clone(),
                    context.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                renderer.record_commands(image_index, &mut builder, context);

                let command_buffer = builder.build().unwrap();
                let future = context
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(context.queue.clone(), command_buffer)
                    .unwrap();

                let swapchain_image = renderer.swapchain_image_sets[image_index as usize]
                    .0
                    .image();

                let mut after_gui_future = self.gui.as_mut().unwrap().draw_on_image(
                    future,
                    ImageView::new(
                        swapchain_image.clone(),
                        ImageViewCreateInfo {
                            usage: ImageUsage::COLOR_ATTACHMENT,
                            view_type: ImageViewType::Dim2d,
                            format: context.swapchain.image_format(),
                            subresource_range: ImageSubresourceRange {
                                aspects: ImageAspects::COLOR,
                                mip_levels: 0..1,
                                array_layers: 0..1,
                            },

                            ..Default::default()
                        },
                    )
                    .unwrap(),
                );

                after_gui_future.cleanup_finished();

                // self.gui.as_mut().unwrap().

                // let after_gui_future = future;

                let future = after_gui_future
                    .then_swapchain_present(
                        context.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            context.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        context.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        self._queue_recreate_swapchain = true;
                        context.previous_frame_end =
                            Some(vulkano::sync::now(context.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        context.previous_frame_end =
                            Some(vulkano::sync::now(context.device.clone()).boxed());
                    }
                }

                // Prevent background processing that might mess with buffers
                context.wait_for_previous_frame_end();
            }
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
            } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Escape => {
                            event_loop.exit();
                        }
                        KeyCode::Space => {
                            if let Some(context) = &mut self.context {
                                context.winit.request_redraw();
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(context) = &mut self.context {
            context.winit.request_redraw();
        }
    }
}

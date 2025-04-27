use std::{sync::Arc, time::Instant};

use egui_wgpu::{
    RenderState, ScreenDescriptor, WgpuConfiguration, WgpuSetup,
    wgpu::{self, InstanceDescriptor},
};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalPosition, LogicalSize, PhysicalPosition},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopBuilder},
    keyboard::{KeyCode, PhysicalKey},
    platform::{
        pump_events::{EventLoopExtPumpEvents, PumpStatus},
        run_on_demand::EventLoopExtRunOnDemand,
        windows::{EventLoopBuilderExtWindows, WindowAttributesExtWindows},
    },
    window::{self, Window, WindowAttributes},
};

use crate::{PARENT_CHILD_GAP, render_window::RenderApp};

pub struct ConfigApp {
    pub window: Option<Arc<winit::window::Window>>,
    pub render_state: Option<RenderState>,
    pub instance: Option<wgpu::Instance>,
    pub surface: Option<wgpu::Surface<'static>>,
    // pub egui_state: Option<egui_winit::State>,
    pub platform: Option<egui_winit_platform::Platform>,
    pub start_time: Instant,
    pub surface_config: Option<wgpu::SurfaceConfiguration>,
    pub egui_renderer: Option<egui_wgpu::Renderer>,

    pub current_screen: ConfigAppScreen,
    _child_window: Option<Arc<Window>>,

    state: ConfigAppState,
}

pub struct ConfigAppState {
    save_image_oidn: bool,
}

impl Default for ConfigAppState {
    fn default() -> Self {
        Self {
            save_image_oidn: false,
        }
    }
}

impl ConfigApp {
    pub fn new() -> Self {
        Self {
            window: None,
            render_state: None,
            instance: None,
            surface: None,
            // egui_state: None,
            platform: None,
            start_time: Instant::now(),
            surface_config: None,
            egui_renderer: None,
            current_screen: ConfigAppScreen::Stats,
            _child_window: None,
            state: ConfigAppState::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigAppScreen {
    Stats,
    Scene,
    Settings,
    Render,
}

impl ConfigApp {
    pub fn main_screen(
        state: &mut ConfigAppState,
        current_screen: &mut ConfigAppScreen,
        render_app: &mut RenderApp,
        ctx: &egui::Context,
    ) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.selectable_value(current_screen, ConfigAppScreen::Stats, "Stats");
                ui.selectable_value(current_screen, ConfigAppScreen::Settings, "Settings");
                ui.selectable_value(current_screen, ConfigAppScreen::Scene, "Scene");
                ui.selectable_value(current_screen, ConfigAppScreen::Render, "Render");
            });
        });

        match current_screen {
            ConfigAppScreen::Stats => Self::stats(state, render_app, ctx),
            ConfigAppScreen::Scene => Self::scene(state, render_app, ctx),
            ConfigAppScreen::Settings => Self::settings(state, render_app, ctx),
            ConfigAppScreen::Render => Self::render(state, render_app, ctx),
        }
    }

    pub fn scene(state: &mut ConfigAppState, render_app: &mut RenderApp, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {});
    }

    pub fn render(state: &mut ConfigAppState, render_app: &mut RenderApp, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if ui.button("Start Render").clicked() {
                // render_app.start_rendering();
            }

            ui.separator();

            ui.label("Save Image");

            ui.horizontal(|ui| {
                ui.checkbox(&mut state.save_image_oidn, "OpenImageDenoise");
            });

            if ui.button("Save Image").clicked() {
                let save_path = rfd::FileDialog::new()
                    .set_title("Save Image")
                    .set_file_name("render.png")
                    .save_file();

                if let None = save_path {
                    return;
                }

                render_app
                    .renderer
                    .as_ref()
                    .unwrap()
                    .save_render_texture_to_file(
                        render_app.vulkan_context.as_ref().unwrap(),
                        save_path.as_ref().unwrap(),
                        image::ImageFormat::Png,
                        state.save_image_oidn,
                    )
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to save image to {:?}: {}", save_path, e);
                    });
            }
        });
    }

    pub fn settings(state: &mut ConfigAppState, render_app: &mut RenderApp, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Resolution");

            let current_resolution = render_app.renderer.as_ref().unwrap().render_resolution();
            let current_resolution = (current_resolution[0], current_resolution[1]);

            let mut temp_resolution = current_resolution;

            ui.horizontal(|ui| {
                ui.label("Width:");
                ui.add(
                    egui::DragValue::new(&mut temp_resolution.0)
                        .range(1..=4096)
                        .update_while_editing(false),
                );
            });

            ui.horizontal(|ui| {
                ui.label("Height:");
                ui.add(
                    egui::DragValue::new(&mut temp_resolution.1)
                        .range(1..=4096)
                        .update_while_editing(false),
                );
            });

            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("PresetsDropdown")
                    .selected_text(format!("Presets"))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut temp_resolution, (1280, 720), "720p");
                        ui.selectable_value(&mut temp_resolution, (1600, 900), "900p");
                        ui.selectable_value(&mut temp_resolution, (1920, 1080), "1080p");
                        ui.selectable_value(&mut temp_resolution, (2560, 1440), "1440p");
                        ui.selectable_value(&mut temp_resolution, (3840, 2160), "4K");
                    });
            });

            if temp_resolution != current_resolution {
                let _ = render_app.renderer.as_mut().unwrap().resize_render_buffer(
                    render_app.vulkan_context.as_ref().unwrap(),
                    [temp_resolution.0, temp_resolution.1],
                );
            }

            ui.separator();

            ui.label("Render Settings");

            ui.horizontal(|ui| {
                ui.label("Accumulation Count:");
                let mut test = 0;
                ui.add(egui::DragValue::new(&mut test).range(1..=1000));
            });

            ui.horizontal(|ui| {
                ui.label("Max Ray Depth:");
                let mut test = 0;
                ui.add(egui::DragValue::new(&mut test).range(1..=1000));
            });

            ui.horizontal(|ui| {
                ui.label("Max Bounces:");
                let mut test = 0;
                ui.add(egui::DragValue::new(&mut test).range(1..=1000));
            });

            ui.horizontal(|ui| {
                ui.label("Mouse Sensitivity:");
                ui.add(egui::DragValue::new(&mut render_app.mouse_sensitivity).range(0.0..=10.0));
            });

            ui.horizontal(|ui| {
                ui.label("Camera Speed:");
                ui.add(egui::DragValue::new(&mut render_app.move_speed).range(0.0..=100.0));
            });

            ui.horizontal(|ui| {
                ui.label("Camera FOV:");
                let fov_y = render_app.renderer.as_ref().unwrap().camera.fov_y;
                let mut fov_y_deg = fov_y.to_degrees();
                
                ui.add(egui::DragValue::new(&mut fov_y_deg).range(0.0..=180.0));

                render_app.renderer.as_mut().unwrap().camera.fov_y = fov_y_deg.to_radians();
            });

            ui.separator();

            ui.label("Denoising Settings");

            ui.horizontal(|ui| {
                let mut test = false;
                ui.checkbox(&mut test, "Real Time Denoising");
            });
        });
    }

    pub fn stats(state: &mut ConfigAppState, render_app: &mut RenderApp, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(format!(
                "FPS: {:.2}",
                1.0 / render_app.stats.last_second_delta_time.as_secs_f32()
            ));

            ui.label(format!(
                "Delta Time: {:.2?}",
                render_app.stats.last_second_delta_time
            ));

            ui.label(format!(
                "Entities: {}",
                render_app.renderer.as_ref().unwrap().scene.world().len()
            ));

            ui.label(format!(
                "Meshes: {}",
                render_app
                    .renderer
                    .as_ref()
                    .unwrap()
                    .scene
                    .geometries_map
                    .len()
            ));

            ui.label(format!(
                "Textures: {}",
                render_app
                    .renderer
                    .as_ref()
                    .unwrap()
                    .loaded_textures_map
                    .len()
            ));

            ui.label(format!(
                "Materials: {}",
                render_app.renderer.as_ref().unwrap().scene.material_offset
            ));

            ui.label(format!(
                "Camera Position: {:?}",
                render_app
                    .renderer
                    .as_ref()
                    .unwrap()
                    .camera
                    .transform
                    .position
            ));

            ui.label(format!(
                "Accumulation Count: {}",
                render_app.renderer.as_ref().unwrap().accumulated_count
            ));

            ui.label(format!(
                "Raw Mouse Delta: {:?}",
                render_app.input.get_raw_mouse_delta()
            ));
        });
    }

    pub fn redraw(&mut self, event_loop: &ActiveEventLoop, render_app: &mut RenderApp) {
        let render_state = self.render_state.as_ref().unwrap();
        let platform = self.platform.as_mut().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let egui_renderer = self.egui_renderer.as_mut().unwrap();

        platform.update_time(self.start_time.elapsed().as_secs_f64());

        let output_frame = match surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Outdated) => {
                // This error occurs when the app is minimized on Windows.
                // Silently return here to prevent spamming the console with:
                // "The underlying surface has changed, and therefore the swap chain must be updated"
                return;
            }
            Err(e) => {
                eprintln!("Dropped frame with error: {}", e);
                return;
            }
        };
        let output_view = output_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Begin to draw the UI frame.
        platform.begin_pass();

        Self::main_screen(
            &mut self.state,
            &mut self.current_screen,
            render_app,
            &platform.context(),
        );

        // End the UI frame. We could now handle the output and draw the UI with the backend.
        let output = platform.end_pass(Some(&self.window.as_ref().unwrap()));
        let paint_jobs = platform
            .context()
            .tessellate(output.shapes, output.pixels_per_point);

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [output_frame.texture.width(), output_frame.texture.height()],
            pixels_per_point: output.pixels_per_point,
        };

        for texture in output.textures_delta.set.iter() {
            let id = texture.0;
            let delta = &texture.1;
            egui_renderer.update_texture(&render_state.device, &render_state.queue, id, delta);
        }

        let mut encoder =
            render_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

        // egui_renderer.update_
        egui_renderer.update_buffers(
            &render_state.device,
            &render_state.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        {
            let mut render_pass: wgpu::RenderPass<'static> = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();

            egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }

        // Submit the commands.
        render_state.queue.submit(std::iter::once(encoder.finish()));

        // Redraw egui
        output_frame.present();
    }

    pub fn set_child_window(&mut self, window: Arc<Window>) {
        self._child_window = Some(window);
    }
}

impl ApplicationHandler for ConfigApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let size = (400, 300);

        let window = event_loop
            .create_window({
                let mut attribs = WindowAttributes::default()
                    .with_title("Vulkan Path Tracer")
                    .with_inner_size(LogicalSize::new(size.0, size.1))
                    // .with_window_level(window::WindowLevel::AlwaysOnTop)
                    .with_resizable(true);

                #[cfg(target_os = "windows")]
                {
                    attribs = attribs.with_clip_children(false);
                }

                attribs
            })
            .expect("Failed to create window");

        self.window = Some(Arc::new(window));
        let window = self.window.as_ref().unwrap();

        let instance = wgpu::Instance::new(&InstanceDescriptor::default());
        let surface = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(window).unwrap())
                .expect("Failed to create surface")
        };

        let config = WgpuConfiguration::default();

        self.render_state = Some({
            pollster::block_on(RenderState::create(
                &config,
                &instance,
                Some(&surface),
                None,
                1,
                true,
            ))
            .unwrap()
        });

        self.instance = Some(instance);
        self.surface = Some(surface);

        let platform =
            egui_winit_platform::Platform::new(egui_winit_platform::PlatformDescriptor {
                physical_width: window.inner_size().width as u32,
                physical_height: window.inner_size().height as u32,
                scale_factor: window.scale_factor(),
                font_definitions: egui::FontDefinitions::default(),
                style: egui::Style::default(),
            });

        self.platform = Some(platform);

        let renderer = egui_wgpu::Renderer::new(
            &self.render_state.as_ref().unwrap().device,
            self.render_state.as_ref().unwrap().target_format,
            None,
            1,
            true,
        );
        self.egui_renderer = Some(renderer);

        self.surface_config = Some(wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.render_state.as_ref().unwrap().target_format,
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 3,
        });

        self.surface.as_ref().unwrap().configure(
            &self.render_state.as_ref().unwrap().device,
            &self.surface_config.as_ref().unwrap(),
        );
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: window::WindowId,
        event: WindowEvent,
    ) {
        self.platform.as_mut().unwrap().handle_event(&event);

        if self.platform.as_ref().unwrap().captures_event(&event) {
            return;
        }

        match event {
            winit::event::WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.surface_config = Some(wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: self.render_state.as_ref().unwrap().target_format,
                        width: size.width,
                        height: size.height,
                        present_mode: wgpu::PresentMode::AutoVsync,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 3,
                    });

                    self.surface.as_ref().unwrap().configure(
                        &self.render_state.as_ref().unwrap().device,
                        &self.surface_config.as_ref().unwrap(),
                    );
                }
            }
            winit::event::WindowEvent::CloseRequested => {
                _event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                panic!("This is supposed to be handled in main.rs");
            }
            winit::event::WindowEvent::Moved(pos) => {
                if let Some(child_window) = &self._child_window {
                    let window = self.window.as_ref().unwrap();

                    // Move child to self, NOTE: also check render_window creation, the initial position is set there.
                    let parent_size = window.inner_size();
                    let parent_pos = pos.to_logical::<f32>(window.scale_factor());
                    let parent_size = parent_size.to_logical::<f32>(window.scale_factor());
                    let new_pos =
                        LogicalPosition::new(parent_pos.x + parent_size.width, parent_pos.y);
                    let new_pos = new_pos.to_physical::<i32>(window.scale_factor());
                    let new_pos = PhysicalPosition::new(new_pos.x + PARENT_CHILD_GAP, new_pos.y);
                    child_window.set_outer_position(new_pos);
                }
            }
            _ => {}
        }

        // egui_winit::State::new(egui_ctx, viewport_id, display_target, native_pixels_per_point, theme, max_texture_side)
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl ConfigApp {
    pub fn window_id(&self) -> Option<winit::window::WindowId> {
        self.window.as_ref().map(|w| w.id())
    }
}

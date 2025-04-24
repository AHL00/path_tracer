use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use glam::Vec3;
use path_tracer::{graphics::VulkanContext, renderer::Renderer, scene::Scene};
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
    dpi::{LogicalPosition, LogicalSize, PhysicalPosition},
    event::WindowEvent,
    keyboard::{KeyCode, PhysicalKey},
    platform::windows::{WindowAttributesExtWindows, WindowExtWindows, HWND},
    raw_window_handle::RawWindowHandle,
    window::{Window, WindowAttributes, WindowButtons},
};

use crate::PARENT_CHILD_GAP;

pub struct RenderAppStats {
    pub last_second_delta_time: Duration,

    _last_deltas: Vec<f32>,
    _last_second: Instant,
    _last_frame_start: Instant,
}

impl RenderAppStats {
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

pub struct RenderApp {
    pub context: Option<VulkanContext>,
    pub renderer: Option<Renderer>,

    pub stats: RenderAppStats,

    _parent_window: Option<Arc<Window>>,
    _queue_recreate_swapchain: bool,
}

impl RenderApp {
    pub fn new() -> Self {
        Self {
            context: None,
            renderer: None,
            stats: RenderAppStats::new(),

            _parent_window: None,
            _queue_recreate_swapchain: false,
        }
    }

    /// Will not change existing window's parent. Use before `resumed` to set the parent window.
    pub fn set_parent_window(&mut self, window: Arc<Window>) {
        self._parent_window = Some(window.clone());
    }

    pub fn redraw(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.stats.update_frame_start();

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
            match acquire_next_image(context.swapchain.clone(), None).map_err(Validated::unwrap) {
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

        context.winit.pre_present_notify();

        let future = context
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(context.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                context.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(context.swapchain.clone(), image_index),
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
}

impl ApplicationHandler for RenderApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(unsafe {
                let mut attribs = WindowAttributes::default()
                    .with_title("Vulkan Window")
                    .with_inner_size(LogicalSize::new(1280, 720))
                    .with_enabled_buttons(WindowButtons::empty())
                    .with_decorations(false)
                    .with_resizable(true);

                if let Some(parent_window) = self._parent_window.clone() {
                    #[cfg(target_os = "windows")]
                    {
                        let hwnd = {
                            match parent_window
                                .window_handle_any_thread()
                                .expect("Failed to get window handle of parent window")
                                .as_raw()
                            {
                                RawWindowHandle::Win32(win32_handle) => win32_handle.hwnd,
                                _ => {
                                    panic!("Expected window to be a HWND")
                                }
                            }
                        };

                        attribs = attribs.with_owner_window(hwnd.into());
                    }

                    #[cfg(target_os = "linux")]
                    {
                        attribs =
                            attribs.with_parent_window(self._parent_window.clone().map(|w| {
                                w.window_handle_any_thread()
                                    .expect("Failed to get window handle of parent window")
                                    .as_raw()
                            }));
                    }

                    #[cfg(target_os = "macos")]
                    {
                        attribs =
                            attribs.with_parent_window(self._parent_window.clone().map(|w| {
                                w.window_handle_any_thread()
                                    .expect("Failed to get window handle of parent window")
                                    .as_raw()
                            }));
                    }
                }

                attribs
            })
            .expect("Failed to create window");

        if let Some(parent_window) = self._parent_window.clone() {
            // Move self to right of parent window
            let parent_pos = parent_window.outer_position().unwrap();
            let parent_size = parent_window.inner_size();
            let parent_pos = parent_pos.to_logical::<f32>(parent_window.scale_factor());
            let parent_size = parent_size.to_logical::<f32>(parent_window.scale_factor());
            let new_pos = LogicalPosition::new(parent_pos.x + parent_size.width, parent_pos.y);
            let new_pos = new_pos.to_physical::<i32>(parent_window.scale_factor());
            let new_pos =
                PhysicalPosition::new(new_pos.x + PARENT_CHILD_GAP, new_pos.y);
            window.set_outer_position(new_pos);
        }

        self.context = Some(VulkanContext::new(window));

        let mut renderer = Renderer::new(&self.context.as_ref().unwrap());

        log::info!("Loading GLTF scene...");

        // Scene::import_gltf(
        //     &mut renderer,
        //     std::path::Path::new("./assets/sponza/Sponza.gltf"),
        //     &self.context.as_ref().unwrap(),
        // )
        // .unwrap();

        // Scene::import_gltf(
        //     &mut renderer,
        //     std::path::Path::new("./assets/bistro/bistro.gltf"),
        //     &self.context.as_ref().unwrap(),
        // )
        // .unwrap();

        // Scene::import_gltf(
        //     &mut renderer,
        //     std::path::Path::new("./assets/cornell/cornell.gltf"),
        //     &self.context.as_ref().unwrap(),
        // )
        // .unwrap();

        Scene::import_gltf(
            &mut renderer,
            std::path::Path::new("./assets/lion_head_2k/lion_head_2k.gltf"),
            &self.context.as_ref().unwrap(),
            Vec3::new(2.0, 0.0, 0.0),
        )
        .unwrap();

        Scene::import_gltf(
            &mut renderer,
            std::path::Path::new("./assets/boulder_01_2k/boulder_01_2k.gltf"),
            &self.context.as_ref().unwrap(),
            Vec3::new(-2.0, 0.0, 0.0),
        )
        .unwrap();

        Scene::import_gltf(
            &mut renderer,
            std::path::Path::new("./assets/spheres/spheres.gltf"),
            &self.context.as_ref().unwrap(),
            Vec3::ZERO,
        )
        .unwrap();

        renderer.camera.transform.position = [-1.5, 1.0, 5.0].into();

        log::info!("GLTF scene loaded");

        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) => {
                if let Some(context) = &mut self.context {
                    if size.width <= 0 || size.height <= 0 {
                        return;
                    }

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
                panic!("This is supposed to be handled in main.rs");
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
                        KeyCode::KeyW => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.camera.transform.position +=
                                    0.1 * renderer.camera.transform.forward();
                            }
                        }
                        KeyCode::KeyS => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.camera.transform.position -=
                                    0.1 * renderer.camera.transform.forward();
                            }
                        }
                        KeyCode::KeyA => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.camera.transform.position -=
                                    0.1 * renderer.camera.transform.right();
                            }
                        }
                        KeyCode::KeyD => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.camera.transform.position +=
                                    0.1 * renderer.camera.transform.right();
                            }
                        }
                        KeyCode::Space => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.camera.transform.position +=
                                    0.1 * renderer.camera.transform.up();
                            }
                        }
                        KeyCode::KeyC => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.camera.transform.position -=
                                    0.1 * renderer.camera.transform.up();
                            }
                        }
                        KeyCode::ArrowLeft => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.camera.transform.rotation *=
                                    glam::Quat::from_rotation_y(0.1);
                            }
                        }
                        KeyCode::ArrowRight => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.camera.transform.rotation *=
                                    glam::Quat::from_rotation_y(-0.1);
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(context) = &mut self.context {
            context.winit.request_redraw();
        }
    }
}

impl RenderApp {
    pub fn window_id(&self) -> Option<winit::window::WindowId> {
        self.context.as_ref().map(|c| c.winit.id())
    }
}

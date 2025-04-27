use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use glam::Vec3;
use path_tracer::{
    graphics::VulkanContext,
    input::{ButtonState, Input},
    renderer::Renderer,
    scene::Scene,
};
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
    event::{MouseButton, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    platform::windows::{WindowAttributesExtWindows, WindowExtWindows},
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
    pub vulkan_context: Option<VulkanContext>,
    pub renderer: Option<Renderer>,

    pub stats: RenderAppStats,
    pub input: Input,
    pub mouse_sensitivity: f32,
    pub move_speed: f32,

    _mouse_locked: bool,
    _pre_render_update_last: Instant,
    _parent_window: Option<Arc<Window>>,
    _queue_recreate_swapchain: bool,
}

impl RenderApp {
    pub fn new() -> Self {
        Self {
            vulkan_context: None,
            renderer: None,

            stats: RenderAppStats::new(),
            input: Input::new(),
            mouse_sensitivity: 0.2,
            move_speed: 4.0,

            _mouse_locked: false,
            _pre_render_update_last: Instant::now(),
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

        let context = self.vulkan_context.as_mut().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        context.wait_for_previous_frame_end();

        if self._queue_recreate_swapchain {
            context
                .handle_resize_recreate_swap(renderer, context.swapchain.image_extent().into())
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

        // log::debug!(
        //     "Rendered to swapchain image {}",
        //     image_index
        // );

        // Prevent background processing that might mess with buffers
        context.wait_for_previous_frame_end();
    }

    pub fn pre_render_update(&mut self) {
        let now = Instant::now();
        let delta_time = now.duration_since(self._pre_render_update_last);
        self._pre_render_update_last = now;

        if let Some(renderer) = &mut self.renderer {
            match self.input.get_mouse_right() {
                ButtonState::Pressed => {
                    // Toggle mouse lock
                    if self._mouse_locked {
                        self._mouse_locked = false;

                        self.vulkan_context
                            .as_ref()
                            .unwrap()
                            .winit
                            .set_cursor_visible(true);

                        let _ = self
                            .vulkan_context
                            .as_ref()
                            .unwrap()
                            .winit
                            .set_cursor_grab(winit::window::CursorGrabMode::None);
                    } else {
                        self._mouse_locked = true;

                        self.vulkan_context
                            .as_ref()
                            .unwrap()
                            .winit
                            .set_cursor_visible(false);

                        self.vulkan_context
                            .as_ref()
                            .unwrap()
                            .winit
                            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                            .unwrap();
                    }
                }
                ButtonState::Held => {}
                ButtonState::Released => {}
            }

            if self._mouse_locked {
                let delta_s = delta_time.as_secs_f32();
                let mouse_delta = self.input.get_raw_mouse_delta();
                let yaw_delta = -mouse_delta.0 * self.mouse_sensitivity * delta_s;
                let pitch_delta = -mouse_delta.1 * self.mouse_sensitivity * delta_s;

                // Apply yaw around the global Y axis
                renderer.camera.transform.rotation =
                    glam::Quat::from_rotation_y(yaw_delta) * renderer.camera.transform.rotation;
                // Apply pitch around the local X axis
                renderer.camera.transform.rotation *= glam::Quat::from_rotation_x(pitch_delta);
            }

            // Keyboard movement
            let mut current_move_speed = self.move_speed;

            match self.input.get_key_state(KeyCode::ShiftLeft) {
                ButtonState::Pressed | ButtonState::Held => {
                    current_move_speed *= 2.0;
                }
                _ => {}
            }

            let mut move_vector = Vec3::ZERO;

            match self.input.get_key_state(KeyCode::KeyW) {
                ButtonState::Pressed | ButtonState::Held => {
                    move_vector += renderer.camera.transform.forward();
                }
                _ => {}
            }

            match self.input.get_key_state(KeyCode::KeyS) {
                ButtonState::Pressed | ButtonState::Held => {
                    move_vector -= renderer.camera.transform.forward();
                }
                _ => {}
            }

            match self.input.get_key_state(KeyCode::KeyA) {
                ButtonState::Pressed | ButtonState::Held => {
                    move_vector -= renderer.camera.transform.right();
                }
                _ => {}
            }

            match self.input.get_key_state(KeyCode::KeyD) {
                ButtonState::Pressed | ButtonState::Held => {
                    move_vector += renderer.camera.transform.right();
                }
                _ => {}
            }

            match self.input.get_key_state(KeyCode::Space) {
                ButtonState::Pressed | ButtonState::Held => {
                    move_vector += renderer.camera.transform.up();
                }
                _ => {}
            }

            match self.input.get_key_state(KeyCode::ControlLeft) {
                ButtonState::Pressed | ButtonState::Held => {
                    move_vector -= renderer.camera.transform.up();
                }
                _ => {}
            }

            renderer.camera.transform.position +=
                move_vector.normalize_or_zero() * current_move_speed * delta_time.as_secs_f32();
        }
    }

    pub fn post_frame(&mut self) {
        self.input.end_frame();
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
                    // .with_decorations(false)
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
            let new_pos = PhysicalPosition::new(new_pos.x + PARENT_CHILD_GAP, new_pos.y);
            window.set_outer_position(new_pos);
        }

        self.vulkan_context = Some(VulkanContext::new(window));

        const DEFAULT_RENDER_RESOLUTION: [u32; 2] = [1280, 720];

        let mut renderer = Renderer::new(
            &self.vulkan_context.as_ref().unwrap(),
            DEFAULT_RENDER_RESOLUTION,
        );

        log::info!("Loading GLTF scene...");

        // Scene::import_gltf(
        //     &mut renderer,
        //     std::path::Path::new("./assets/sponza/Sponza.gltf"),
        //     &self.context.as_ref().unwrap(),
        //     Vec3::new(0.0, 0.0, 0.0),
        // )
        // .unwrap();

        // Scene::import_gltf(
        //     &mut renderer,
        //     std::path::Path::new("./assets/bistro/bistro.gltf"),
        //     &self.context.as_ref().unwrap(),
        //     Vec3::new(0.0, 0.0, 0.0),
        // )
        // .unwrap();

        // Scene::import_gltf(
        //     &mut renderer,
        //     std::path::Path::new("./assets/cornell/cornell.gltf"),
        //     &self.context.as_ref().unwrap(),
        //     Vec3::new(0.0, 0.0, 0.0),
        // )
        // .unwrap();

        Scene::import_gltf(
            &mut renderer,
            std::path::Path::new("./assets/lion_head_2k/lion_head_2k.gltf"),
            &self.vulkan_context.as_ref().unwrap(),
            Vec3::new(2.0, 0.0, 0.0),
        )
        .unwrap();

        Scene::import_gltf(
            &mut renderer,
            std::path::Path::new("./assets/boulder_01_2k/boulder_01_2k.gltf"),
            &self.vulkan_context.as_ref().unwrap(),
            Vec3::new(-2.0, 0.0, 0.0),
        )
        .unwrap();

        Scene::import_gltf(
            &mut renderer,
            std::path::Path::new("./assets/spheres/spheres.gltf"),
            &self.vulkan_context.as_ref().unwrap(),
            Vec3::ZERO,
        )
        .unwrap();

        renderer.camera.transform.position = [-1.5, 1.0, 5.0].into();

        log::info!("GLTF scene loaded");

        self.renderer = Some(renderer);
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        self.input.handle_device_event(&event, device_id);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) => {
                if let Some(context) = &mut self.vulkan_context {
                    if size.width <= 0 || size.height <= 0 {
                        return;
                    }

                    // Disabled for speed, may cause crashyness
                    // context.wait_for_previous_frame_end();

                    // Calls the renderer's resize handler inside
                    context
                        .handle_resize_recreate_swap(self.renderer.as_mut().unwrap(), size)
                        .unwrap();
                }
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                panic!("This is supposed to be handled in main.rs");
            }
            WindowEvent::MouseInput {
                device_id,
                state,
                button,
            } => {
                self.input.handle_mouse_event(device_id, state, button);
            }
            WindowEvent::MouseWheel {
                device_id,
                delta,
                phase,
            } => {
                // self.input.handle_mouse_wheel(delta);
            }
            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {
                self.input.handle_mouse_move(
                    self.vulkan_context.as_ref().unwrap(),
                    position.x as f32,
                    position.y as f32,
                );
            }
            WindowEvent::KeyboardInput {
                device_id, event, ..
            } => match event.physical_key {
                PhysicalKey::Code(key_code) => {
                    match key_code {
                        KeyCode::Escape => {
                            event_loop.exit();
                        }
                        KeyCode::F11 => {
                            if let Some(context) = &mut self.vulkan_context {
                                context.winit.set_fullscreen(None);
                            }
                        }
                        KeyCode::F12 => {
                            if let Some(context) = &mut self.vulkan_context {
                                context.winit.set_fullscreen(None);
                            }
                        }
                        _ => {}
                    }

                    self.input
                        .handle_key_event(device_id, event.state, key_code);
                }
                PhysicalKey::Unidentified(native_key_code) => {
                    log::warn!("Unidentified key code: {native_key_code:?}");
                }
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(context) = &mut self.vulkan_context {
            context.winit.request_redraw();
        }
    }
}

impl RenderApp {
    pub fn window_id(&self) -> Option<winit::window::WindowId> {
        self.vulkan_context.as_ref().map(|c| c.winit.id())
    }
}

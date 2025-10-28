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
    pub context: Option<VulkanContext>,
    pub renderer: Option<Renderer>,

    pub stats: RenderAppStats,

    _parent_window: Option<Arc<Window>>,
    _queue_recreate_swapchain: bool,

    // Input state for camera
    keys_pressed: [bool; 9], // W, S, A, D, Space, C, (unused), (unused), Shift

    // Mouse state
    mouse_position: Option<(f64, f64)>,
    mouse_prev_position: Option<(f64, f64)>,
    mouse_captured: bool,
    mouse_capture_start: Option<(f64, f64)>, // Position where right-click started
}

impl RenderApp {
    pub fn new() -> Self {
        Self {
            context: None,
            renderer: None,
            stats: RenderAppStats::new(),

            _parent_window: None,
            _queue_recreate_swapchain: false,
            keys_pressed: [false; 9],
            mouse_position: None,
            mouse_prev_position: None,
            mouse_captured: false,
            mouse_capture_start: None,
        }
    }

    /// Will not change existing window's parent. Use before `resumed` to set the parent window.
    pub fn set_parent_window(&mut self, window: Arc<Window>) {
        self._parent_window = Some(window.clone());
    }

    pub fn redraw(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.stats.update_frame_start();

        // Update camera based on pressed keys with smooth delta_time movement
        let mut delta_time = self.stats.last_second_delta_time.as_secs_f32();

        // Clamp delta time to prevent huge jumps on first frame or after stalls
        const MAX_DELTA_TIME: f32 = 0.05; // 50ms max per frame (20 FPS minimum)
        if delta_time > MAX_DELTA_TIME {
            delta_time = MAX_DELTA_TIME;
        }

        let mut movement_speed = if let Some(renderer) = &self.renderer {
            renderer.movement_speed
        } else {
            5.0
        };

        // Apply speed boost when shift is held
        if self.keys_pressed[8] {
            // Shift
            movement_speed *= 2.5; // 2.5x faster when sprinting
        }

        if let Some(renderer) = &mut self.renderer {
            let forward = renderer.camera.transform.forward();
            let right = renderer.camera.transform.right();
            let up = renderer.camera.transform.up();

            // Position movement with WASD keys
            if self.keys_pressed[0] {
                // W
                renderer.camera.transform.position += movement_speed * delta_time * forward;
            }
            if self.keys_pressed[1] {
                // S
                renderer.camera.transform.position -= movement_speed * delta_time * forward;
            }
            if self.keys_pressed[2] {
                // A
                renderer.camera.transform.position -= movement_speed * delta_time * right;
            }
            if self.keys_pressed[3] {
                // D
                renderer.camera.transform.position += movement_speed * delta_time * right;
            }
            if self.keys_pressed[4] {
                // Space
                renderer.camera.transform.position += movement_speed * delta_time * up;
            }
            if self.keys_pressed[5] {
                // C
                renderer.camera.transform.position -= movement_speed * delta_time * up;
            }

            const SENSITIVITY_MULTIPLIER: f32 = 0.0025;

            // Mouse-based rotation when right button is held
            if self.mouse_captured {
                if let (Some((curr_x, curr_y)), Some((prev_x, prev_y))) =
                    (self.mouse_position, self.mouse_prev_position)
                {
                    let delta_x = (curr_x - prev_x) as f32;
                    let delta_y = (curr_y - prev_y) as f32;

                    if delta_x != 0.0 || delta_y != 0.0 {
                        // Extract current euler angles to maintain FPS-style camera
                        let (mut yaw, mut pitch, _roll) = renderer
                            .camera
                            .transform
                            .rotation
                            .to_euler(glam::EulerRot::YXZ);

                        // Update yaw (horizontal) - no clamping needed
                        yaw -= delta_x * renderer.mouse_sensitivity * SENSITIVITY_MULTIPLIER;

                        // Update pitch (vertical) with clamping to prevent over-rotation
                        pitch -= delta_y * renderer.mouse_sensitivity * SENSITIVITY_MULTIPLIER;
                        const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.1; // ~89 degrees
                        pitch = pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);

                        // Reconstruct quaternion with roll locked to 0
                        renderer.camera.transform.rotation =
                            glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0);
                    }
                }
            }

            // Update previous mouse position for next frame
            self.mouse_prev_position = self.mouse_position;
        }

        let context = self.context.as_mut().unwrap();
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

        self.context = Some(VulkanContext::new(window));

        const DEFAULT_RENDER_RESOLUTION: [u32; 2] = [1280, 720];

        let mut renderer =
            Renderer::new(&self.context.as_ref().unwrap(), DEFAULT_RENDER_RESOLUTION);

        log::info!("Loading HDRI...");

        renderer.load_hdri(
            self.context.as_ref().unwrap(),
            std::path::Path::new("./assets/hdri/meadow_2_4k.exr"),
        )
        .expect("Failed to load default HDRI");

        log::info!("Loading GLTF scene...");

        // Scene::import_gltf(
        //     &mut renderer,
        //     std::path::Path::new("./assets/sponza/Sponza.gltf"),
        //     &self.context.as_ref().unwrap(),
        //     Vec3::new(0.0, 0.0, 0.0),
        //     2.0,
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
            std::path::Path::new("./assets/toy_car/ToyCar.gltf"),
            &self.context.as_ref().unwrap(),
            Vec3::new(0.0, 0.0, 0.0),
            100.0,
        )
        .unwrap();

        Scene::import_gltf(
            &mut renderer,
            std::path::Path::new("./assets/lion_head_2k/lion_head_2k.gltf"),
            &self.context.as_ref().unwrap(),
            Vec3::new(4.0, 0.0, 0.0),
            5.0,
        )
        .unwrap();

        Scene::import_gltf(
            &mut renderer,
            std::path::Path::new("./assets/boulder_01_2k/boulder_01_2k.gltf"),
            &self.context.as_ref().unwrap(),
            Vec3::new(-4.0, 0.0, 0.0),
            1.0,
        )
        .unwrap();

        // Scene::import_gltf(
        //     &mut renderer,
        //     std::path::Path::new("./assets/spheres/spheres.gltf"),
        //     &self.context.as_ref().unwrap(),
        //     Vec3::ZERO,
        // )
        // .unwrap();

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
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let is_pressed = event.state == winit::event::ElementState::Pressed;

                    match code {
                        KeyCode::Escape => {
                            event_loop.exit();
                        }
                        KeyCode::KeyW => {
                            self.keys_pressed[0] = is_pressed;
                        }
                        KeyCode::KeyS => {
                            self.keys_pressed[1] = is_pressed;
                        }
                        KeyCode::KeyA => {
                            self.keys_pressed[2] = is_pressed;
                        }
                        KeyCode::KeyD => {
                            self.keys_pressed[3] = is_pressed;
                        }
                        KeyCode::Space => {
                            self.keys_pressed[4] = is_pressed;
                        }
                        KeyCode::KeyC => {
                            self.keys_pressed[5] = is_pressed;
                        }
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                            self.keys_pressed[8] = is_pressed;
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                use winit::event::{ElementState, MouseButton};
                if button == MouseButton::Right {
                    if state == ElementState::Pressed {
                        // Right click pressed - capture starting position, hide cursor, lock mouse
                        self.mouse_captured = true;
                        self.mouse_capture_start = self.mouse_position;
                        
                        if let Some(context) = &self.context {
                            let _ = context.winit.set_cursor_visible(false);
                        }
                    } else {
                        // Right click released - show cursor and unlock
                        self.mouse_captured = false;
                        self.mouse_capture_start = None;
                        
                        if let Some(context) = &self.context {
                            let _ = context.winit.set_cursor_visible(true);
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_position = Some((position.x, position.y));
                
                // When mouse is captured, wrap it at screen edges
                if self.mouse_captured {
                    if let Some(context) = &self.context {
                        let window = &context.winit;
                        let size = window.inner_size();
                        let width = size.width as f64;
                        let height = size.height as f64;
                        
                        let mut new_x = position.x;
                        let mut new_y = position.y;
                        let margin = 5.0; // Wrap when near edge
                        
                        // Wrap horizontally
                        if new_x < margin {
                            new_x = width - margin - 1.0;
                        } else if new_x > width - margin {
                            new_x = margin + 1.0;
                        }
                        
                        // Wrap vertically
                        if new_y < margin {
                            new_y = height - margin - 1.0;
                        } else if new_y > height - margin {
                            new_y = margin + 1.0;
                        }
                        
                        // If we wrapped, set cursor position and update tracking
                        if new_x != position.x || new_y != position.y {
                            let _ = window.set_cursor_position(
                                winit::dpi::PhysicalPosition::new(new_x, new_y)
                            );
                            self.mouse_position = Some((new_x, new_y));
                            self.mouse_prev_position = Some((new_x, new_y));
                        }
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

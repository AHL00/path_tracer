use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use glam::Vec3;
use simple_logger::SimpleLogger;
use winit::event_loop::EventLoopBuilder;

mod config_window;
mod render_window;
use config_window::ConfigApp;
use render_window::RenderApp;

fn main() {
    SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .without_timestamps()
        .with_module_level("vulkano", log::LevelFilter::Warn)
        .with_module_level("wgpu", log::LevelFilter::Warn)
        .with_module_level("wgpu_core", log::LevelFilter::Warn)
        .with_module_level("naga", log::LevelFilter::Warn)
        .init()
        .unwrap();

    let mut app = MainApp::new();
    let event_loop = EventLoopBuilder::default().build().unwrap();

    event_loop.run_app(&mut app).unwrap();
}

pub struct MainApp {
    render_app: RenderApp,
    config_app: ConfigApp,
}

impl MainApp {
    pub fn new() -> Self {
        Self {
            render_app: RenderApp::new(),
            config_app: ConfigApp::new(),
        }
    }
}

pub const PARENT_CHILD_GAP: i32 = 10;

// Just do the wgpu example on EGUI
impl winit::application::ApplicationHandler for MainApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.config_app.resumed(event_loop);
        self.render_app.set_parent_window(
            self.config_app
                .window
                .clone()
                .expect("Parent window not found"),
        );
        self.render_app.resumed(event_loop);
        self.config_app
            .set_child_window(self.render_app.vulkan_context.as_ref().unwrap().winit.clone());
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let winit::event::WindowEvent::RedrawRequested = event {
            self.render_app.pre_render_update();
            // Stupid workaround cause of winit only sending redraw events to the focused window
            self.render_app.redraw(event_loop);
            self.config_app.redraw(event_loop, &mut self.render_app);
            
            self.render_app.post_frame();
            return;
        }

        if self.config_app.window_id() == Some(window_id) {
            self.config_app
                .window_event(event_loop, window_id.clone(), event.clone());
            return;
        }

        if self.render_app.window_id() == Some(window_id) {
            self.render_app
                .window_event(event_loop, window_id.clone(), event.clone());
            return;
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.render_app.about_to_wait(event_loop);
        self.config_app.about_to_wait(event_loop);
    }

    fn exiting(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.render_app.exiting(event_loop);
        self.config_app.exiting(event_loop);
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.render_app.suspended(event_loop);
        self.config_app.suspended(event_loop);
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        self.render_app
            .device_event(event_loop, device_id, event.clone());
        self.config_app.device_event(event_loop, device_id, event);
    }

    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        self.render_app.new_events(event_loop, cause);
        self.config_app.new_events(event_loop, cause);
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: ()) {
        self.render_app.user_event(event_loop, event);
        self.config_app.user_event(event_loop, event);
    }

    fn memory_warning(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.render_app.memory_warning(event_loop);
        self.config_app.memory_warning(event_loop);
    }
}

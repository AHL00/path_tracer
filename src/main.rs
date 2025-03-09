use path_tracer::{graphics::VulkanContext, renderer::Renderer};
use simple_logger::SimpleLogger;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage}, swapchain::{acquire_next_image, SwapchainPresentInfo}, sync::GpuFuture, Validated, VulkanError
};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, Size},
    event::WindowEvent,
    event_loop::EventLoop,
    window::WindowAttributes,
};

fn main() {
    SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .without_timestamps()
        .with_module_level("vulkano", log::LevelFilter::Warn)
        .init()
        .unwrap();

    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let mut app = App::new();

    event_loop.run_app(&mut app).unwrap();
}

pub struct App {
    pub context: Option<VulkanContext>,
    pub renderer: Option<Renderer>,
    _queue_recreate_swapchain: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            context: None,
            renderer: None,
            _queue_recreate_swapchain: false,
        }
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
        self.renderer = Some(Renderer::new(&self.context.as_ref().unwrap()));
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

                renderer.record_commands(image_index, &mut builder);

                let command_buffer = builder.build().unwrap();
                let future = context
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(context.queue.clone(), command_buffer)
                    .unwrap()
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
                        context.previous_frame_end = Some(vulkano::sync::now(context.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        context.previous_frame_end = Some(vulkano::sync::now(context.device.clone()).boxed());
                    }
                }
            }

            _ => {}
        }
    }
}

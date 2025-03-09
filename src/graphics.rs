use std::{error::Error, sync::Arc};

use vulkano::{
    Validated, Version, VulkanError, VulkanLibrary,
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::{
        self,
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    image::{Image, ImageFormatInfo, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::GpuFuture,
};
use winit::dpi::{LogicalSize, PhysicalSize, Size};

use crate::renderer::{self, Renderer};

/// Context containing Vulkan resources
pub struct VulkanContext {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,

    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    pub winit: Arc<winit::window::Window>,
    pub surface: Arc<Surface>,
    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<Image>>,

    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl VulkanContext {
    /// Create a new Vulkan context
    pub fn new(winit: winit::window::Window) -> Self {
        let winit = Arc::new(winit);

        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

        let required_extensions =
            Surface::required_extensions(&winit).expect("failed to get required extensions");

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create Vulkan instance");

        log::info!("Vulkan instance created");

        // Create a Vulkan surface
        let surface = Surface::from_window(instance.clone(), winit.clone()).unwrap();
        let window_size = winit.inner_size();

        log::info!("Vulkan surface created");

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_ray_tracing_pipeline: true,
            khr_ray_tracing_maintenance1: true,
            khr_synchronization2: true,
            khr_deferred_host_operations: true,
            khr_acceleration_structure: true,
            ..DeviceExtensions::empty()
        };

        let device_features = DeviceFeatures {
            acceleration_structure: true,
            ray_tracing_pipeline: true,
            buffer_device_address: true,
            synchronization2: true,
            ..Default::default()
        };

        let physical_device = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.api_version() >= Version::V1_3)
            .filter(|physical_device| {
                // Check if the device supports the required extensions
                physical_device
                    .supported_extensions()
                    .contains(&device_extensions)
                    && physical_device
                        .supported_features()
                        .contains(&device_features)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    // Find the first first queue family that is suitable.
                    // If none is found, `None` is returned to `filter_map`,
                    // which disqualifies this physical device.
                    .position(|(i, q)| {
                        q.queue_flags
                            .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,

                // Note that there exists `PhysicalDeviceType::Other`, however,
                // `PhysicalDeviceType` is a non-exhaustive enum. Thus, one should
                // match wildcard `_` to catch all unknown device types.
                _ => 4,
            });

        if let None = physical_device {
            log::error!("No Vulkan devices found");
            log::error!("Requires Vulkan 1.3 or higher");
            log::error!("Required features: {:?}", device_features);
            log::error!("Required extensions: {:?}", device_extensions);
        }

        let physical_device = physical_device.unwrap().0;

        log::info!(
            "Vulkan device: {:?}",
            physical_device.properties().device_name
        );

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
            })
            .expect("couldn't find a graphical queue family")
            as u32;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                ..Default::default()
            },
        )
        .expect("failed to create device");

        log::info!("Vulkan device and queue created");

        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let surface_capabilities = physical_device
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let (image_format, image_color_space) = physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()
                .into_iter()
                .find(|(format, _)| {
                    physical_device
                        .image_format_properties(ImageFormatInfo {
                            format: *format,
                            usage: ImageUsage::STORAGE,
                            ..Default::default()
                        })
                        .unwrap()
                        .is_some()
                })
                .unwrap();

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_color_space,
                    image_extent: window_size.into(),
                    // We will directly write to the swapchain images from the ray
                    // tracing shader. This requires the images to support storage usage.
                    image_usage: ImageUsage::STORAGE,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                ..Default::default()
            },
        ));

        let previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());

        Self {
            instance,
            device,
            queue,
            command_buffer_allocator,
            memory_allocator,
            descriptor_set_allocator,

            winit,
            surface,
            swapchain,
            images,

            previous_frame_end,
        }
    }

    pub fn wait_for_previous_frame_end(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();
    }

    pub fn recreate_swapchain(
        &mut self,
        renderer: &mut Renderer,
        size: PhysicalSize<u32>,
    ) -> Result<(), Validated<VulkanError>> {
        let (new_swapchain, new_images) = self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: size.into(),
            ..self.swapchain.create_info()
        })?;

        self.swapchain = new_swapchain;
        renderer.resize(self, &new_images);

        Ok(())
    }
}

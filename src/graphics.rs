use std::{error::Error, sync::Arc};

use vulkano::{
    DeviceSize, Validated, Version, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract,
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    },
    descriptor_set::{
        self,
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    format::Format,
    image::{
        Image, ImageCreateFlags, ImageCreateInfo, ImageFormatInfo, ImageType, ImageUsage,
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
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
    pub descriptor_set_allocator_update_after_bind: Arc<StandardDescriptorSetAllocator>,

    pub winit: Arc<winit::window::Window>,
    pub surface: Arc<Surface>,
    pub swapchain: Arc<Swapchain>,
    pub swapchain_images: Vec<Arc<Image>>,

    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl VulkanContext {
    /// Create a new Vulkan context
    pub fn new(winit: winit::window::Window) -> Self {
        let winit = Arc::new(winit);

        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

        let mut required_extensions =
            Surface::required_extensions(&winit).expect("failed to get required extensions");

        required_extensions.ext_debug_utils = true;

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

            // Bindless textures
            ext_descriptor_indexing: true,
            khr_maintenance3: true,
            ..DeviceExtensions::empty()
        };

        let device_features = DeviceFeatures {
            acceleration_structure: true,
            ray_tracing_pipeline: true,
            buffer_device_address: true,
            synchronization2: true,

            // Bindless
            descriptor_binding_partially_bound: true,
            descriptor_binding_variable_descriptor_count: true,
            descriptor_binding_sampled_image_update_after_bind: true,
            shader_sampled_image_array_dynamic_indexing: true,
            shader_sampled_image_array_non_uniform_indexing: true,
            runtime_descriptor_array: true,

            ..DeviceFeatures::empty()
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
                            usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
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
                    min_image_count: surface_capabilities.min_image_count,
                    image_format,
                    image_color_space,
                    image_extent: window_size.into(),
                    // We will blit to the swapchain images, so we need to use a
                    // transfer dst usage flag.
                    image_usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    // present_mode: vulkano::swapchain::PresentMode::FifoRelaxed,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        log::debug!(
            "Swapchain created: {} images, format: {:?}",
            images.len(),
            swapchain.image_format()
        );

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

        let descriptor_set_allocator_update_after_bind =
            Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                StandardDescriptorSetAllocatorCreateInfo {
                    update_after_bind: true,
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
            descriptor_set_allocator_update_after_bind,

            winit,
            surface,
            swapchain,
            swapchain_images: images,

            previous_frame_end,
        }
    }

    pub fn wait_for_previous_frame_end(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();
    }

    pub fn handle_resize_recreate_swap(
        &mut self,
        renderer: &mut Renderer,
        size: PhysicalSize<u32>,
    ) -> Result<(), Validated<VulkanError>> {
        // If it doesn't match the render resolution aspect, we don't want to allow the swapchain to be resized.
        let render_aspect =
            renderer.render_resolution()[0] as f32 / renderer.render_resolution()[1] as f32;

        let window_aspect = size.width as f32 / size.height as f32;
        if (window_aspect - render_aspect).abs() > 0.01 {
            let size_delta = [
                size.width as i32 - self.swapchain.image_extent()[0] as i32,
                size.height as i32 - self.swapchain.image_extent()[1] as i32,
            ];

            let current_width = self.swapchain.image_extent()[0] as f32;
            let current_height = self.swapchain.image_extent()[1] as f32;

            let mut new_width = size.width as f32;
            let mut new_height = size.height as f32;

            let width_based_height = new_width / render_aspect;
            let height_based_width = new_height * render_aspect;

            let width_change = (new_width - current_width).abs();
            let height_change = (new_height - current_height).abs();

            if width_change > height_change {
                new_height = width_based_height;
            } else {
                new_width = height_based_width;
            }

            // If it doesn't match, request a resize that will match the aspect ratio
            if let None = self.winit.request_inner_size(LogicalSize::new(
                new_width.round() as u32,
                new_height.round() as u32,
            )) {
                // log::error!("Failed to resize window to match aspect ratio");
            }



            return Ok(());
        }

        let (new_swapchain, new_images) = self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: size.into(),
            image_usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
            ..self.swapchain.create_info()
        })?;

        self.swapchain = new_swapchain;
        self.swapchain_images = new_images.clone();

        Ok(())
    }
}

#[derive(Debug, Clone)]
/// Texture struct containing image, image view and sampler
pub struct Texture {
    image: Arc<Image>,
    image_view: Arc<ImageView>,
    sampler: Arc<Sampler>,

    bindless_indice: u32,
}

impl Texture {
    pub fn bindless_indice(&self) -> u32 {
        self.bindless_indice
    }

    pub fn image(&self) -> Arc<Image> {
        self.image.clone()
    }

    pub fn image_view(&self) -> Arc<ImageView> {
        self.image_view.clone()
    }

    pub fn sampler(&self) -> Arc<Sampler> {
        self.sampler.clone()
    }

    /// The image ident must be unique to each image source. It is used to
    /// look up whether the image is already loaded.
    pub fn from_gltf(
        image: gltf::image::Data,
        image_ident: String,
        sampler: gltf::texture::Sampler,
        renderer: &mut Renderer,
        context: &VulkanContext,
    ) -> Self {
        let format = Self::map_gltf_format_to_vulkan(image.format);
        let extent = [image.width, image.height, 1];
        let pixels = Self::map_gltf_image_data_to_vulkan(image.format, image.pixels);
        let mb_size = pixels.len() as f32 / 1024.0 / 1024.0;

        // Try looking up the image in the renderer
        if let Some(texture) = renderer.lookup_texture(&image_ident) {
            return texture;
        }

        let staging_buffer = Buffer::from_iter(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            pixels.into_iter(),
        )
        .unwrap();

        let image = Image::new(
            context.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: format,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                extent: extent.into(),

                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        // Copy the staging buffer to the image
        let mut uploader = AutoCommandBufferBuilder::primary(
            context.command_buffer_allocator.clone(),
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        uploader
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging_buffer,
                image.clone(),
            ))
            .unwrap();

        let image_view = ImageView::new_default(image.clone()).unwrap();

        let mag = sampler.mag_filter();
        let min = sampler.min_filter();
        let address_mode_u = sampler.wrap_s();
        let address_mode_v = sampler.wrap_t();

        // TODO

        let sampler = Sampler::new(
            context.device.clone(),
            SamplerCreateInfo {
                ..Default::default()
            },
        )
        .unwrap();

        let future = uploader
            .build()
            .unwrap()
            .execute(context.queue.clone())
            .unwrap();

        // TODO: Wait to finish?
        future.flush().unwrap();

        let texture_index = renderer.add_texture(image_view.clone(), sampler.clone(), context);

        let texture = Self {
            image,
            image_view,
            sampler,

            bindless_indice: texture_index,
        };

        renderer.add_texture_to_lookup_map(image_ident.clone(), texture.clone());
        log::debug!(
            "Texture loaded: {}, {}x{}, {}mb",
            image_ident,
            extent[0],
            extent[1],
            mb_size
        );

        texture
    }

    fn map_gltf_format_to_vulkan(format: gltf::image::Format) -> Format {
        match format {
            gltf::image::Format::R8G8B8A8 => Format::R8G8B8A8_SRGB,
            gltf::image::Format::R8G8B8 => Format::R8G8B8A8_SRGB, // NOTE: Converted to RGBA8
            gltf::image::Format::R16G16B16A16 => Format::R16G16B16A16_SFLOAT,
            gltf::image::Format::R32G32B32A32FLOAT => Format::R32G32B32A32_SFLOAT,
            gltf::image::Format::R32G32B32FLOAT => Format::R32G32B32_SFLOAT,
            gltf::image::Format::R8 => Format::R8_UNORM,
            gltf::image::Format::R16 => Format::R16_UNORM,
            gltf::image::Format::R8G8 => Format::R8G8_UNORM,
            gltf::image::Format::R16G16 => Format::R16G16_UNORM,
            gltf::image::Format::R16G16B16 => Format::R16G16B16_UNORM,
        }
    }

    fn rgb8_srg_to_rgba8_srgb(pixels: Vec<u8>) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(pixels.len() * 4 / 3);
        for chunk in pixels.chunks(3) {
            rgba.push(chunk[0]);
            rgba.push(chunk[1]);
            rgba.push(chunk[2]);
            rgba.push(255);
        }
        rgba
    }

    fn map_gltf_image_data_to_vulkan(format: gltf::image::Format, pixels: Vec<u8>) -> Vec<u8> {
        match format {
            gltf::image::Format::R8G8B8A8 => pixels,
            gltf::image::Format::R8G8B8 => Self::rgb8_srg_to_rgba8_srgb(pixels),
            gltf::image::Format::R16G16B16A16 => pixels,
            gltf::image::Format::R32G32B32A32FLOAT => pixels,
            gltf::image::Format::R32G32B32FLOAT => pixels,
            gltf::image::Format::R8 => pixels,
            gltf::image::Format::R16 => pixels,
            gltf::image::Format::R8G8 => pixels,
            gltf::image::Format::R16G16 => pixels,
            gltf::image::Format::R16G16B16 => pixels,
        }
    }

    fn gltf_format_pixel_byte_size(format: gltf::image::Format) -> u32 {
        match format {
            gltf::image::Format::R8G8B8A8 => 4,
            gltf::image::Format::R8G8B8 => 3,
            gltf::image::Format::R16G16B16A16 => 8,
            gltf::image::Format::R32G32B32A32FLOAT => 16,
            gltf::image::Format::R32G32B32FLOAT => 12,
            gltf::image::Format::R8 => 1,
            gltf::image::Format::R16 => 2,
            gltf::image::Format::R8G8 => 2,
            gltf::image::Format::R16G16 => 4,
            gltf::image::Format::R16G16B16 => 6,
        }
    }
}

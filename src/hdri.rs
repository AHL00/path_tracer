use crate::graphics::VulkanContext;
use image::ImageReader;
use std::path::Path;
use std::sync::Arc;
use vulkano::{
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageUsage,
        sampler::{Sampler, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};

/// HDRI texture management
pub struct HdriTexture {
    pub image: Arc<Image>,
    pub view: Arc<ImageView>,
    pub sampler: Arc<Sampler>,
    pub width: u32,
    pub height: u32,
    pub rotation: f32,  // in degrees (0-360)
    pub intensity: f32, // multiplier for brightness
}

impl HdriTexture {
    /// Load an HDRI from file (.hdr or .exr) - simplified approach
    pub fn load(context: &VulkanContext, path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // Load image using image crate (supports .hdr, .exr, and other formats)
        let img = ImageReader::open(path)?.decode()?.to_rgba32f();

        let (width, height) = img.dimensions();

        // Convert to flat f32 buffer
        let raw_pixels: Vec<f32> = img
            .pixels()
            .flat_map(|p| vec![p[0], p[1], p[2], p[3]].into_iter())
            .collect();

        // Create Vulkan image
        let image = Image::new(
            context.memory_allocator.clone(),
            ImageCreateInfo {
                format: Format::R32G32B32A32_SFLOAT,
                extent: [width, height, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        // Create staging buffer with the pixel data
        let staging_buffer = vulkano::buffer::Buffer::new_slice(
            context.memory_allocator.clone(),
            vulkano::buffer::BufferCreateInfo {
                usage: vulkano::buffer::BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            raw_pixels.len() as u64,
        )?;

        // Copy pixel data to staging buffer
        {
            let mut write = staging_buffer.write()?;
            write.copy_from_slice(&raw_pixels);
        }

        // Copy from staging buffer to GPU image
        let mut cmd_builder = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
            context.command_buffer_allocator.clone(),
            context.queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        cmd_builder.copy_buffer_to_image(
            vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
                staging_buffer.clone(),
                image.clone(),
            ),
        )?;

        let cmd = cmd_builder.build()?;
        vulkano::sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        // Create image view
        let view = ImageView::new_default(image.clone())?;

        // Create sampler with linear filtering
        let sampler = Sampler::new(
            context.device.clone(),
            SamplerCreateInfo::simple_repeat_linear(),
        )?;

        log::info!("Loaded HDRI: {}x{} from {:?}", width, height, path);

        Ok(Self {
            image,
            view,
            sampler,
            width,
            height,
            rotation: 0.0,
            intensity: 1.0,
        })
    }
}

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{c_char, c_void};

pub type Instance = c_void;
pub type Identifier = u32;

pub mod enums {
    use super::*;
    #[repr(u32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum Result {
        SUCCESS = 0,
        FAILURE = 1,
        INVALID_ARGUMENT = 2,
        UNSUPPORTED = 3,
        NON_UNIQUE_IDENTIFIER = 4,
        MAX_NUM = 5,
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum ResourceType {
        IN_MV = 0,
        IN_NORMAL_ROUGHNESS = 1,
        IN_VIEWZ = 2,
        IN_DIFF_CONFIDENCE = 3,
        IN_SPEC_CONFIDENCE = 4,
        IN_DISOCCLUSION_THRESHOLD_MIX = 5,
        IN_BASECOLOR_METALNESS = 6,
        IN_DIFF_RADIANCE_HITDIST = 7,
        IN_SPEC_RADIANCE_HITDIST = 8,
        IN_DIFF_HITDIST = 9,
        IN_SPEC_HITDIST = 10,
        IN_DIFF_DIRECTION_HITDIST = 11,
        IN_DIFF_SH0 = 12,
        IN_DIFF_SH1 = 13,
        IN_SPEC_SH0 = 14,
        IN_SPEC_SH1 = 15,
        IN_PENUMBRA = 16,
        IN_TRANSLUCENCY = 17,
        IN_SIGNAL = 18,
        OUT_DIFF_RADIANCE_HITDIST = 19,
        OUT_SPEC_RADIANCE_HITDIST = 20,
        OUT_DIFF_SH0 = 21,
        OUT_DIFF_SH1 = 22,
        OUT_SPEC_SH0 = 23,
        OUT_SPEC_SH1 = 24,
        OUT_DIFF_HITDIST = 25,
        OUT_SPEC_HITDIST = 26,
        OUT_DIFF_DIRECTION_HITDIST = 27,
        OUT_SHADOW_TRANSLUCENCY = 28,
        OUT_SIGNAL = 29,
        OUT_VALIDATION = 30,
        TRANSIENT_POOL = 31,
        PERMANENT_POOL = 32,
        MAX_NUM = 33,
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum Denoiser {
        REBLUR_DIFFUSE = 0,
        REBLUR_DIFFUSE_OCCLUSION = 1,
        REBLUR_DIFFUSE_SH = 2,
        REBLUR_SPECULAR = 3,
        REBLUR_SPECULAR_OCCLUSION = 4,
        REBLUR_SPECULAR_SH = 5,
        REBLUR_DIFFUSE_SPECULAR = 6,
        REBLUR_DIFFUSE_SPECULAR_OCCLUSION = 7,
        REBLUR_DIFFUSE_SPECULAR_SH = 8,
        REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION = 9,
        RELAX_DIFFUSE = 10,
        RELAX_DIFFUSE_SH = 11,
        RELAX_SPECULAR = 12,
        RELAX_SPECULAR_SH = 13,
        RELAX_DIFFUSE_SPECULAR = 14,
        RELAX_DIFFUSE_SPECULAR_SH = 15,
        SIGMA_SHADOW = 16,
        SIGMA_SHADOW_TRANSLUCENCY = 17,
        REFERENCE = 18,
        MAX_NUM = 19,
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum Format {
        R8_UNORM = 0,
        R8_SNORM = 1,
        R8_UINT = 2,
        R8_SINT = 3,
        RG8_UNORM = 4,
        RG8_SNORM = 5,
        RG8_UINT = 6,
        RG8_SINT = 7,
        RGBA8_UNORM = 8,
        RGBA8_SNORM = 9,
        RGBA8_UINT = 10,
        RGBA8_SINT = 11,
        RGBA8_SRGB = 12,
        R16_UNORM = 13,
        R16_SNORM = 14,
        R16_UINT = 15,
        R16_SINT = 16,
        R16_SFLOAT = 17,
        RG16_UNORM = 18,
        RG16_SNORM = 19,
        RG16_UINT = 20,
        RG16_SINT = 21,
        RG16_SFLOAT = 22,
        RGBA16_UNORM = 23,
        RGBA16_SNORM = 24,
        RGBA16_UINT = 25,
        RGBA16_SINT = 26,
        RGBA16_SFLOAT = 27,
        R32_UINT = 28,
        R32_SINT = 29,
        R32_SFLOAT = 30,
        RG32_UINT = 31,
        RG32_SINT = 32,
        RG32_SFLOAT = 33,
        RGB32_UINT = 34,
        RGB32_SINT = 35,
        RGB32_SFLOAT = 36,
        RGBA32_UINT = 37,
        RGBA32_SINT = 38,
        RGBA32_SFLOAT = 39,
        R10_G10_B10_A2_UNORM = 40,
        R10_G10_B10_A2_UINT = 41,
        R11_G11_B10_UFLOAT = 42,
        R9_G9_B9_E5_UFLOAT = 43,
        MAX_NUM = 44,
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum DescriptorType {
        TEXTURE = 0,
        STORAGE_TEXTURE = 1,
        MAX_NUM = 2,
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum Sampler {
        NEAREST_CLAMP = 0,
        LINEAR_CLAMP = 1,
        MAX_NUM = 2,
    }

    #[repr(u8)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum NormalEncoding {
        RGBA8_UNORM = 0,
        RGBA8_SNORM = 1,
        R10_G10_B10_A2_UNORM = 2,
        RGBA16_UNORM = 3,
        RGBA16_SNORM = 4,
        MAX_NUM = 5,
    }

    #[repr(u8)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum RoughnessEncoding {
        SQ_LINEAR = 0,
        LINEAR = 1,
        SQRT_LINEAR = 2,
        MAX_NUM = 3,
    }

    #[repr(u8)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum CheckerboardMode {
        OFF = 0,
        BLACK = 1,
        WHITE = 2,
        MAX_NUM = 3,
    }

    #[repr(u8)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum AccumulationMode {
        CONTINUE = 0,
        RESTART = 1,
        CLEAR_AND_RESTART = 2,
        MAX_NUM = 3,
    }

    #[repr(u8)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum HitDistanceReconstructionMode {
        OFF = 0,
        AREA_3X3 = 1,
        AREA_5X5 = 2,
        MAX_NUM = 3,
    }
}

pub mod structs {
    use super::*;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct AllocationCallbacks {
        pub Allocate: Option<
            unsafe extern "C" fn(
                userArg: *mut c_void,
                size: usize,
                alignment: usize,
            ) -> *mut c_void,
        >,
        pub Reallocate: Option<
            unsafe extern "C" fn(
                userArg: *mut c_void,
                memory: *mut c_void,
                size: usize,
                alignment: usize,
            ) -> *mut c_void,
        >,
        pub Free: Option<unsafe extern "C" fn(userArg: *mut c_void, memory: *mut c_void)>,
        pub userArg: *mut c_void,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct SPIRVBindingOffsets {
        pub samplerOffset: u32,
        pub textureOffset: u32,
        pub constantBufferOffset: u32,
        pub storageTextureAndBufferOffset: u32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct LibraryDesc {
        pub spirvBindingOffsets: SPIRVBindingOffsets,
        pub supportedDenoisers: *const Denoiser,
        pub supportedDenoisersNum: u32,
        pub versionMajor: u8,
        pub versionMinor: u8,
        pub versionBuild: u8,
        pub normalEncoding: NormalEncoding,
        pub roughnessEncoding: RoughnessEncoding,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct DenoiserDesc {
        pub identifier: Identifier,
        pub denoiser: Denoiser,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct InstanceCreationDesc {
        pub allocationCallbacks: AllocationCallbacks,
        pub denoisers: *const DenoiserDesc,
        pub denoisersNum: u32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct TextureDesc {
        pub format: Format,
        pub downsampleFactor: u16,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct ResourceDesc {
        pub descriptorType: DescriptorType,
        pub type_: ResourceType,
        pub indexInPool: u16,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct ResourceRangeDesc {
        pub descriptorType: DescriptorType,
        pub baseRegisterIndex: u32,
        pub descriptorsNum: u32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct ComputeShaderDesc {
        pub bytecode: *const c_void,
        pub size: u64,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct PipelineDesc {
        pub computeShaderDXBC: ComputeShaderDesc,
        pub computeShaderDXIL: ComputeShaderDesc,
        pub computeShaderSPIRV: ComputeShaderDesc,
        pub shaderFileName: *const c_char,
        pub shaderEntryPointName: *const c_char,
        pub resourceRanges: *const ResourceRangeDesc,
        pub resourceRangesNum: u32,
        pub hasConstantData: bool,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct DescriptorPoolDesc {
        pub setsMaxNum: u32,
        pub constantBuffersMaxNum: u32,
        pub samplersMaxNum: u32,
        pub texturesMaxNum: u32,
        pub storageTexturesMaxNum: u32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct InstanceDesc {
        pub constantBufferMaxDataSize: u32,
        pub constantBufferSpaceIndex: u32,
        pub constantBufferRegisterIndex: u32,
        pub samplers: *const Sampler,
        pub samplersNum: u32,
        pub samplersSpaceIndex: u32,
        pub samplersBaseRegisterIndex: u32,
        pub pipelines: *const PipelineDesc,
        pub pipelinesNum: u32,
        pub resourcesSpaceIndex: u32,
        pub permanentPool: *const TextureDesc,
        pub permanentPoolSize: u32,
        pub transientPool: *const TextureDesc,
        pub transientPoolSize: u32,
        pub descriptorPoolDesc: DescriptorPoolDesc,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct DispatchDesc {
        pub name: *const c_char,
        pub identifier: Identifier,
        pub resources: *const ResourceDesc,
        pub resourcesNum: u32,
        pub constantBufferData: *const u8,
        pub constantBufferDataSize: u32,
        pub constantBufferDataMatchesPreviousDispatch: bool,
        pub pipelineIndex: u16,
        pub gridWidth: u16,
        pub gridHeight: u16,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct HitDistanceParameters {
        pub A: f32,
        pub B: f32,
        pub C: f32,
        pub D: f32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct ReblurAntilagSettings {
        pub luminanceSigmaScale: f32,
        pub luminanceSensitivity: f32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct ReblurSettings {
        pub hitDistanceParameters: HitDistanceParameters,
        pub antilagSettings: ReblurAntilagSettings,
        pub maxAccumulatedFrameNum: u32,
        pub maxFastAccumulatedFrameNum: u32,
        pub maxStabilizedFrameNum: u32,
        pub historyFixFrameNum: u32,
        pub historyFixBasePixelStride: u32,
        pub diffusePrepassBlurRadius: f32,
        pub specularPrepassBlurRadius: f32,
        pub minHitDistanceWeight: f32,
        pub minBlurRadius: f32,
        pub maxBlurRadius: f32,
        pub lobeAngleFraction: f32,
        pub roughnessFraction: f32,
        pub responsiveAccumulationRoughnessThreshold: f32,
        pub planeDistanceSensitivity: f32,
        pub specularProbabilityThresholdsForMvModification: [f32; 2],
        pub fireflySuppressorMinRelativeScale: f32,
        pub checkerboardMode: CheckerboardMode,
        pub hitDistanceReconstructionMode: HitDistanceReconstructionMode,
        pub enableAntiFirefly: bool,
        pub enablePerformanceMode: bool,
        pub minMaterialForDiffuse: f32,
        pub minMaterialForSpecular: f32,
        pub usePrepassOnlyForSpecularMotionEstimation: bool,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct RelaxAntilagSettings {
        pub accelerationAmount: f32,
        pub spatialSigmaScale: f32,
        pub temporalSigmaScale: f32,
        pub resetAmount: f32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct RelaxSettings {
        pub antilagSettings: RelaxAntilagSettings,
        pub diffuseMaxAccumulatedFrameNum: u32,
        pub specularMaxAccumulatedFrameNum: u32,
        pub diffuseMaxFastAccumulatedFrameNum: u32,
        pub specularMaxFastAccumulatedFrameNum: u32,
        pub historyFixFrameNum: u32,
        pub historyFixBasePixelStride: u32,
        pub historyFixEdgeStoppingNormalPower: f32,
        pub spatialVarianceEstimationHistoryThreshold: u32,
        pub diffusePrepassBlurRadius: f32,
        pub specularPrepassBlurRadius: f32,
        pub minHitDistanceWeight: f32,
        pub diffusePhiLuminance: f32,
        pub specularPhiLuminance: f32,
        pub lobeAngleFraction: f32,
        pub roughnessFraction: f32,
        pub specularVarianceBoost: f32,
        pub specularLobeAngleSlack: f32,
        pub historyClampingColorBoxSigmaScale: f32,
        pub atrousIterationNum: u32,
        pub diffuseMinLuminanceWeight: f32,
        pub specularMinLuminanceWeight: f32,
        pub depthThreshold: f32,
        pub confidenceDrivenRelaxationMultiplier: f32,
        pub confidenceDrivenLuminanceEdgeStoppingRelaxation: f32,
        pub confidenceDrivenNormalEdgeStoppingRelaxation: f32,
        pub luminanceEdgeStoppingRelaxation: f32,
        pub normalEdgeStoppingRelaxation: f32,
        pub roughnessEdgeStoppingRelaxation: f32,
        pub checkerboardMode: CheckerboardMode,
        pub hitDistanceReconstructionMode: HitDistanceReconstructionMode,
        pub enableAntiFirefly: bool,
        pub enableRoughnessEdgeStopping: bool,
        pub minMaterialForDiffuse: f32,
        pub minMaterialForSpecular: f32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct SigmaSettings {
        pub lightDirection: [f32; 3],
        pub planeDistanceSensitivity: f32,
        pub maxStabilizedFrameNum: u32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct ReferenceSettings {
        pub maxAccumulatedFrameNum: u32,
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct CommonSettings {
        pub viewToClipMatrix: [f32; 16],
        pub viewToClipMatrixPrev: [f32; 16],
        pub worldToViewMatrix: [f32; 16],
        pub worldToViewMatrixPrev: [f32; 16],
        pub worldPrevToWorldMatrix: [f32; 16],
        pub motionVectorScale: [f32; 3],
        pub cameraJitter: [f32; 2],
        pub cameraJitterPrev: [f32; 2],
        pub resourceSize: [u16; 2],
        pub resourceSizePrev: [u16; 2],
        pub rectSize: [u16; 2],
        pub rectSizePrev: [u16; 2],
        pub viewZScale: f32,
        pub timeDeltaBetweenFrames: f32,
        pub denoisingRange: f32,
        pub disocclusionThreshold: f32,
        pub disocclusionThresholdAlternate: f32,
        pub cameraAttachedReflectionMaterialID: f32,
        pub strandMaterialID: f32,
        pub strandThickness: f32,
        pub splitScreen: f32,
        pub printfAt: [u16; 2],
        pub debug: f32,
        pub rectOrigin: [u32; 2],
        pub frameIndex: u32,
        pub accumulationMode: AccumulationMode,
        pub isMotionVectorInWorldSpace: bool,
        pub isHistoryConfidenceAvailable: bool,
        pub isDisocclusionThresholdMixAvailable: bool,
        pub isBaseColorMetalnessAvailable: bool,
        pub enableValidation: bool,
    }
}

use enums::*;
use structs::*;

unsafe extern "C" {
    pub fn CreateInstance(
        instanceCreationDesc: *const InstanceCreationDesc,
        instance: *mut *mut Instance,
    ) -> Result;

    pub fn DestroyInstance(instance: *mut Instance);

    pub fn GetLibraryDesc() -> LibraryDesc;

    pub fn GetInstanceDesc(instance: *const Instance) -> *const InstanceDesc;

    pub fn SetCommonSettings(
        instance: *mut Instance,
        commonSettings: *const CommonSettings,
    ) -> Result;

    pub fn SetDenoiserSettings(
        instance: *mut Instance,
        identifier: Identifier,
        denoiserSettings: *const c_void,
    ) -> Result;

    pub fn GetComputeDispatches(
        instance: *mut Instance,
        identifiers: *const Identifier,
        identifiersNum: u32,
        dispatchDescs: *mut *const DispatchDesc,
        dispatchDescsNum: *mut u32,
    ) -> Result;

    pub fn GetResourceTypeString(resourceType: ResourceType) -> *const c_char;

    pub fn GetDenoiserString(denoiser: Denoiser) -> *const c_char;
}

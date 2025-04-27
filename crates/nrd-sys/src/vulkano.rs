#![allow(unsafe_op_in_unsafe_fn)]

use crate::allocation::NrdAllocator;
use crate::ffi::structs::InstanceCreationDesc;
use crate::{NrdDenoiser, NrdResult};

pub struct VulkanoNrdInstance {
    instance: *mut crate::ffi::Instance,
    allocator: NrdAllocator,
}

impl VulkanoNrdInstance {
    pub fn new(denoisers: &[NrdDenoiser]) -> Result<Self, NrdResult> {
        let allocator = NrdAllocator::new();

        let mut instance = std::ptr::null_mut();
        let creation_desc = InstanceCreationDesc {
            allocationCallbacks: allocator.get_callbacks(),
            denoisers: denoisers.as_ptr() as *const _,
            denoisersNum: denoisers.len() as u32,
        };

        let result = unsafe { crate::ffi::CreateInstance(&creation_desc, &mut instance) };

        match result {
            NrdResult::SUCCESS => (),
            _ => return Err(result),
        }

        Ok(Self {
            instance,
            allocator,
        })
    }
}

impl Drop for VulkanoNrdInstance {
    fn drop(&mut self) {
        unsafe {
            crate::ffi::DestroyInstance(self.instance);
        }
    }
}


#![allow(unsafe_op_in_unsafe_fn)]
use std::{
    alloc::Layout,
    collections::HashMap,
    ffi::c_void,
    sync::{Arc, Mutex},
};

use crate::ffi::structs::AllocationCallbacks;

/// Tracks allocations made by NRD so we can properly free them
#[derive(Default)]
pub struct AllocationTracker {
    allocations: HashMap<*mut c_void, Layout>,
}

/// Vulkano-compatible allocator for NRD
pub struct NrdAllocator {
    tracker: Arc<Mutex<AllocationTracker>>,
}

impl NrdAllocator {
    /// Create a new NRD allocator
    pub fn new() -> Self {
        Self {
            tracker: Arc::new(Mutex::new(AllocationTracker::default())),
        }
    }

    /// Get NRD allocation callbacks for this allocator
    pub fn get_callbacks(&self) -> AllocationCallbacks {
        let tracker_ptr = Arc::as_ptr(&self.tracker) as *mut c_void;

        AllocationCallbacks {
            Allocate: Some(Self::allocate),
            Reallocate: Some(Self::reallocate),
            Free: Some(Self::free),
            userArg: tracker_ptr as *mut _,
        }
    }

    /// Allocation callback for NRD
    unsafe extern "C" fn allocate(
        user_arg: *mut c_void,
        size: usize,
        alignment: usize,
    ) -> *mut c_void {
        if size == 0 {
            return std::ptr::null_mut();
        }

        // Get our tracker from the user argument pointer
        let tracker = &*(user_arg as *const Mutex<AllocationTracker>);

        // Create a layout with the requested size and alignment
        let layout = match Layout::from_size_align(size, alignment) {
            Ok(layout) => layout,
            Err(_) => return std::ptr::null_mut(),
        };

        // Allocate memory
        let ptr = std::alloc::alloc(layout);
        if !ptr.is_null() {
            // Track the allocation
            let mut tracker = tracker.lock().unwrap();
            tracker.allocations.insert(ptr as *mut c_void, layout);
        }

        ptr as *mut c_void
    }

    /// Reallocation callback for NRD
    unsafe extern "C" fn reallocate(
        user_arg: *mut c_void,
        old_ptr: *mut c_void,
        size: usize,
        alignment: usize,
    ) -> *mut c_void {
        // Get our tracker
        let tracker = &*(user_arg as *const Mutex<AllocationTracker>);

        // Handle null pointer as a new allocation
        if old_ptr.is_null() {
            return Self::allocate(user_arg, size, alignment);
        }

        // Handle zero size as a free operation
        if size == 0 {
            Self::free(user_arg, old_ptr);
            return std::ptr::null_mut();
        }

        // Create new layout
        let new_layout = match Layout::from_size_align(size, alignment) {
            Ok(layout) => layout,
            Err(_) => return std::ptr::null_mut(),
        };

        // Get the old layout
        let mut tracker_lock = tracker.lock().unwrap();
        let old_layout = match tracker_lock.allocations.get(&old_ptr) {
            Some(layout) => *layout,
            None => return std::ptr::null_mut(), // Invalid pointer
        };

        // Allocate new memory
        let new_ptr = std::alloc::alloc(new_layout);
        if new_ptr.is_null() {
            return std::ptr::null_mut();
        }

        // Copy old data to new location
        let copy_size = std::cmp::min(old_layout.size(), new_layout.size());
        std::ptr::copy_nonoverlapping(old_ptr as *const u8, new_ptr, copy_size);

        // Free old memory and update tracking
        std::alloc::dealloc(old_ptr as *mut u8, old_layout);
        tracker_lock.allocations.remove(&old_ptr);
        tracker_lock
            .allocations
            .insert(new_ptr as *mut c_void, new_layout);

        new_ptr as *mut c_void
    }

    /// Free callback for NRD
    unsafe extern "C" fn free(user_arg: *mut c_void, ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }

        // Get our tracker
        let tracker = &*(user_arg as *const Mutex<AllocationTracker>);
        let mut tracker_lock = tracker.lock().unwrap();

        // Find the layout for this allocation
        if let Some(layout) = tracker_lock.allocations.remove(&ptr) {
            // Free the memory
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
    }
}

impl Drop for NrdAllocator {
    fn drop(&mut self) {
        // Check if we have any leaks when dropping the allocator
        let tracker = self.tracker.lock().unwrap();
        if !tracker.allocations.is_empty() {
            eprintln!(
                "Warning: NRD allocator dropped with {} allocations still active",
                tracker.allocations.len()
            );
        }
    }
}

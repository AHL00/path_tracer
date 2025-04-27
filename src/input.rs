use std::collections::HashMap;

use winit::{
    event::{DeviceEvent, DeviceId, ElementState, MouseButton},
    keyboard::KeyCode,
};

use crate::graphics::VulkanContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonState {
    Pressed,
    Held,
    Released,
}

#[derive(Debug, Clone)]
pub struct Input {
    mouse_left: ButtonState,
    mouse_right: ButtonState,
    mouse_middle: ButtonState,
    mouse_x: f32,
    mouse_y: f32,
    mouse_dx: f32,
    mouse_dy: f32,

    raw_mouse_dx: f32,
    raw_mouse_dy: f32,

    pub mouse_wheel: f32,
    pub mouse_wheel_d: f32,

    key_states: HashMap<KeyCode, ButtonState>,
}

impl Input {
    pub fn new() -> Self {
        Self {
            mouse_left: ButtonState::Released,
            mouse_right: ButtonState::Released,
            mouse_middle: ButtonState::Released,
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_dx: 0.0,
            mouse_dy: 0.0,

            raw_mouse_dx: 0.0,
            raw_mouse_dy: 0.0,

            mouse_wheel: 0.0,
            mouse_wheel_d: 0.0,
            key_states: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.mouse_wheel_d = 0.0;
    }

    pub fn get_mouse_right(&mut self) -> ButtonState {
        if ButtonState::Pressed == self.mouse_right {
            self.mouse_right = ButtonState::Held;
            ButtonState::Pressed
        } else {
            self.mouse_right
        }
    }

    pub fn get_mouse_left(&mut self) -> ButtonState {
        if ButtonState::Pressed == self.mouse_left {
            self.mouse_left = ButtonState::Held;
            ButtonState::Pressed
        } else {
            self.mouse_left
        }
    }

    pub fn get_mouse_middle(&mut self) -> ButtonState {
        if ButtonState::Pressed == self.mouse_middle {
            self.mouse_middle = ButtonState::Held;
            ButtonState::Pressed
        } else {
            self.mouse_middle
        }
    }

    pub fn get_key_state(&mut self, key: KeyCode) -> ButtonState {
        if let Some(state) = self.key_states.get_mut(&key) {
            if ButtonState::Pressed == *state {
                *state = ButtonState::Held;
                ButtonState::Pressed
            } else {
                *state
            }
        } else {
            ButtonState::Released
        }
    }

    pub fn get_mouse_position(&self) -> (f32, f32) {
        (self.mouse_x, self.mouse_y)
    }

    pub fn get_mouse_delta(&self) -> (f32, f32) {
        (self.mouse_dx, self.mouse_dy)
    }
    
    pub fn get_raw_mouse_delta(&mut self) -> (f32, f32) {
        (self.raw_mouse_dx, self.raw_mouse_dy)
    }

    pub fn end_frame(&mut self) {
        self.raw_mouse_dx = 0.0;
        self.raw_mouse_dy = 0.0;
        self.mouse_dx = 0.0;
        self.mouse_wheel = 0.0;
        self.mouse_wheel_d = 0.0;
    }

    pub fn handle_device_event(&mut self, event: &DeviceEvent, device_id: DeviceId) {
        match event {
            winit::event::DeviceEvent::MouseMotion { delta } => {
                self.raw_mouse_dx = delta.0 as f32;
                self.raw_mouse_dy = delta.1 as f32;
            }
            _ => {}
        }
    }

    pub fn handle_key_event(
        &mut self,
        device_id: DeviceId,
        state: ElementState,
        key: KeyCode,
    ) {
        match state {
            ElementState::Pressed => {
                self.key_states.insert(key, ButtonState::Pressed);
            }
            ElementState::Released => {
                self.key_states.insert(key, ButtonState::Released);
            }
        }
    }

    pub fn handle_mouse_event(
        &mut self,
        device_id: DeviceId,
        state: ElementState,
        event: MouseButton,
    ) {
        match event {
            MouseButton::Left => match state {
                ElementState::Pressed => {
                    self.mouse_left = ButtonState::Pressed;
                }
                ElementState::Released => {
                    self.mouse_left = ButtonState::Released;
                }
            },
            MouseButton::Right => match state {
                ElementState::Pressed => {
                    self.mouse_right = ButtonState::Pressed;
                }
                ElementState::Released => {
                    self.mouse_right = ButtonState::Released;
                }
            },
            MouseButton::Middle => match state {
                ElementState::Pressed => {
                    self.mouse_middle = ButtonState::Pressed;
                }
                ElementState::Released => {
                    self.mouse_middle = ButtonState::Released;
                }
            },
            _ => {}
        }
    }

    pub fn handle_mouse_wheel(&mut self, delta: f32) {
        self.mouse_wheel = delta;
        self.mouse_wheel_d = delta;
    }

    pub fn handle_mouse_move(&mut self, context: &VulkanContext, x: f32, y: f32) {
        self.mouse_dx = x - self.mouse_x;
        self.mouse_dy = y - self.mouse_y;

        self.mouse_x = x;
        self.mouse_y = y;
    }
}

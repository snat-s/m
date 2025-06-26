pub const std = @import("std");
pub const infinity = std.math.inf(f32);
pub const pi = std.math.pi;

pub const Ray = @import("ray.zig").Ray;
pub const Vec3 = @import("vec3.zig").Vec3;
//@import("interval.zig");
pub const write_color = @import("color.zig").write_color;

pub fn degrees_to_radians(degrees: f32) f32 {
    return degrees * pi / 180.0;
}


pub fn random_number() f32 {
    const rand = std.crypto.random;
    return rand.float(f32); 
}

pub fn random_number_in_interval(min: f32, max: f32) f32 {
    return min + (max-min)*random_number();
}


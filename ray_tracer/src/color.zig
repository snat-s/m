const std = @import("std");
const Vec3 = @import("vec3.zig").Vec3;

pub fn linear_to_gamma(linear_component: f32) f32 {
    if (linear_component > 0) {
        return std.math.sqrt(linear_component);
    }
    return 0;
}

pub fn write_color(out: std.fs.File.Writer, pixel_color: Vec3) !void {
    // i mean i know we have implemented clamp but std already has it
    const r: f32 = std.math.clamp(linear_to_gamma(pixel_color.x),0.000,0.999);
    const g: f32 = std.math.clamp(linear_to_gamma(pixel_color.y),0.000,0.999);
    const b: f32 = std.math.clamp(linear_to_gamma(pixel_color.z),0.000,0.999);
    const rbyte: i16 = @intFromFloat(255.999*r);
    const gbyte: i16 = @intFromFloat(255.999*g);
    const bbyte: i16 = @intFromFloat(255.999*b);
    try out.print("{d} {d} {d}\n",.{rbyte, gbyte, bbyte});
}

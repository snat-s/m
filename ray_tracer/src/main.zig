const std = @import("std");
const Vec3 = @import("vec3.zig").Vec3;
const write_color = @import("color.zig").write_color;
const Ray = @import("ray.zig").Ray;
const infinity = @import("rtweekend.zig").infinity;
const Sphere = @import("sphere.zig").Sphere;
const Hittable = @import("hittable.zig").Hittable;
const HittableList = @import("hittable.zig").HittableList;
const HitRecord = @import("hittable.zig").HitRecord;
const Interval = @import("interval.zig").Interval;
const Camera = @import("camera.zig").Camera;
const Material = @import("material.zig").Material;
const pi = @import("rtweekend.zig").pi;


pub fn main() !void {
    
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();


    //const R: f32 = std.math.cos(pi/4.0);

    // materials
    //const material_left = Material{.lambertian = .{ .albedo =.{.x = 0, .y = 0.0, .z = 1.0}}};
    //const material_right = Material{.lambertian = .{ .albedo =.{.x = 1, .y = 0.0, .z = 0.0}}};
    
    const material_ground = Material{ .lambertian = .{ .albedo = .{ .x = 0.8, .y = 0.8, .z = 0.0 } } };
    const material_center = Material{ .lambertian = .{ .albedo = .{ .x = 0.1, .y = 0.2, .z = 0.5 } } };
    const material_left = Material{ .dielectric = .{ .refraction_index = 1.50 }};
    const material_bubble = Material{ .dielectric = .{ .refraction_index = 1.00 / 1.50 }};
    const material_right = Material{ .metal = .{ .albedo = .{ .x = 0.8, .y = 0.6, .z = 0.2 }, .fuzz = 0.8 } };

    // world
    var world = HittableList.init(allocator);
    defer world.deinit();

    //var sphere1 = try allocator.create(Sphere);
    //defer allocator.destroy(sphere1);
    //sphere1.* = Sphere.init(.{ .x = -R, .y = 0, .z = -1.0 }, R, &material_left);
    //try world.add(sphere1.hittable());

    //var sphere2 = try allocator.create(Sphere);
    //defer allocator.destroy(sphere2);
    //sphere2.* = Sphere.init(.{ .x = R, .y = 0, .z = -1.0 }, R, &material_right);
    //try world.add(sphere2.hittable());


    var sphere1 = try allocator.create(Sphere);
    defer allocator.destroy(sphere1);
    sphere1.* = Sphere.init(.{ .x = 0, .y = 0, .z = -1.2 }, 0.5, &material_center);
    try world.add(sphere1.hittable());
                                       
    var sphere2 = try allocator.create(Sphere);
    defer allocator.destroy(sphere2);
    sphere2.* = Sphere.init(.{ .x = -1.0, .y = 0, .z = -1.0 }, 0.5, &material_left);
    try world.add(sphere2.hittable());
                                    
    var sphere3 = try allocator.create(Sphere);
    defer allocator.destroy(sphere3);
    sphere3.* = Sphere.init(.{ .x = 1.0, .y = 0, .z = -1.0 }, 0.5, &material_right);
    try world.add(sphere3.hittable());

    var sphere4 = try allocator.create(Sphere);
    defer allocator.destroy(sphere4);
    sphere4.* = Sphere.init(.{ .x = 0, .y = -100.5, .z = -1 }, 100, &material_ground);
    try world.add(sphere4.hittable());

    var sphere5 = try allocator.create(Sphere);
    defer allocator.destroy(sphere5);
    sphere5.* = Sphere.init(.{ .x = -1.0, .y = 0.0, .z = -1 }, 0.4, &material_bubble);
    try world.add(sphere5.hittable());


    var camera: Camera = undefined;
    camera.aspect_ratio = 16.0 / 9.0;
    camera.image_width = 400;
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;
    camera.vfov = 20; 
    camera.lookfrom = Vec3.init(-2,2,1);
    camera.lookat   = Vec3.init(0,0,-1);
    camera.vup      = Vec3.init(0,1,0);


    // camera focus
    camera.defocus_angle = 10.0;
    camera.focus_dist = 3.4;

    try camera.render(world.hittable());
}


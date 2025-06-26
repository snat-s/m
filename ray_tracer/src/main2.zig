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
const random_number = @import("rtweekend.zig").random_number;
const random_number_in_interval = @import("rtweekend.zig").random_number_in_interval;


pub fn main() !void {

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const material_ground = Material{ .lambertian = .{ .albedo = .{ .x = 0.5, .y = 0.5, .z = 0.5 } } };

    // world
    var world = HittableList.init(allocator);
    defer world.deinit();

    var sphere_ground = try allocator.create(Sphere);
    defer allocator.destroy(sphere_ground);
    sphere_ground.* = Sphere.init(.{ .x = 0, .y = -1000, .z = 0.0 }, 1000, &material_ground);
    try world.add(sphere_ground.hittable());

    var a: i32 = -11;

    while(a < 11) : (a+=1) {
        var b: i32 = -11;
        while(b < 11) : (b += 1) {
            const choose_mat = random_number();
            const center = Vec3.init(@as(f32, @floatFromInt(a)) + 0.9 * random_number(), 0.2, @as(f32, @floatFromInt(b)) + 0.9 * random_number());

            if (Vec3.sum(center, Vec3.negate(Vec3.init(4.0, 0.2, 0.0))).length() > 0.9) {
                var sphere_material: Material = undefined;
                var mat_ptr: *Material = undefined;

                if (choose_mat < 0.8) {
                    // diffuse
                    const albedo = Vec3.multiply(Vec3.random(), Vec3.random());
                    sphere_material = Material{ .lambertian = .{ .albedo = albedo } };
                    mat_ptr = try allocator.create(Material);
                    defer allocator.destroy(mat_ptr);
                    mat_ptr.* = sphere_material;
                } else if (choose_mat < 0.95) {
                    // metal
                    const albedo = Vec3.param_random(0.5, 1);
                    const fuzz = random_number_in_interval(0, 0.5);
                    sphere_material = Material{ .metal = .{ .albedo = albedo, .fuzz = fuzz } };
                    mat_ptr = try allocator.create(Material);
                    defer allocator.destroy(mat_ptr);
                    mat_ptr.* = sphere_material;
                } else {
                    // glass
                    sphere_material = Material{ .dielectric = .{ .refraction_index = 1.5 } };
                    mat_ptr = try allocator.create(Material);
                    defer allocator.destroy(mat_ptr);
                    mat_ptr.* = sphere_material;
                }

                var new_sphere = try allocator.create(Sphere);
                defer allocator.destroy(new_sphere);
                new_sphere.* = Sphere.init(center, 0.2, mat_ptr);
                try world.add(new_sphere.hittable());
            }
        }
    }

    const material1 = Material{ .dielectric = .{ .refraction_index = 1.50 }};
    const material2 = Material{ .lambertian = .{ .albedo = .{ .x = 0.4, .y = 0.2, .z = 0.1 } } };
    const material3 = Material{ .metal = .{ .albedo = .{ .x = 0.7, .y = 0.6, .z = 0.5 }, .fuzz = 0.0 } };

    var sphere1 = try allocator.create(Sphere);
    defer allocator.destroy(sphere1);
    sphere1.* = Sphere.init(.{ .x = 0.0, .y = 1.0, .z = 0.0 }, 1.0, &material1);
    try world.add(sphere1.hittable());

    var sphere2 = try allocator.create(Sphere);
    defer allocator.destroy(sphere2);
    sphere2.* = Sphere.init(.{ .x = -4.0, .y = 1.0, .z = 0.0 }, 1.0, &material2);
    try world.add(sphere2.hittable());

    var sphere3 = try allocator.create(Sphere);
    defer allocator.destroy(sphere3);
    sphere3.* = Sphere.init(.{ .x = 4.0, .y = 1.0, .z = 0.0 }, 1.0, &material3);
    try world.add(sphere3.hittable());

    var camera: Camera = undefined;
    camera.aspect_ratio = 16.0 / 9.0;
    camera.image_width = 400;
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;
    camera.vfov = 20; 
    camera.lookfrom = Vec3.init(13,2,3);
    camera.lookat   = Vec3.init(0,0,0);
    camera.vup      = Vec3.init(0,1,0);


    // camera focus
    camera.defocus_angle = 0.6;
    camera.focus_dist = 10.0;

    try camera.render(world.hittable());
}


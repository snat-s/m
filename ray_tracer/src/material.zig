const Hittable = @import("hittable.zig").Hittable;
const HitRecord = @import("hittable.zig").HitRecord;
const Ray = @import("ray.zig").Ray;
const Vec3 = @import("vec3.zig").Vec3;
const std = @import("std");
const rtweekend = @import("rtweekend.zig");

// i learned from ai that besides vtables we also have tagged unions! endedup
// using it for this two abstractions but i liked learning this. so this is way
// different from the original book

//pub const Material = struct {
//};
//
//pub const Lambertian = struct {
//    albedo: Vec3,
//
//    pub fn scatter(self: Lambertian, r_in: Ray, rec: HitRecord, attenuation: Vec3, scattered: Ray) bool {
//        const scatter_direction = rec.normal + Vec3.random_unit_vector();
//        if (scatter_direction.near_zero()) {
//            scatter_direction = rec.normal;
//        }
//        scattered = Ray.init(rec.p, scatter_direction);
//        attenuation = self.albedo;
//        return true;
//    }
//};
//
//pub const Metal = struct {
//    albedo: Vec3,
//
//    pub fn scatter(self: Metal, r_in: Ray, rec: HitRecord, attenuation: Vec3, scattered: Ray) bool {
//        const reflected: Vec3 = Vec3.reflect(r_in.direction(), rec.normal);
//        scattered = Ray.init(rec.p, reflected);
//        attenuation = self.albedo;
//        return true;
//    }
//
//};
//
//
pub const Lambertian = struct {
    albedo: Vec3,
};

pub const Metal = struct {
    albedo: Vec3,
    fuzz: f32,
};
pub const Dielectric = struct {
    refraction_index: f32,
};

pub const Material = union(enum) {
    lambertian: Lambertian,
    metal: Metal,
    dielectric: Dielectric,

    pub fn scatter(self: Material, r_in: Ray, rec: *HitRecord, attenuation: *Vec3, scattered: *Ray) bool {
        return switch (self) {
            .lambertian => |l| {
                var scatter_direction = Vec3.sum(rec.normal, Vec3.random_on_hemisphere(rec.normal));
                if (scatter_direction.near_zero()) {
                    scatter_direction = rec.normal;
                }
                scattered.* = Ray.init(rec.p, scatter_direction);
                attenuation.* = l.albedo;
                return true;
            },
            .metal => |m| {
                const reflected = Vec3.reflect(r_in.direction, rec.normal);

                const fuzzed_direction = Vec3.sum(reflected, Vec3.multiply_s(Vec3.random_unit_vector(), m.fuzz));
                scattered.* = Ray.init(rec.p, fuzzed_direction);
                attenuation.* = m.albedo;
                return Vec3.dot(scattered.direction, rec.normal) > 0;
            },
            .dielectric => |d| {
                attenuation.* = Vec3.init(1.0,1.0,1.0);
                const ri = if (rec.front_face) (1.0 / d.refraction_index) else d.refraction_index;
                const unit_direction: Vec3 = Vec3.unit_vector(r_in.direction);
                const cos_theta: f32 = @min(Vec3.dot(Vec3.negate(unit_direction), rec.normal), 1.0);
                const sin_theta: f32 = std.math.sqrt(1.0 - cos_theta*cos_theta);
                const cannot_refract: bool = (ri * sin_theta) > 1.0;
                var direction: Vec3 = undefined;

                if (cannot_refract or (reflectance(cos_theta, ri) > rtweekend.random_number())) {
                    direction = Vec3.reflect(unit_direction, rec.normal);
                } else {
                    direction = Vec3.refract(unit_direction, rec.normal, ri);
                }

                scattered.* = Ray.init(rec.p, direction);
                return true;
            }
        };
    }
    pub fn reflectance(cosine: f32, refraction_index: f32) f32 {
        var r0: f32 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*std.math.pow(f32, (1 - cosine),5);
    }
};

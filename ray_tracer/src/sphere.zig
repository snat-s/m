const std = @import("std");
const Vec3 = @import("vec3.zig").Vec3;
const Ray = @import("ray.zig").Ray;
const Hittable = @import("hittable.zig").Hittable;
const HitRecord = @import("hittable.zig").HitRecord;
const Interval = @import("interval.zig").Interval;
const Material = @import("material.zig").Material;

pub const Sphere = struct {
    
    radius: f32,
    center: Vec3,
    mat: *const Material,

    pub fn init(center: Vec3, radius: f32, mat: *const Material) Sphere {
        return Sphere{
            .radius = if (radius < 0) 0 else radius,
            .center = center,
            .mat = mat,
        };
    }

    pub fn hit(self: *const Sphere, ray: Ray, ray_t: Interval, rec: *HitRecord) bool {
        const oc: Vec3 = Vec3.sum(self.center, Vec3.negate(ray.origin));
        const a: f32 = ray.direction.length_squared();
        const h: f32 = Vec3.dot(ray.direction, oc);
        const c: f32 = oc.length_squared() - self.radius*self.radius;

        const discriminant = h*h - a*c;

        if (discriminant < 0) {
            return false;
        }

        const sqrtd: f32 = std.math.sqrt(discriminant);
        var root: f32 = (h-sqrtd) / a;

        if (!ray_t.surrounds(root)) {
            root = (h+sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                return false;
            }
        }
        
        // kind of nasty, not sure if this is the proper way of doing this
        rec.*.t = root;
        rec.*.p = ray.at(rec.*.t);
        const outward_normal = Vec3.division_s(Vec3.sum(rec.*.p, Vec3.negate(self.center)), self.radius);
        rec.set_face_normal(ray, outward_normal);
        rec.mat = self.mat;
        return true;
    }

    const vtable = Hittable.VTable{
        .hit = sphereHit,
    };

    pub fn hittable(self: *Sphere) Hittable {
        return Hittable {
            .self = self,
            .vtable = &vtable,
        };
    }
};

fn sphereHit(self_ptr: *anyopaque, r: Ray, ray_t: Interval, rec: *HitRecord) bool{
    const self: *const Sphere = @ptrCast(@alignCast(self_ptr));
    return self.hit(r, ray_t, rec);
}

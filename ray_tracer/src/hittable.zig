const std = @import("std");
const Vec3 = @import("vec3.zig").Vec3;
const Ray = @import("ray.zig").Ray;
const Interval = @import("interval.zig").Interval;
const Material = @import("material.zig").Material;

pub const HitRecord = struct {
    p: Vec3,
    normal: Vec3,
    mat: *const Material,
    t: f32,
    front_face: bool,

    pub fn set_face_normal(self: *HitRecord, r: Ray, outward_normal: Vec3) void {
        // `outward_normal` assumed to have unit length
        self.front_face = Vec3.dot(r.direction, outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else Vec3.negate(outward_normal);
    }
};

// this was translated by gemini, it's basically the only part that i used it for.
// polymorphism in Zig doesn't feel like the correct answer 
pub const Hittable = struct {
    // A pointer to the concrete object (e.g., a Sphere, a HittableList).
    // *anyopaque is a type-erased pointer.
    self: *anyopaque,
    // A pointer to a table of functions for this type.
    vtable: *const VTable,

    // The VTable defines all the functions a "Hittable" object must have.
    // For now, it's just `hit`.
    pub const VTable = struct {
        hit: *const fn (self: *anyopaque, r: Ray, ray_t: Interval, rec: *HitRecord) bool,
    };

    // This is a convenience function to make calling the interface easy.
    // `my_hittable.hit(...)` instead of `my_hittable.vtable.hit(my_hittable.self, ...)`
    pub fn hit(self: Hittable, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
        return self.vtable.hit(self.self, r, ray_t, rec);
    }
};

// Our implementation of HittableList.
pub const HittableList = struct {
    allocator: std.mem.Allocator,
    objects: std.ArrayList(Hittable),

    pub fn init(allocator: std.mem.Allocator) HittableList {
        return HittableList{
            .allocator = allocator,
            .objects = std.ArrayList(Hittable).init(allocator),
        };
    }

    pub fn deinit(self: *HittableList) void {
        self.objects.deinit();
    }

    pub fn add(self: *HittableList, object: Hittable) !void {
        try self.objects.append(object);
    }

    // This is the implementation of the `hit` function for the list itself.
    fn listHit(self_ptr: *anyopaque, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
        const self: *const HittableList = @ptrCast(@alignCast(self_ptr));
        var temp_rec: HitRecord = undefined;
        var hit_anything = false;
        var closest_so_far = ray_t.max;

        for (self.objects.items) |object| {
            if (object.hit(r, Interval.init(ray_t.min, closest_so_far), &temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec.* = temp_rec; // Copy the temporary record to the output record
            }
        }

        return hit_anything;
    }

    // The static vtable for HittableList.
    const vtable = Hittable.VTable{
        .hit = listHit,
    };

    // A function to return this object wrapped in the Hittable interface.
    pub fn hittable(self: *HittableList) Hittable {
        return Hittable{
            .self = self,
            .vtable = &vtable,
        };
    }
};


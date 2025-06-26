const Vec3 = @import("vec3.zig").Vec3;

pub const Ray = struct {
    origin: Vec3,
    direction: Vec3, 

    pub fn init(origin: Vec3, direction: Vec3) Ray {
        return Ray{
            .origin = origin,
            .direction =  direction
        };
    }

    pub fn at(self: Ray, t: f32) Vec3 {
        return Vec3.sum(self.origin, Vec3.multiply_s(self.direction, t));
    }
};

const std = @import("std");
const random_number = @import("rtweekend.zig").random_number;
const random_number_in_interval = @import("rtweekend.zig").random_number_in_interval;
// in theory you also alias this to Vec3 and Point3 but I forgot about
// it and just left it, at the end of the day it makes sense in my head
// but in theory if this was more serious it would give some really bad
// headaches
pub const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,
    pub fn init(x: f32, y: f32, z: f32) Vec3 {
        return Vec3{
            .x = x,
            .y = y,
            .z = z
        };
    }
    pub fn sum(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x + other.x,
            .y = self.y + other.y,
            .z = self.z + other.z,
        };
    }
    pub fn sum_s(self: Vec3, t: f32) Vec3 {
        return Vec3{
            .x = self.x + t,
            .y = self.y + t,
            .z = self.z + t,
        };
    }
    pub fn multiply(self: Vec3, other: Vec3) Vec3 {
        return Vec3{
            .x = self.x * other.x,
            .y = self.y * other.y,
            .z = self.z * other.z,
        };
    }
    pub fn multiply_s(self: Vec3, t: f32) Vec3 {
        return Vec3{
            .x = self.x * t,
            .y = self.y * t,
            .z = self.z * t,
        };
    }

    pub fn negate(self: Vec3) Vec3 {
        return Vec3{
            .x = -self.x,
            .y = -self.y,
            .z = -self.z,
        };
    }

    pub fn division_s(self: Vec3, t: f32) Vec3 {
        return Vec3{
            .x = self.x / t,
            .y = self.y / t,
            .z = self.z / t,
        };
    }

    pub fn length(self: Vec3) f32 {
        return std.math.sqrt(self.length_squared());
    }
    pub fn unit_vector(self: Vec3) Vec3 {
        return Vec3.division_s(self, self.length());
    }
    pub fn length_squared(self: Vec3) f32 {
        return self.x*self.x + self.y*self.y + self.z*self.z;
    }
    pub fn dot(self: Vec3, v: Vec3) f32 {
        return self.x * v.x + self.y * v.y + self.z * v.z;
    }
    pub fn cross(u: Vec3, v: Vec3) Vec3 {
        return Vec3{
            .x = u.y * v.z - u.z * v.y,
            .y = u.z * v.x - u.x * v.z,
            .z = u.x * v.y - u.y * v.x,
        };
    }
    pub fn random() Vec3 {
        return Vec3.init(random_number(), random_number(), random_number());
    }
    pub fn param_random(min: f32, max: f32) Vec3 {
        return Vec3.init(random_number_in_interval(min, max), random_number_in_interval(min, max), random_number_in_interval(min, max));
    }
    pub fn random_unit_vector() Vec3 {
        while (true) {
            const p : Vec3 = Vec3.param_random(-1, 1);
            const lensq = p.length_squared();
            if ((1e-80 < lensq) and (lensq <= 1)) {
                return Vec3.division_s(p, std.math.sqrt(lensq));
            }
        }
    }
    pub fn random_on_hemisphere(normal: Vec3) Vec3 {
        const on_unit_sphere: Vec3 = Vec3.random_unit_vector();
        if (Vec3.dot(on_unit_sphere, normal) > 0.0) {
            return on_unit_sphere;
        } else {
            return Vec3.negate(on_unit_sphere);
        }
    }
    pub fn near_zero(self: Vec3) bool {
        const s:f32 = 1e-8;
        return (@abs(self.x) < s) and (@abs(self.y) < s) and (@abs(self.z) < s);
    }
    pub fn reflect(v: Vec3, n: Vec3) Vec3 {
        return Vec3.sum(v, Vec3.negate(Vec3.multiply_s(n, Vec3.dot(v, n)*2)));
    }

    pub fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) Vec3 {
        const cos_theta = @min(Vec3.dot(Vec3.negate(uv), n), 1.0);
        const r_out_perp: Vec3 = Vec3.multiply_s(Vec3.sum(uv, Vec3.multiply_s(n, cos_theta)), etai_over_etat);
        const r_out_parallel: Vec3 = Vec3.multiply_s(n,-std.math.sqrt(@abs(1.0-r_out_perp.length_squared())));
        return Vec3.sum(r_out_perp, r_out_parallel);
    }
    pub fn random_in_unit_disk() Vec3 {
        while (true) {
            const p = Vec3.init(random_number_in_interval(-1,1), random_number_in_interval(-1,1), 0);
            if (p.length_squared() < 1) {
                return p;
            }
        }
    }
};

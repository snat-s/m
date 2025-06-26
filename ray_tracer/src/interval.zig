const infinity = @import("rtweekend.zig").infinity;

pub const Interval = struct {
    min: f32,
    max: f32,

    pub fn init(min: f32, max: f32) Interval {
        return Interval{
            .min = min,
            .max = max,
        };
    }
   //pub fn init_e() Interval {
   //    return Interval{
   //        min = -infinity,
   //        max = infinity,
   //    };
   //}
    pub fn size(self: Interval) f32 {
        return self.max - self.min;
    }

    pub fn surrounds(self: Interval, x: f32) bool {
        return (self.min < x) and (x < self.max);
    }
    pub fn clamp(self: Interval, x:f32) f32 {
        if (x < self.min) {
            return self.min;
        }
        if (x > self.max) {
            return self.max;
        }
        return x;
    }
};

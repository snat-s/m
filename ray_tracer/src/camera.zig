const Vec3 = @import("vec3.zig").Vec3;
const Ray = @import("ray.zig").Ray;
const Hittable = @import("hittable.zig").Hittable;
const HitRecord = @import("hittable.zig").HitRecord;
const Interval = @import("interval.zig").Interval;
const std = @import("std");
const write_color = @import("color.zig").write_color;
const infinity = @import("rtweekend.zig").infinity;
const random_number = @import("rtweekend.zig").random_number;
const degrees_to_radians = @import("rtweekend.zig").degrees_to_radians;


pub const Camera = struct {
    aspect_ratio: f32,
    image_width: u64,
    image_height: u64,
    camera_center: Vec3,
    pixel00_loc: Vec3,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    samples_per_pixel: u64,
    pixel_sample_scale: f32,
    max_depth: i16,
    vfov: f32,
    lookfrom: Vec3,
    lookat: Vec3,
    vup: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    defocus_angle: f32,
    focus_dist: f32,
    defocus_disk_u: Vec3,
    defocus_disk_v: Vec3,

    pub fn render(self: *Camera, world: Hittable) !void {
        const stdout = std.io.getStdOut().writer();
        const stderr = std.io.getStdErr().writer();

        self.initialize();

        try stdout.print("P3\n{d} {d}\n255\n", .{self.image_width, self.image_height});

        for (0..self.image_height) |j| {

            try stderr.print("\r Scanlines remaining {d}", .{self.image_width-j});

            for (0..self.image_width) |i| {
                var pixel_color: Vec3 = Vec3.init(0,0,0);

                // i don't get why this has to be u64 and 
                // can't be i16 or something more resonable
                for (0 .. self.samples_per_pixel) |_| {
                    const r: Ray = get_ray(self, i, j);
                    pixel_color = Vec3.sum(ray_color(r, self.max_depth, world), pixel_color);
                }
                try write_color(stdout, Vec3.multiply_s(pixel_color, self.pixel_sample_scale));
            }
        }
        try stderr.print("\r Done.                                \n",.{});
    }

    fn initialize(self: *Camera) void {
        //self.samples_per_pixel = 10;
        //self.max_depth = 10;
        //self.lookfrom = Vec3.init(0,0,0);
        //self.lookat = Vec3(0,0,-1);
        //self.vup = Vec3(0,1,0);
        //self.vfov = 90;

        self.image_height = @intFromFloat(@as(f32, @floatFromInt(self.image_width)) / self.aspect_ratio);
        self.image_height = if (self.image_height < 1) 1 else self.image_height;
        self.pixel_sample_scale = 1.0 / @as(f32, @floatFromInt(self.samples_per_pixel));

        // camera
        //const focal_length : f32 = Vec3.sum(self.lookfrom, Vec3.negate(self.lookat)).length();
        const theta = degrees_to_radians(self.vfov);
        const h = std.math.tan(theta/2);
        const viewport_height : f32 = 2 * h * self.focus_dist;
        const viewport_width : f32 = viewport_height * (@as(f32, @floatFromInt(self.image_width)) / @as(f32, @floatFromInt(self.image_height)));
        self.camera_center = self.lookfrom; //Vec3.init(0,0,0);
                                        
        // vecs u,v,w basis vectors for the camera coordinate frame
        self.w = Vec3.unit_vector(Vec3.sum(self.lookfrom, Vec3.negate(self.lookat)));
        self.u = Vec3.unit_vector(Vec3.cross(self.vup, self.w));
        self.v = Vec3.cross(self.w,self.u);


        // vecs for horizonal and down the vertical viewport edges
        const viewport_u: Vec3 = Vec3.multiply_s(self.u, viewport_width);
        const viewport_v: Vec3 = Vec3.multiply_s(Vec3.negate(self.v), viewport_height);

        // delta vectors for horizontal and vertical vectors from pixel to pixel
        self.pixel_delta_u = Vec3.division_s(viewport_u, @floatFromInt(self.image_width));
        self.pixel_delta_v = Vec3.division_s(viewport_v, @floatFromInt(self.image_height));

        // upper left pixel
        const viewport_upper_left: Vec3 = Vec3.sum(
            self.camera_center,
            Vec3.sum( 
                Vec3.negate(Vec3.multiply_s(self.w, self.focus_dist)),
                Vec3.sum(
                    Vec3.negate(Vec3.division_s(viewport_u,2.0)),
                    Vec3.negate(Vec3.division_s(viewport_v,2.0)),
                )
            )
        );

        self.pixel00_loc = Vec3.sum(viewport_upper_left, Vec3.multiply_s(Vec3.sum(self.pixel_delta_u, self.pixel_delta_v), 0.5));
        const defocus_radius = self.focus_dist * std.math.tan(degrees_to_radians(self.defocus_angle/2));
        self.defocus_disk_u = Vec3.multiply_s(self.u, defocus_radius);
        self.defocus_disk_v = Vec3.multiply_s(self.v, defocus_radius);
    }

    fn get_ray(self: *Camera, i: u64, j: u64) Ray {
        const offset = sample_square();
        const pixel_sample = Vec3.sum(self.pixel00_loc,Vec3.sum(Vec3.multiply_s(self.pixel_delta_u, @as(f32, @floatFromInt(i)) + offset.x), Vec3.multiply_s(self.pixel_delta_v, @as(f32, @floatFromInt(j)) + offset.y)));

        const ray_origin = if (self.defocus_angle <= 0) self.camera_center else self.defocus_disk_sample();
        const ray_direction = Vec3.sum(pixel_sample, Vec3.negate(ray_origin));

        return Ray.init(ray_origin, ray_direction);
    }

    fn sample_square() Vec3 {
        return Vec3.init(random_number() - 0.5, random_number() - 0.5, 0);
    }

    fn ray_color(r: Ray, depth: i16, world: Hittable) Vec3 {
        // return no more light
        if (depth <= 0) {
            return Vec3.init(0,0,0);
        }
        
        var rec: HitRecord = undefined;
        // you can avoid the ugly look if you put in here 0.001 
        // so this thing is called shadow acne, not making it UP HAHA
        if (world.hit(r, Interval.init(0.001, infinity), &rec)) {
            var scattered: Ray = undefined;
            var attenuation: Vec3 = undefined;

            if (rec.mat.scatter(r, &rec, &attenuation, &scattered)) {
                return Vec3.multiply(attenuation, ray_color(scattered, depth-1, world));
            }
            return Vec3.init(0,0,0);
            //const direction: Vec3 = Vec3.sum(rec.normal, Vec3.random_unit_vector());
            //return Vec3.multiply_s(ray_color(Ray.init(rec.p, direction), depth-1, world), 0.9);
        }

        const unit_direction: Vec3 = Vec3.unit_vector(r.direction);
        const a : f32 = 0.5*(unit_direction.y + 1.0);
        return Vec3.sum(Vec3.multiply_s(Vec3.init(1.0,1.0,1.0), (1.0-a)), Vec3.multiply_s(Vec3.init(0.5,0.7,1.0), a));
    }

    fn defocus_disk_sample(self: *Camera) Vec3 {
        const p: Vec3 = Vec3.random_in_unit_disk();
        return Vec3.sum(self.camera_center,
            Vec3.sum(
                Vec3.multiply_s(self.defocus_disk_u, p.x), 
                Vec3.multiply_s(self.defocus_disk_v, p.y)
                )
            );
    }
};

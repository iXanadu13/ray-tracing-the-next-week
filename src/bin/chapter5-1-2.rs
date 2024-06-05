use indicatif::ParallelProgressIterator;
use itertools::Itertools;
//use noise::{NoiseFn, Perlin};
use rand::Rng;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{ops::Range,ops::Neg, fs, path::Path, env, io};
use glam::DVec3 as vec3;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use image::{DynamicImage, GenericImageView};


fn main() -> Result<(), Box<dyn std::error::Error>>{
    let args: Vec<String> = env::args().collect();
    let mut func: 
        fn(&mut HittableList)-> Result<CameraBuilder, Box<dyn std::error::Error>> = perlin_spheres;
    if args.len() > 1{
        let shading_mode = args[1].parse::<i32>();
        match shading_mode {
            Ok(1) => func = bouncing_spheres,
            Ok(2) => func = checker_spheres,
            Ok(3) => func = earth,
            Ok(4) => func = perlin_spheres,
            _ => {
                println!("Unknown parameter!");
            }
        }
    }

    let mut world = HittableList::new();
    let builder = func(&mut world)?;
    let camera = builder.build();

    let st = match SystemTime::now().duration_since(UNIX_EPOCH){
        Ok(d) => d.as_millis(),
        Err(_) => panic!("failed to get current time")
    };

    camera.render(&world)?;

    let ed =  match SystemTime::now().duration_since(UNIX_EPOCH){
        Ok(d) => d.as_millis(),
        Err(_) => panic!("failed to get current time")
    };
    println!("Time elapsed: {} ms",(ed-st));

    Ok(())
    
}

fn bouncing_spheres(world: &mut HittableList) -> Result<CameraBuilder, Box<dyn std::error::Error>>{
    let checker = Texture::Checker {
        even: vec3::new(0.2, 0.3, 0.1),
        odd: vec3::new(0.9, 0.9, 0.9),
        scale: 0.32
    };
    world.add(
        Sphere::new(
            vec3::new(0.0, -1000.0, 0.0),
            1000.,
            Material::Lambertian { albedo: checker }
        )
    );

    let mut rng = rand::thread_rng();
    for (a, b) in (-11..11).cartesian_product(-11..11).into_iter()
    {
        let choose_mat = rng.gen::<f64>();
        let center = vec3::new(
            a as f64 + 0.9 * rng.gen::<f64>(),
            0.2,
            b as f64 + 0.9 * rng.gen::<f64>()
        );
        if (center - vec3::new(4., 0.2, 0.)).length() > 0.9 {
            if choose_mat < 0.8 {
                // diffuse
                let albedo = vec3::new(
                    rng.gen_range(0f64..1.),
                    rng.gen_range(0f64..1.),
                    rng.gen_range(0f64..1.),
                ) * vec3::new(
                    rng.gen_range(0f64..1.),
                    rng.gen_range(0f64..1.),
                    rng.gen_range(0f64..1.),
                );
                let sphere_material = Material::Lambertian { albedo: Texture::from(albedo) };
                let center2 = center + vec3::new(0., rng.gen_range(0.0..0.5), 0.);
                world.add(
                    Sphere::new(
                        center,
                        0.2,
                        sphere_material
                    ).with_move_to(center2)
                );
            } else if choose_mat < 0.95 {
                // metal
                let albedo = vec3::new(
                    rng.gen_range(0.5..1.),
                    rng.gen_range(0.5..1.),
                    rng.gen_range(0.5..1.),
                );
                let fuzz = rng.gen_range(0f64..0.5);
                let sphere_material = Material::Metal { albedo, fuzz };
                world.add(Sphere::new(center, 0.2, sphere_material));
            } else {
                // glass
                let sphere_material = Material::Dielectric { refraction_index: 1.5 };
                world.add(Sphere::new(center, 0.2, sphere_material));
            };
            
        }
    }

    world.add(
        Sphere::new(
            vec3::new(0., 1., 0.),
            1.0,
            Material::Dielectric { refraction_index: 1.5 }
        )
    );
    world.add(
        Sphere::new(
            vec3::new(-4., 1., 0.),
            1.0,
            Material::Lambertian { albedo: Texture::from((0.4, 0.2, 0.1)) }
        )
    );
    world.add(
        Sphere::new(
            vec3::new(4., 1., 0.),
            1.0,
            Material::Metal { albedo: vec3::new(0.7, 0.6, 0.5), fuzz: 0.0 }
        )
    );

    let mut builder = CameraBuilder::new(
        400,
        16.0 / 9.0
    )
        .samples_per_pixel(100)
        .max_depth(50)
        .lookfrom(13., 2., 3.)
        .lookat(0., 0., 0.);
    builder.vup = vec3::new(0., 1., 0.);
    builder.vfov = 20.;

    builder.defocus_angle = 0.6;
    builder.focus_dist = 10.0;

    Ok(builder)
}

fn checker_spheres(world: &mut HittableList) -> Result<CameraBuilder, Box<dyn std::error::Error>>{
    let checker = Texture::Checker {
        even: vec3::new(0.2, 0.3, 0.1),
        odd: vec3::new(0.9, 0.9, 0.9),
        scale: 0.32
    };
    world.add(
        Sphere::new(
            vec3::new(0.0, -10.0, 0.0),
            10.,
            Material::Lambertian { albedo: checker.clone() }
        )
    );
    world.add(
        Sphere::new(
            vec3::new(0.0, 10.0, 0.0),
            10.,
            Material::Lambertian { albedo: checker }
        )
    );
    let mut builder = CameraBuilder::new(
        400,
        16.0 / 9.0
    )
        .samples_per_pixel(100)
        .max_depth(50)
        .lookfrom(13., 2., 3.)
        .lookat(0., 0., 0.);
    builder.vup = vec3::new(0., 1., 0.);
    builder.vfov = 20.;

    builder.defocus_angle = 0.;
    //builder.focus_dist = 10.0;

    Ok(builder)

}

fn earth(world: &mut HittableList) -> Result<CameraBuilder, Box<dyn std::error::Error>>{
    let earth_texture = Texture::load_image("assets/earthmap.jpg").unwrap();
    let earth_surface = Material::Lambertian { albedo: earth_texture };
    let globe = Sphere::new(
        vec3::new(0., 0., 0.),
        2., 
        earth_surface
    );
    world.add(globe);
    

    let mut builder = CameraBuilder::new(
        400,
        16.0 / 9.0
    )
        .samples_per_pixel(100)
        .max_depth(50)
        .lookfrom(0., 0., 12.)
        .lookat(0., 0., 0.);
    builder.vup = vec3::new(0., 1., 0.);
    builder.vfov = 20.;

    builder.defocus_angle = 0.;
    //builder.focus_dist = 10.0;

    Ok(builder)
}

fn perlin_spheres(world: &mut HittableList) -> Result<CameraBuilder, Box<dyn std::error::Error>>{
    let pertext = Texture::PerlinNoise(Perlin::default());
    world.add(Sphere{
        center: vec3::new(0., -1000., 0.),
        radius: 1000.,
        material: Material::Lambertian { albedo: pertext.clone() },
        move_to: None
    });
    world.add(Sphere{
        center: vec3::new(0., 2., 0.),
        radius: 2.,
        material: Material::Lambertian { albedo: pertext },
        move_to: None
    });

    let mut builder = CameraBuilder::new(
        400,
        16.0 / 9.0
    )
        .samples_per_pixel(100)
        .max_depth(50)
        .lookfrom(13., 2., 3.)
        .lookat(0., 0., 0.);
    builder.vup = vec3::new(0., 1., 0.);
    builder.vfov = 20.;

    builder.defocus_angle = 0.;

    Ok(builder)
}
struct Ray{
    origin: vec3,
    direction: vec3,
    time: f64
}
impl Ray{
    fn at(&self, t: f64) -> vec3 {
        self.origin + t * self.direction
    }
    fn color<T>(&self, depth: u32, world: &T) -> vec3 
    where
        T: Hittable
    {
        if depth <= 0 {
            return vec3::ZERO;
        }
        if let Some(rec) = 
            // 使用0.001，避免t过小时浮点误差导致反复hit
            world.hit(&self, (0.001)..f64::INFINITY)
        {
            let optional = rec.material.scatter(&self, &rec);

            return optional.map_or_else(|| vec3::ZERO, |f|{
                f.attenuation * f.scattered.color(depth-1, world)
            });
        }
        let unit_direction: vec3 = 
            self.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        // 线性插值
        // blendedValue=(1−a)⋅startValue+a⋅endValue
        return (1.0 - a) * vec3::new(1.0, 1.0, 1.0)
            + a * vec3::new(0.5, 0.7, 1.0);
    }
}

struct Sphere{
    center: vec3,
    radius: f64,
    material: Material,
    move_to: Option<vec3>,
}
impl Sphere {
    fn new(
        center: vec3,
        radius: f64,
        material: Material
    ) -> Self{
        Self{
            center,
            radius,
            material,
            move_to: None
        }
    }
    fn with_move_to(mut self, to: vec3) -> Self{
        self.move_to = Some(to - self.center);
        self
    }
    fn get_center(&self, time: f64) -> vec3 {
        match self.move_to {
            Some(vec) => {
                self.center + time * vec
            }
            None => self.center
        }
    }
    fn get_sphere_uv(&self, p: vec3) -> (f64, f64) {
        use std::f64::consts::PI as PI;
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>
        let theta = (-p.y).acos();
        let phi = (-p.z).atan2(p.x) + PI;
        let u = phi / (2. * PI);
        let v = theta / PI;
        (u, v)
    }
}
impl Hittable for Sphere{
    fn hit(
        &self,
        ray: &Ray,
        interval: Range<f64>
    ) -> Option<HitRecord> {
        let center = self.get_center(ray.time);
        let oc: vec3 = ray.origin - center;  // we use Q - C instead
        let a = ray.direction.length_squared();
        let half_b = ray.direction.dot(oc);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0. {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        // Find the nearest root that lies in the acceptable range.
        let mut root = (-half_b - sqrtd) / a;
        if !interval.contains(&root) {
            root = (-half_b + sqrtd) / a;
            if !interval.contains(&root) {
                return None;
            }
        }
        let t = root;
        let point = ray.at(t);
        let outward_normal = (point - center) / self.radius;
        let (u, v) = self.get_sphere_uv(outward_normal);
        let rec = HitRecord::with_face_normal(
            &self.material,
            point,
            outward_normal,
            t,
            ray,
            u,
            v
        );
        Some(rec)

    }
}


struct HitRecord<'a>{
    point: vec3,
    normal: vec3, // normalized
    t: f64,
    front_face: bool,
    material: &'a Material,
    u: f64,
    v: f64
}
impl<'a> HitRecord<'a>{
    fn calc_face_normal(outward_normal: &vec3, ray: &Ray) -> (bool, vec3){
        let front_face = ray.direction.dot(*outward_normal) < 0.;
        let normal = if front_face {
            *outward_normal
        }
        else{
            - *outward_normal
        };
        (front_face, normal)
    }
    fn with_face_normal(
        material: &'a Material,
        point: vec3,
        outward_normal: vec3,
        t: f64,
        ray: &Ray,
        u: f64,
        v: f64
    ) -> Self{
        let (front_face, normal) = HitRecord::calc_face_normal(&outward_normal, ray);
        HitRecord{
            material,
            point,
            normal,
            t,
            front_face,
            u,
            v
        }
    }
}

trait Hittable {
    fn hit(
        &self,
        ray: &Ray,
        interval: Range<f64>
    ) -> Option<HitRecord>;
}

struct HittableList{
    objects: Vec<Box<dyn Hittable + Sync>>
}
impl HittableList{
    fn new() -> Self{
        HittableList{
            objects: vec![]
        }
    }
    #[allow(dead_code)]
    fn clear(&mut self){
        self.objects = vec![];
    }
    fn add<T>(&mut self, object: T)
    where
        T: Hittable + 'static + Sync,
    {
        self.objects.push(Box::new(object));
    }
}

impl Hittable for HittableList{
    fn hit(
        &self,
        ray: &Ray,
        interval: Range<f64>
    ) -> Option<HitRecord> {
        let (_closest, hit_record) = self
            .objects
            .iter()
            .fold((interval.end, None), |acc, item|{
                if let Some(temp_rec) = item.hit(
                    ray,
                    interval.start..acc.0,
                ){
                    (temp_rec.t, Some(temp_rec))
                }
                else {
                    acc
                }
            });
        hit_record
    }
}

#[derive(Clone)]
enum Material {
    Lambertian {albedo: Texture},
    Metal {albedo: vec3, fuzz: f64},
    Dielectric {refraction_index: f64},
}
struct Scattered {
    attenuation: vec3,
    scattered: Ray
}
impl Material{
    fn scatter(
        &self,
        r_in: &Ray,
        hit_record: &HitRecord
    ) -> Option<Scattered> {
        match self {
            Material::Lambertian { albedo } => {
                // Lambertian反射既可以始终散射并根据反射率R衰减光线，
                // 也可以有时散射（概率为1-R）而不衰减（未散射的光线会被材料吸收），
                // 也可以是这两种策略的混合。
                // 这里采用的是始终散射
                let mut scatter_direction = hit_record.normal + random_unit_vector();
                // Catch degenerate scatter direction
                if scatter_direction.abs_diff_eq(
                    vec3::ZERO,
                    1e-8
                ) {
                    scatter_direction = hit_record.normal;
                }
                let scattered = Ray{
                    origin: hit_record.point,
                    direction: scatter_direction,
                    time: r_in.time
                };
                Some(
                    Scattered{
                        attenuation: albedo.color(
                            hit_record.u,
                            hit_record.v,
                            hit_record.point
                        ),
                        scattered
                    }
                )
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = reflect(r_in.direction.normalize(), hit_record.normal);
                
                let scattered = Ray{
                    origin: hit_record.point,
                    direction: reflected + *fuzz * random_unit_vector(),
                    time: r_in.time
                };
                if scattered.direction.dot(hit_record.normal) > 0. {
                    Some(
                        Scattered{
                            attenuation: *albedo,
                            scattered
                        }
                    )
                }
                // 吸收散射到表面下的光线
                else {
                    None
                }
            }
            Material::Dielectric { refraction_index } => {
                let mut rng = rand::thread_rng();
                let attenuation = vec3::splat(1.);
                let refraction_index = if hit_record.front_face {
                    refraction_index.recip()
                } 
                else {
                    *refraction_index
                };
                let unit_direction = r_in.direction.normalize();
                let cos_theta = unit_direction
                    .neg()
                    .dot(hit_record.normal)
                    .min(1.);
                let sin_theta = (1. - cos_theta * cos_theta).sqrt();
                let cannot_refract = refraction_index * sin_theta > 1.;
                let direction = if cannot_refract 
                    || reflectance(
                        cos_theta,
                        refraction_index,
                    ) > rng.gen::<f64>()
                {
                    reflect(unit_direction, hit_record.normal)
                } 
                else{
                    refract(unit_direction, hit_record.normal, refraction_index)
                };
                Some(
                    Scattered{
                        attenuation,
                        scattered: Ray{
                            origin: hit_record.point,
                            direction,
                            time: r_in.time
                        }
                    }
                )
            }
        }
    }
}

#[derive(Clone)]
enum Texture {
    SolidColor(vec3),
    Checker{ even: vec3, odd: vec3, scale: f64 },
    Image(DynamicImage),
    PerlinNoise(Perlin),
}
impl Texture {
    fn load_image<P>(path: P) -> io::Result<Self>
    where
        P: AsRef<Path>,
    {
        use image::io::Reader as ImageReader;
        let img = ImageReader::open(path)?
            .decode()
            .unwrap();
        Ok(Self::Image(img))
    }
    fn color(
        &self,
        u: f64,
        v: f64,
        point: vec3
    ) -> vec3 {
        match self{
            Texture::SolidColor(color) => *color,
            Texture::Checker { even, odd, scale } => {
                let inv_scale = scale.recip();
                let x_int = (inv_scale * point.x).floor() as i32;
                let y_int = (inv_scale * point.y).floor() as i32;
                let z_int = (inv_scale * point.z).floor() as i32;
                let is_even = (x_int + y_int + z_int) & 1 == 0;
                if is_even {
                    *even
                } else {
                    *odd
                }
            },
            Texture::Image(image) => {
                // If we have no texture data, then return solid cyan as a debugging aid.
                if image.height() <= 0 {
                    return vec3::new(0., 1., 1.);
                }
                // Clamp input texture coordinates to [0,1] x [1,0]
                let u = u.clamp(0.0, 1.0);
                // Flip V to image coordinates
                let v = 1.0 - v.clamp(0.0, 1.0);
                let i = (u * image.width() as f64) as u32;
                let j = (v * image.height() as f64) as u32;
                let pixel = image.get_pixel(i, j);
                let color_scale = 1.0 / 255.0;
                vec3::new(
                    color_scale * pixel[0] as f64,
                    color_scale * pixel[1] as f64,
                    color_scale * pixel[2] as f64,
                )
            },
            Texture::PerlinNoise(noise) => {
                vec3::ONE * noise.get(point)
            },
        }
    }
}
impl From<vec3> for Texture {
    fn from(vec: vec3) -> Self {
        Texture::SolidColor(vec)
    }
}
impl From<(f64, f64, f64)> for Texture {
    fn from((x, y, z): (f64, f64, f64)) -> Self {
        Texture::SolidColor(vec3::new(x, y, z))
    }
}

#[derive(Clone)]
struct Perlin{
    rand:   [f64; 256],
    perm_x: [u8; 256],
    perm_y: [u8; 256],
    perm_z: [u8; 256],
}
impl Default for Perlin{
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        Perlin {
            rand: (0..256).map(|_| rng.gen()).collect::<Vec<f64>>().try_into().unwrap(),
            perm_x: Perlin::perlin_generate_perm(),
            perm_y: Perlin::perlin_generate_perm(),
            perm_z: Perlin::perlin_generate_perm(),
        }
    }
}
impl Perlin{
    fn get(&self, point: vec3) -> f64 {
        let i = ((4. * point.x) as i32 & 255) as usize;
        let j = ((4. * point.y) as i32 & 255) as usize;
        let k = ((4. * point.z) as i32 & 255) as usize;
        // enhance the hash algorithm
        // let hash: usize = (self.perm_x[i] ^ self.perm_y[j] ^ self.perm_z[k]).into();
        let hash: usize = (self.perm_x[i] ^ self.perm_y[j] * 13 ^ self.perm_z[k] * 31).into();
        self.rand[hash]
    }
    fn perlin_generate_perm() -> [u8; 256] {
        let mut p = (0..256).map(|x| x as u8).collect();
        Perlin::permute(&mut p);
        p.try_into().unwrap()
    }
    fn permute(p: &mut Vec<u8>){
        let mut rng = rand::thread_rng();
        for i in 255usize..0 {
            let target = rng.gen_range(0..i+1);
            p.swap(i, target);
        }
    }
}

struct CameraBuilder{
    image_width: u32,
    aspect_ratio: f64,
    samples_per_pixel: u32,
    max_depth: u32,
    vfov: f64,
    lookfrom: vec3,
    lookat: vec3,
    vup: vec3,
    defocus_angle: f64,
    focus_dist: f64, // Distance from camera lookfrom point to plane of perfect focus
}
impl Default for CameraBuilder{
    fn default() -> Self {
        Self{
            image_width: 400,
            aspect_ratio: 16.0 / 9.0,
            samples_per_pixel: 100,
            max_depth: 10,
            vfov: 90f64,
            lookfrom: vec3::ZERO,
            lookat: vec3::NEG_Z,
            vup: vec3::Y,
            defocus_angle: 0.,
            focus_dist: 10.
        }
    }
}
impl CameraBuilder{
    fn new(image_width: u32, aspect_ratio: f64) -> Self {
        let mut this = Self::default();
        this.image_width = image_width;
        this.aspect_ratio = aspect_ratio;
        this
    }
    fn max_depth(mut self, max_depth: u32) -> Self{
        self.max_depth = max_depth;
        self
    }
    fn samples_per_pixel(mut self, samples_per_pixel: u32) -> Self{
        self.samples_per_pixel = samples_per_pixel;
        self
    }
    fn lookfrom(mut self, x: f64, y: f64, z: f64) -> Self{
        self.lookfrom = vec3::new(x, y, z);
        self
    }
    fn lookat(mut self, x: f64, y: f64, z: f64) -> Self{
        self.lookat = vec3::new(x, y, z);
        self
    }

    fn build(self) -> Camera{
        let max_value: u8 = 255;
        let image_width = self.image_width;
        let image_height: u32 = (image_width as f64 / self.aspect_ratio) as u32;
        // Calculate the image height, and ensure that it's at least 1.
        if image_height < 1 {
            panic!("image height is at least 1");
        }
        let center = self.lookfrom;
        //let focal_length: f64 = (self.lookfrom - self.lookat).length();
        let theta = self.vfov.to_radians();
        let h = (theta / 2.).tan();
        let viewport_height = 2. * h * self.focus_dist;
        let viewport_width: f64 = viewport_height * (image_width as f64/image_height as f64);

        let w = (self.lookfrom - self.lookat).normalize();
        let u = self.vup.cross(w).normalize();
        let v = w.cross(u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        let viewport_u: vec3 = viewport_width * u;
        let viewport_v: vec3 = -viewport_height * v;

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        let pixel_delta_u: vec3 = viewport_u / image_width as f64;
        let pixel_delta_v: vec3 = viewport_v / image_height as f64;

        // Calculate the location of the upper left pixel.
        let viewport_upper_left: vec3 = center
            - self.focus_dist * w
            - viewport_u / 2.
            - viewport_v / 2.;
        let pixel00_loc: vec3 = viewport_upper_left
            + 0.5 * (pixel_delta_u + pixel_delta_v);

        let samples_per_pixel = self.samples_per_pixel;
        let pixel_samples_scale = (samples_per_pixel as f64).recip();

        let defocus_radius = self.focus_dist
            * (self.defocus_angle / 2.).to_radians().tan();
        let defocus_disk_u = u * defocus_radius;
        let defocus_disk_v = v * defocus_radius;

        Camera{
            image_width,
            image_height,
            max_value,
            _aspect_ratio: self.aspect_ratio,
            center,
            pixel_delta_u,
            pixel_delta_v,
            pixel00_loc,
            samples_per_pixel,
            pixel_samples_scale,
            max_depth: self.max_depth,
            vfov: self.vfov,
            u,v,w,
            defocus_angle: self.defocus_angle,
            defocus_disk_u,
            defocus_disk_v
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct Camera{
    image_width: u32,
    image_height: u32,
    max_value: u8,
    _aspect_ratio: f64,
    center: vec3,
    pixel_delta_u: vec3,
    pixel_delta_v: vec3,
    pixel00_loc: vec3,
    samples_per_pixel: u32,
    pixel_samples_scale: f64,
    max_depth: u32,
    vfov: f64, // 视场角
    u: vec3, // 相机右, normalized
    v: vec3, // 相机上, normalized
    w: vec3, // view_direction.neg(), normalized
    defocus_angle: f64,
    defocus_disk_u: vec3, // Defocus disk horizontal radius
    defocus_disk_v: vec3, // Defocus disk vertical radius
}
impl Camera{
    fn get_ray(&self, x: i32, y: i32) -> Ray{
        let pixel_center = self.pixel00_loc
            + (x as f64 * self.pixel_delta_u)
            + (y as f64 * self.pixel_delta_v);
        let pixel_sample = pixel_center + self.pixel_sample_square();
        let ray_origin = if self.defocus_angle <= 0. {
            self.center
        } else {
            self.defocus_disk_sample()
        };
        let ray_direction = pixel_sample - ray_origin;
        let mut rng = rand::thread_rng();
        let ray_time = rng.gen();

        Ray{
            origin: ray_origin,
            direction: ray_direction,
            time: ray_time
        }
    }
    #[inline]
    fn defocus_disk_sample(&self) -> vec3 {
        // Returns a random point in the camera defocus disk.
        let vec = random_in_unit_disk();
        self.center
            + (vec.x * self.defocus_disk_u)
            + (vec.y * self.defocus_disk_v)
    }
    fn pixel_sample_square(&self) -> vec3{
        let mut rng = rand::thread_rng();
        let px = -0.5 + rng.gen::<f64>();
        let py = -0.5 + rng.gen::<f64>();
        (px * self.pixel_delta_u) + (py * self.pixel_delta_v)
    }
    fn render<T>(&self, world: &T) -> Result<(), Box<dyn std::error::Error>>
    where
        T: Hittable + Sync
    {
        let pixels = (0..self.image_height)
        .cartesian_product(0..self.image_width)
        .collect::<Vec<(u32, u32)>>()
        .into_par_iter()
        .progress_count(
            self.image_height as u64 * self.image_width as u64
        )
        .map(|(y, x)|{
            let multisampled_pixel_color = (0..self.samples_per_pixel)
                .into_iter()
                .map(|_|{
                    self.get_ray(x as i32, y as i32).color(self.max_depth, world)
                })
                .sum::<vec3>() * self.pixel_samples_scale;

            let pixel_color = vec3{
                x: linear_to_gamma(multisampled_pixel_color.x),
                y: linear_to_gamma(multisampled_pixel_color.y),
                z: linear_to_gamma(multisampled_pixel_color.z),
            } * 255.;
            format!{
                "{} {} {}",
                pixel_color.x, pixel_color.y, pixel_color.z
            }
        })
        .collect::<Vec<String>>()
        .join("\n");

        fs::write("output.ppm", format!(
"P3
{0} {1}
{2}
{pixels}
", self.image_width, self.image_height, self.max_value
        ))?;
        Ok(())
    } 
}

#[inline]
fn random_in_unit_sphere() -> vec3 {
    let mut rng = rand::thread_rng();
    loop {
        let vec = vec3::new(
            rng.gen_range(-1.0..1.),
            rng.gen_range(-1.0..1.),
            rng.gen_range(-1.0..1.),
        );
        if vec.length_squared() < 1. {
            break vec;
        }
    }
}

#[inline]
fn random_unit_vector() -> vec3 {
    return random_in_unit_sphere().normalize();
}

#[inline]
fn random_in_unit_disk() -> vec3 {
    let mut rng = rand::thread_rng();
    loop {
        let vec = vec3::new(
            rng.gen_range(-1.0..1.),
            rng.gen_range(-1.0..1.),
            0.
        );
        if vec.length_squared() < 1. {
            break vec;
        }
    }
}

#[inline]
fn linear_to_gamma(scalar: f64) -> f64 {
    scalar.sqrt()
}

#[inline]
fn reflect(v: vec3, n: vec3) -> vec3{
    v - 2. * v.dot(n) * n
}

#[inline]
fn refract(uv: vec3, n:vec3, etai_over_etat: f64) -> vec3 {
    let cos_theta = uv.neg().dot(n).min(1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = (1.0 - r_out_perp.length_squared()).abs().sqrt().neg() * n;
    r_out_perp + r_out_parallel
}

#[inline]
fn reflectance(cosine: f64, refraction_index: f64) -> f64 {
    // use Schlick's approximation
    let mut r0 = (1. - refraction_index) / (1. + refraction_index);
    r0 = r0 * r0;
    r0 + (1. - r0) * (1. - cosine).powf(5.)
}

#[deprecated]
#[allow(dead_code)]
fn random_on_hemisphere(normal: &vec3) -> vec3 {
    let on_unit_sphere = random_unit_vector();
    if on_unit_sphere.dot(*normal) > 0. {
        on_unit_sphere
    }
    else{
        -on_unit_sphere
    }
}
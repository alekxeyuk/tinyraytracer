from math import sqrt, tan
import numpy as np
import sys 
    
class Light:
    position = np.array([None] * 3)
    intensity = 0.0
    def __init__(self, p, i):
        self.position = p
        self.intensity = i

class Material:
    refractive_index = 0.0
    albedo = np.array([None] * 4)
    diffuse_color = np.array([None] * 3)
    specular_exponent = 0.0
    def __init__(self,r = 1, a = np.array([1, 0, 0, 0]), color = np.array([None] * 3), spec = 0.0):
        self.refractive_index = r
        self.albedo = a
        self.diffuse_color = color
        self.specular_exponent = spec

class Sphere:
    center = np.array([None] * 3)
    radius = 0.0
    material = Material()
    
    def __init__(self, c, r, m):
        self.center = c
        self.radius = r
        self.material = m

    def ray_intersect(self, orig, dir):
        L = self.center - orig
        tca = sum(L*dir)
        d2 = (sum(L*L) - tca*tca)
        if (d2 > self.radius * self.radius): return False, 0
        thc = sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        if (t0 < 0): t0 = t1
        if (t0 < 0): return False, 0
        return True, t0

def reflect(I, N):
    return I - N * 2.0 * sum(I * N)

def refract(I, N, refractive_index): # Snell's law
    cosi = -max(-1.0, min(1.0, sum(I * N))) #float
    etai, etat = 1, refractive_index       #float
    n = N                                   #Vec3f
    if cosi < 0: # if the ray is inside the object, swap the indices and invert the normal to get the correct result
        cosi = -cosi
        (etai, etat) = (etat, etai)         #swap
        n = -N
    eta = etai / etat                       #float
    k = 1 - eta * eta * (1 - cosi * cosi)   #float
    answer = (np.array([0,0,0]), n) if k < 0 else (I * eta + n * (eta * cosi - sqrt(k)), n)
    return answer
    
def scene_intersect(orig, dir, spheres):
    material = Material(color = np.array([0.2, 0.7, 0.8]))
    N, hit = np.array([None] * 3), np.array([None] * 3)
    spheres_dist = sys.float_info.max
    for i in range(len(spheres)):
        intersect, dist_i = spheres[i].ray_intersect(orig, dir)
        if ( intersect and dist_i < spheres_dist):
            spheres_dist = dist_i
            hit = orig + dir*dist_i
            N = (hit - spheres[i].center)/np.linalg.norm((hit - spheres[i].center))
            material = spheres[i].material
    checkerboard_dist = sys.float_info.max
    if (abs(dir[1]) > 1e-3):
        d = -(orig[1]+4)/dir[1] # the checkerboard plane has equation y = -4
        pt = orig + dir * d
        if (d > 0 and abs(pt[0]) < 10 and pt[2] < -10 and pt[2] > -30 and d < spheres_dist):
            checkerboard_dist = d
            hit = pt
            N = np.array([0,1,0])
            fifa = (int(0.5*hit[0]+1000) + int(0.5*hit[2])) & 1
            material.diffuse_color = np.array([1,1,1]) if fifa else np.array([1, 0.7, 0.3])
            material.diffuse_color = material.diffuse_color * 0.3
    return min(spheres_dist, checkerboard_dist) < 1000, material, N, hit

def cast_ray(orig, dir, spheres, lights, depth=0):
    intersect, material, N, point = scene_intersect(orig, dir, spheres)
    if not intersect or depth > 4:
        return material.diffuse_color #background color
    
    ray = reflect(dir, N)
    reflect_dir = ray/np.linalg.norm(ray)                           #VECTOR
    ray, N = refract(dir, N, material.refractive_index)
    refract_dir = ray/np.linalg.norm(ray) #VECTOR
    reflect_orig = sum(point - N) * 1e-3 if sum(reflect_dir * N) < 0 else point + N * 1e-3  #VECTOR  offset the original point to avoid occlusion by the object itself
    refract_orig = point - N * 1e-3 if sum(refract_dir * N) < 0 else point + N * 1e-3  #VECTOR
    reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, lights, depth + 1)         #VECTOR
    refract_color = cast_ray(refract_orig, refract_dir, spheres, lights, depth + 1)         #VECTOR

    diffuse_light_intensity = 0
    specular_light_intensity = 0
    for i in range(len(lights)):
        light_dir = (lights[i].position - point)/np.linalg.norm(lights[i].position - point)
        light_distance = np.linalg.norm(lights[i].position - point)

        shadow_orig = point - N * 1e-3 if sum(light_dir * N) < 0 else point + N * 1e-3 # checking if the point lies in the shadow of the lights[i]
        
        intersect, tmpmaterial, shadow_N, shadow_pt = scene_intersect(shadow_orig, light_dir, spheres)
        if (intersect and np.linalg.norm(shadow_pt - shadow_orig) < light_distance):
            continue
        diffuse_light_intensity += lights[i].intensity * max(0.0, sum(light_dir * N))
        specular_light_intensity += pow(max(0.0, sum(-reflect(-light_dir, N) * dir)), material.specular_exponent) * lights[i].intensity
    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + np.array([1.0, 1.0, 1.0]) * specular_light_intensity * material.albedo[1] + reflect_color * material.albedo[2] + refract_color * material.albedo[3]
    
def render(spheres, lights, width, height):
    fov = int(np.pi/2.0)
    framebuffer = [None] * (width*height)

    for j in range(height):
        for i in range(width):
            x =  (2*(i + 0.5)/float(width)  - 1)*tan(fov/2.)*width/float(height)    #float
            y = -(2*(j + 0.5)/float(height) - 1)*tan(fov/2.)                        #float
            dir = np.array([x, y, -1])/np.linalg.norm(np.array([x, y, -1]))         #SAME
            framebuffer[i+j*width] = cast_ray(np.array([0,0,0]), dir, spheres, lights)
        
    with open('out.ppm', 'w', encoding="utf-8") as ofs:
        ofs.write(f"P3\n{width} {height}\n255\n")
        for i in range(height*width):
            c = framebuffer[i]
            maximum = max(c[0], max(c[1], c[2]))
            if (maximum > 1): framebuffer[i] = c * (1.0 / maximum)
            for j in range(3):
                char = int((255 * max(0, min(1, framebuffer[i][j]))))
                ofs.write(str(char)+' ')
            ofs.write('\n')

if __name__ == "__main__":
    
    ivory =         Material(1.0, np.array([0.6,  0.3, 0.1, 0.0]), np.array([0.4, 0.4, 0.3]), 50.0)
    red_rubber =    Material(1.0, np.array([0.9,  0.1, 0.0, 0.0]), np.array([0.3, 0.1, 0.1]), 10.0)
    mirror =        Material(1.0, np.array([0.0, 10.0, 0.8, 0.0]), np.array([1.0, 1.0, 1.0]), 1425.0)
    glass =         Material(1.5, np.array([0.0,  0.5, 0.1, 0.8]), np.array([0.6, 0.7, 0.8]), 125.0) 
    
    spheres = []
    spheres.append(Sphere(np.array([  -3,    0, -16]), 2, ivory))
    spheres.append(Sphere(np.array([-1.0, -1.5, -12]), 2, glass))
    spheres.append(Sphere(np.array([ 1.5, -0.5, -18]), 3, red_rubber))
    spheres.append(Sphere(np.array([   7,    5, -18]), 4, mirror))

    lights = []
    lights.append(Light(np.array([-20, 20,  20]), 1.5))
    lights.append(Light(np.array([ 30, 50, -25]), 1.8))
    lights.append(Light(np.array([ 30, 20,  30]), 1.7))
    if len(sys.argv) < 3: print("Not enough arguments");exit()
    render(spheres, lights, int(sys.argv[1]), int(sys.argv[2]))
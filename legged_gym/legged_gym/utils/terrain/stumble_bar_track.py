import numpy as np
from legged_gym.utils import trimesh

from isaacgym import terrain_utils, gymapi

class TerrainStumbleBarTrack:
    def __init__(self, cfg, num_envs):
        self.cfg = cfg
        self.env_width = cfg.terrain_width
        self.env_length = cfg.terrain_length
        print("env_width:", self.env_width)
        print("env_length:", self.env_length)
        print("cfg.num_rows", cfg.num_rows)
        print("cfg.num_cols", cfg.num_cols)
        self.xSize = cfg.terrain_width * cfg.num_cols # int(cfg.horizontal_scale * cfg.tot_cols)
        self.ySize = cfg.terrain_length * cfg.num_rows # int(cfg.horizontal_scale * cfg.tot_rows)
        print("self.xSize:", self.xSize)
        print("self.ySize:", self.ySize)
        self.vertical_scale = cfg.vertical_scale
        self.horizontal_scale = cfg.horizontal_scale
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        print("self.length_per_env_pixels:", self.length_per_env_pixels)
        print("self.width_per_env_pixels:", self.width_per_env_pixels)
        self.tot_rows = int(self.ySize / cfg.horizontal_scale)
        self.tot_cols = int(self.xSize / cfg.horizontal_scale)
        self.max_zScale=cfg.max_zScale
        self.min_zScale=cfg.min_zScale
        self.max_num_pit=cfg.max_num_pit
        self.min_num_pit=cfg.min_num_pit
        self.max_num_pole=cfg.max_num_pole
        self.min_num_pole=cfg.min_num_pole
        self.max_num_bar=cfg.max_num_bar
        self.min_num_bar=cfg.min_num_bar
        self.pit_length=cfg.pit_length
        self.pit_width=cfg.pit_width
        self.platform_size=cfg.platform_size
        self.border_width=cfg.border_width
        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))
        self.heightsamples = np.zeros((self.tot_cols , self.tot_rows), dtype=np.int16)
        print("Terrain heightsamples shape: ", self.heightsamples.shape)
        self.heightsamples_perlin= self.generate_fractal_noise_2d(self.xSize, self.ySize, self.tot_cols, self.tot_rows, **cfg.TerrainPerlin_kwargs)
        self.heightsamples_perlin = (self.heightsamples_perlin * (1 / cfg.vertical_scale)).astype(np.int16)
        self.curiculum()

        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.heightsamples,
                                                                                    cfg.horizontal_scale,
                                                                                    cfg.vertical_scale,
                                                                                    cfg.slope_treshold)
        self.num_bar_track = cfg.num_bars
        self.min_bar_width = cfg.min_bar_width
        self.max_bar_width = cfg.max_bar_width
        self.min_bar_height = cfg.min_bar_height
        self.max_bar_height = cfg.max_bar_height
        
    @staticmethod
    def generate_perlin_noise_2d(shape, res):
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1) * 0.5 + 0.5
    
    @staticmethod
    def generate_fractal_noise_2d(xSize=20, ySize=20, xSamples=1600, ySamples=1600, \
        frequency=10, fractalOctaves=2, fractalLacunarity = 2.0, fractalGain=0.25, zScale = 0.23):
        xScale = int(frequency * xSize)
        yScale = int(frequency * ySize)
        amplitude = 1
        shape = (xSamples, ySamples)
        noise = np.zeros(shape)
        for _ in range(fractalOctaves):
            noise += amplitude * TerrainStumbleBarTrack.generate_perlin_noise_2d((xSamples, ySamples), (xScale, yScale)) * zScale
            amplitude *= fractalGain
            xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

        return noise
    
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = terrain_utils.SubTerrain("terrain",
                            width=self.width_per_env_pixels,
                            length=self.length_per_env_pixels,
                            vertical_scale=self.vertical_scale,
                            horizontal_scale=self.horizontal_scale)
                zScale=self.min_zScale
                num_pit=self.cfg.num_pits
                num_pole=self.cfg.num_poles
                terrain = self.make_terrain_stumble(num_pit, num_pole, zScale=0)
                self.add_terrain_to_map(terrain, i, j) 

    def make_terrain_stumble(self, num_pit, num_pole, zScale=0.05):
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
        self.stumble_terrain_border(terrain, num_pit, num_pole, zScale)
        return terrain
    
    def make_terrain_perlin(self,zScale):
        heightsamples = np.zeros(( self.length_per_env_pixels, self.width_per_env_pixels), dtype=np.int16)
        print("heightsamples.shape:", heightsamples.shape)
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
        heightsamples=self.generate_fractal_noise_2d(self.env_width, self.env_length, self.width_per_env_pixels, self.length_per_env_pixels, frequency=5, zScale=zScale)
        heightsamples = (heightsamples * (1 / self.cfg.vertical_scale)).astype(np.int16)

        terrain.height_field_raw=heightsamples
        return terrain

    def stumble_terrain_border(self, terrain, num_pit, num_pole, zScale=0.05):
        (i, j) = terrain.height_field_raw.shape
        if not self.cfg.single_terrain:
            track_length = self.env_length/3 / (num_pit+2)
        else:
            track_length = self.env_length/(num_pit+3)

        for k in range(num_pit):
            width_ = round((self.cfg.min_pit_width + k/(num_pit-1)*(self.cfg.max_pit_width - self.cfg.min_pit_width))/terrain.horizontal_scale)
            height_ = round(1.0/terrain.vertical_scale)
            if not self.cfg.single_terrain:
                center_y = round((2*self.env_length/3+ (k+1)*track_length)/terrain.horizontal_scale)
            else:
                center_y = round((k+2)*track_length/terrain.horizontal_scale)

            terrain.height_field_raw[:, round(center_y-width_/2):round(center_y+width_/2)] = -height_

        for _ in range(num_pole):
            width_ = round(np.random.uniform(0.05, 0.1)/terrain.horizontal_scale)
            height_ = round(np.random.uniform(0.5, 1.0)/terrain.vertical_scale)

            start_i = round(np.random.uniform(0, i))
            if not self.cfg.single_terrain:
                start_j = round(np.random.uniform(j/3, 2*j/3))
            else:
                start_j = round(np.random.uniform(j/5, j))
            terrain.height_field_raw[start_i:start_i+width_, start_j:start_j+width_] = height_

        return terrain
        
    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_y = i * self.length_per_env_pixels
        end_y = (i + 1) * self.length_per_env_pixels
        start_x = j * self.width_per_env_pixels
        end_x = (j + 1) * self.width_per_env_pixels
        print("start_y", start_y)
        print("end_y", end_y)
        print("start_x", start_x)
        print("end_x", end_x)
        print("self.heightsamples.shape:", self.heightsamples.shape)
        print("self.heightsamples[start_x: end_x, start_y: end_y].shape", self.heightsamples[start_x: end_x, start_y: end_y].shape)
        self.heightsamples[start_x: end_x, start_y: end_y] = terrain.height_field_raw

        env_origin_y = (i + 0.5) * self.env_length
        env_origin_x = (j + 0.5) * self.env_width
        x0 = int(self.env_width/2 / terrain.horizontal_scale)
        y0 = int(self.env_length/2 / terrain.horizontal_scale)
        env_origin_z = terrain.height_field_raw[x0, y0]*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        
    def _create_trimesh(self, gym, sim, device= "cpu"):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        trimesh=(self.vertices, self.triangles)
        trimesh_origin=np.zeros(3,)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = trimesh[0].shape[0]
        tm_params.nb_triangles = trimesh[1].shape[0]
        tm_params.transform.p.x = trimesh_origin[0]
        tm_params.transform.p.y = trimesh_origin[1]
        tm_params.transform.p.z = 0.
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        gym.add_triangle_mesh(
            sim,
            trimesh[0].flatten(order= "C"),
            trimesh[1].flatten(order= "C"),
            tm_params,
        )
    
    def add_trimesh_to_sim(self, gym, sim, trimesh):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = trimesh[0].shape[0]
        tm_params.nb_triangles = trimesh[1].shape[0]
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        gym.add_triangle_mesh(
            sim,
            trimesh[0].flatten(order= "C"),
            trimesh[1].flatten(order= "C"),
            tm_params,
        )

    def add_obstacle_to_sim(self, gym, sim, device="cpu"):
        if not self.cfg.single_terrain:
            track_length = self.env_length/3 / (self.num_bar_track+2)
        else:
            track_length = self.env_length/(self.num_bar_track+2)
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                for k in range(self.num_bar_track):
                    length = self.env_width
                    width = self.min_bar_width + k/(self.num_bar_track-1)*(self.max_bar_width - self.min_bar_width)
                    height_ = self.min_bar_height + k/(self.num_bar_track-1)*(self.max_bar_height - self.min_bar_height)

                    width_=length
                    length_=width
                    width_save = width

                    size=np.array([width_, length_, width_save], dtype= np.float32)
                    center_y = -self.env_length/2 + (k+2)*track_length
                    center=np.array([0, center_y, height_], dtype= np.float32)+self.env_origins[i, j]
                    center=np.array(center, dtype=np.float32)

                    obstacle_trimesh = trimesh.box_trimesh(size, center)
                    self.add_trimesh_to_sim(gym, sim, obstacle_trimesh)

    def add_terrain_to_sim(self, gym, sim, device= "cpu"):
        """ deploy the terrain mesh to the simulator
        """
        self._create_trimesh(gym, sim, device)
        self.add_obstacle_to_sim(gym, sim, device)
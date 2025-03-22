import importlib

terrain_registry = dict(
    TerrainPerlin= "legged_gym.utils.terrain.perlin:TerrainPerlin",
    TerrainStumbleMix= "legged_gym.utils.terrain.stumble_mix:TerrainStumbleMix",
    TerrainStumbleBarTrack = "legged_gym.utils.terrain.stumble_bar_track:TerrainStumbleBarTrack",
)

def get_terrain_cls(terrain_cls):
    entry_point = terrain_registry[terrain_cls]
    module, class_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)

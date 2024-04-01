from new_methods.new_methods import My_NuScenes

nusc = My_NuScenes(version='v1.0-mini', dataroot='/data/datasets/nuscenes', verbose=True)
a,b,c = nusc.check_depth(nusc.sample[72]['token'])
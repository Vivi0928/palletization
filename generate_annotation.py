from robosuite.utils.transform_utils import convert_quat
from env import BoxPlanning
import numpy as np
from multiprocessing import Pool, Manager
from functools import partial
from torch.utils.data import Dataset
from task_config import TaskConfig

def quaternion_angle_difference(q1, q2):
    # Calculate the angular difference between two quaternions
    dot_product = np.dot(q1, q2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_diff = 2 * np.arccos(dot_product)
    return angle_diff

# Class for gennerate annotation
class GenAnno(Dataset):
    data_list = [] # Data list that all generators shared

    def __init__(self):
        self.env = BoxPlanning(device='cpu')
        self.pallet_size_discrete = self.env.pallet_size_discrete
        self.max_pallet_height = self.env.max_pallet_height
        self.env.reset()
        
    def recover_pallet(self, pallet_config:dict, to_place_id, to_place_quat, size_after_rotate):
        to_place_box_obj = self.env.id_to_box_obj[to_place_id]
        z_pos = self.env.table_offset[2] + size_after_rotate[2] * self.env.bin_size /2
        pos = np.array([0, 0, z_pos])
        self.env.place_box(to_place_box_obj, pos, to_place_quat)

        for boxid in pallet_config.keys():
            box_obj = self.env.id_to_box_obj[boxid]
            pose = pallet_config[boxid]
            self.env.place_box(box_obj, pose[:3], pose[3:])
        self.env.sim_forward(1)

    def is_place_stable(self, density_map, height_map, box_id, target_quat, size_after_rotate, x, y, density):
        size_after_rotate = size_after_rotate.astype(int)
        length, width, height = size_after_rotate[0], size_after_rotate[1], size_after_rotate[2]
        area_density = density_map[x:x+length, y:y+width]
        max_density_value = np.max(area_density)

        if max_density_value == 0:
            return 1 
        
        area = height_map[x:x+length, y:y+width]
        vals, counts = np.unique(area, return_counts=True)
        index = np.argmax(vals)

        if vals[index] + height > self.max_pallet_height:
            return 0
        
        z = area.max()
        target_x = self.env.pallet_position[0] - self.env.pallet_size[0]/2 + x * self.env.bin_size + size_after_rotate[0] * self.env.bin_size /2
        target_y = self.env.pallet_position[1] - self.env.pallet_size[1]/2 + y * self.env.bin_size + size_after_rotate[1] * self.env.bin_size /2
        target_z = self.env.pallet_position[2] + self.env.pallet_size[2]/2 + z * self.env.bin_size + size_after_rotate[2] * self.env.bin_size /2
        target_pos = np.array([target_x, target_y, target_z])
        box_obj = self.env.id_to_box_obj[box_id]
        self.env.sim.data.set_joint_qpos(box_obj.joints[0], np.concatenate([target_pos, convert_quat(target_quat, to="wxyz")]))
        self.env.sim.data.set_joint_qvel(box_obj.joints[0], np.zeros(6))

        self.env.sim_forward(50)

        pos_diff, angle_diff = self.calculate_diff(box_id, target_pos, target_quat)
        is_stable = pos_diff < 0.01 and angle_diff < 0.01

        return is_stable
    
    def generate_annotation(self, idx):
        # Gennerate annotation at idx in GenAnno.data_list
        record_data = GenAnno.data_list[idx]
        pallet_config = record_data["pallet_config"]
        pallet_obs_density = record_data["pallet_obs_density"]
        to_place_id = record_data["to_place_id"]
        to_place_quat = record_data["to_place_quat"]
        size_after_rotate = record_data["size_after_rotate"]
        to_place_density = record_data["to_place_density"]
    
        image = self.get_image(pallet_obs_density, size_after_rotate, to_place_density)
        
        feasible_map = np.zeros((int(self.pallet_size_discrete[0]), int(self.pallet_size_discrete[1])))
        self.env.init_box_pose()
        density_map, height_map = self.get_map(pallet_obs_density)

        for x in range(int(feasible_map.shape[0]-size_after_rotate[0]+1)):
            for y in range(int(feasible_map.shape[1]-size_after_rotate[1]+1)):
                self.env.init_box_pose()
                self.recover_pallet(pallet_config, to_place_id, to_place_quat, size_after_rotate)
                feasible_map[x, y] = self.is_place_stable(density_map, height_map, to_place_id, to_place_quat, size_after_rotate, x, y, to_place_density)

        return (image, feasible_map)

    def calculate_diff(self, box_id, target_pos, target_quat):
        # Calculate difference between two pose
        box_cur_pose = self.env.get_box_pose(box_id)
        angle_diff = quaternion_angle_difference(box_cur_pose[3:], target_quat) # rad
        pos_diff = np.linalg.norm(box_cur_pose[:3]-target_pos)
        return pos_diff, angle_diff

    def get_map(self, pallet_obs_density):
        # Get density and height map
        reversed_density = pallet_obs_density[:, :, ::-1]
        non_zero_indices = np.argmax(reversed_density > 0, axis=2)
        mask = np.any(reversed_density > 0, axis=2)
        density_map = np.zeros_like(non_zero_indices, dtype=float)
        density_map[mask] = reversed_density[mask, non_zero_indices[mask]]

        height_map = np.max((pallet_obs_density > 0) * np.arange(1, pallet_obs_density.shape[2] + 1), axis=2)

        return density_map, height_map

    def get_image(self, obs_density, box_size, box_density):
        lengths = np.ones_like(obs_density) * box_size[0]
        widths = np.ones_like(obs_density) * box_size[1]
        heights = np.ones_like(obs_density) * box_size[2]
        box_density = np.ones_like(obs_density) * box_density
        image = np.stack([obs_density, lengths, widths, heights, box_density], axis=0).astype(np.float32)
        
        return image
    
# def get_generators(pallet_size_discrete, max_pallet_height, n_gen):
#     # Init n generators
#     return [GenAnno(pallet_size_discrete, max_pallet_height) for _ in range(n_gen)]

def generate_annotations(idx, shared_list):
    # Assigns a data in the data list to a gennerator
    generator = generators[idx % TaskConfig.train.n_gen]
    result = generator.generate_annotation(idx)
    shared_list.append(result)

if TaskConfig.train.mask_type == 'nn':
    generators = [GenAnno() for _ in range(TaskConfig.train.n_gen)]
else:
    generators = None

def imap_gen():
    # Generate annotation in a multithreaded way
    with Manager() as manager:
        shared_list = manager.list()

        with Pool(TaskConfig.train.n_gen) as pool:
            partial_generate_annotations = partial(generate_annotations, shared_list=shared_list)
            
            idx_range = range(0, len(GenAnno.data_list))
            list(pool.imap_unordered(partial_generate_annotations, idx_range))
            pool.close()
            pool.join()

        return list(shared_list)

from robosuite.models.tasks import ManipulationTask
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.utils.transform_utils import convert_quat
from robosuite import load_controller_config
import robosuite.utils.transform_utils as T
from scipy.special import softmax
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import imageio, sys, copy, torch, random, argparse

from task_config import TaskConfig
from helpers.utils import Heuri

sys.path.append("Unet3D")
import Unet3D.model
from Unet3D.utils import load_checkpoint


SIM_TIMESTEP = 0.002

class BoxPlanning(SingleArmEnv):
    n_frame = 0
    def __init__(self, 
                 save_video_path=None ,
                 mask_type=None, 
                 nn_mask_path=None, 
                 device=torch.device("cuda:0"),
                 init_box_pose_path=None):
        # Table config
        self.table_full_size = TaskConfig.table.full_size
        self.table_friction = TaskConfig.table.friction
        self.table_offset = np.array(TaskConfig.table.offset)

        # Pallet config
        self.pallet_size = TaskConfig.pallet.size
        self.pallet_position = self.table_offset + TaskConfig.pallet.relative_table_displacement

        # Gripper Initialization
        controller_configs = load_controller_config(custom_fpath="./helpers/controller.json")

        # Save video config
        control_freq = 20
        if save_video_path is not None:
            self.writer = imageio.get_writer(save_video_path, fps=control_freq)
            self.save_video = True
        else:
            self.save_video = False

        # Task config
        self.N_visible_boxes = TaskConfig.buffer_size
        self.max_pallet_height = TaskConfig.pallet.max_pallet_height
        self.bin_size = TaskConfig.bin_size # use 1cm as bin size to discrete the space
        self.pallet_size_discrete = (np.array(self.pallet_size)[:2] / self.bin_size).astype(int)
        self.n_properties = TaskConfig.box.n_properties
        self.n_box_types = TaskConfig.box.n_type
        self.stable_thres = 0.02
        self.random_generator = None
        self.init_box_pose_path = init_box_pose_path
        
        # Action space mask
        self.mask_type = mask_type
        if self.mask_type == 'heuri':
            self.heuri = Heuri(pallet_size_discrete=self.pallet_size_discrete.astype(int), max_pallet_height=self.max_pallet_height)
        elif self.mask_type == 'nn':
            self.heuri = Heuri(pallet_size_discrete=self.pallet_size_discrete.astype(int), max_pallet_height=self.max_pallet_height)
            self.device = device
            self.nn_mask = Unet3D.model.UNet3D(5, 1, f_maps=8, num_levels=3, final_sigmoid=True).to(self.device)
            self.mean = torch.tensor([0.21183236, 6.6072526, 5.9993353, 5.5127134, 2.9000595], device=self.device, dtype=torch.float32)
            self.std = torch.tensor([9.9945396e-01, 9.1979831e-01, 6.6459063e-04, 8.5840923e-01, 2.4248180e+00], device=self.device, dtype=torch.float32)
            if nn_mask_path is not None:
                load_checkpoint(nn_mask_path, self.nn_mask)
        
        super().__init__(
            robots=["Panda"],
            controller_configs=controller_configs,
            initialization_noise=None,
            horizon=100, 
            has_renderer=False,
            has_offscreen_renderer=self.save_video,
            use_camera_obs=self.save_video,
            control_freq = control_freq, 
            ignore_done=True,
        )

    def creat_box(self, box_type, box_name):
        box = BoxObject(
            name=box_name,
            size=TaskConfig.box.type_dict[box_type]["size"],
            material=TaskConfig.box.type_dict[box_type]["material"],
            friction=TaskConfig.box.type_dict[box_type]["friction"],
            density=TaskConfig.box.type_dict[box_type]["density"],
            solref=[0.02, TaskConfig.box.type_dict[box_type]["softness"]],
            solimp=[0.9, 0.95, 0.001]
        )
        return box
    
    def _load_model(self):
        super()._load_model()
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        
        # Load boxes
        if self.init_box_pose_path is not None:
            self.load_box_record_pose()
        else:
            self.load_box_random_pose()
        
        # Load pallet
        self.pallet = BoxObject(name="pallet", size=np.array(self.pallet_size)/2, material=TaskConfig.pallet.material)
        self.pallet.get_obj().set('pos', array_to_string(self.pallet_position))
        
        # Put together
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.pallet] + [item for sublist in self.box_obj_list for item in sublist],
        )
    
    def load_box_record_pose(self):
        self.box_init_pose = np.load(self.init_box_pose_path).tolist()
        self.total_box_number = 0
        self.box_obj_list = [ [] for _ in range(self.n_box_types)]
        for i in range(self.n_box_types):
            box_type = i+1
            for j in range(TaskConfig.box.type_dict[box_type]["count"]):
                box_name = f"{box_type}_{j}"
                box_obj = self.creat_box(box_type, box_name)
                init_pose = np.array(self.box_init_pose[self.total_box_number])
                box_obj.get_obj().set('pos', array_to_string(init_pose[:3]))
                box_obj.get_obj().set('quat', array_to_string(T.convert_quat(init_pose[3:], to="wxyz")))
                self.box_obj_list[i].append(box_obj)

                self.total_box_number += 1

    def load_box_random_pose(self):
        self.total_box_number = 0
        self.box_obj_list = [ [] for _ in range(self.n_box_types)]
        for i in range(self.n_box_types):
            box_type = i+1
            for j in range(TaskConfig.box.type_dict[box_type]["count"]):
                box_name = f"{box_type}_{j}"
                box_obj = self.creat_box(box_type, box_name)
                init_x = random.uniform(-0.3, 0.3)
                init_y = random.uniform(-0.8, -0.3)
                init_z = random.uniform(0, 0.05) + (self.n_box_types - 1 - i) * 0.05
                box_obj.get_obj().set('pos', array_to_string(self.table_offset - box_obj.bottom_offset + np.array([init_x, init_y, init_z])))
                self.box_obj_list[i].append(box_obj)

                self.total_box_number += 1
    
    def choose_index(self, action):
        sample_logits = action[:self.N_visible_boxes]
        pick_likelihood = softmax(sample_logits)
        # sample_index = np.random.choice(self.N_visible_boxes, p=pick_likelihood)
        sample_index = np.argmax(pick_likelihood)

        # sample_index = np.random.choice(self.N_visible_boxes)
        return sample_index

    def get_orientation(self, action):
        sample_ori_logits = action[self.N_visible_boxes: self.N_visible_boxes+6]
        sample_ori = np.argmax(sample_ori_logits)
        sample_ori = 0

        orders = {
            0: np.array([0,1,2]),
            1: np.array([0,2,1]),
            2: np.array([1,0,2]),
            3: np.array([1,2,0]),
            4: np.array([2,0,1]),
            5: np.array([2,1,0]),
        }
        target_quat = self.compute_target_quat_from_order(sample_ori)
        return orders[sample_ori], target_quat
    
    def compute_target_quat_from_order(self, order):
        if order == 0:
            rotm = np.eye(3)
        elif order == 1:
            rotm = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        elif order == 2:
            rotm = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        elif order == 3:
            rotm = np.array([[0,1,0],[0,0,1],[1,0,0]])
        elif order == 4:
            rotm = np.array([[0,0,1],[1,0,0],[0,1,0]])
        elif order == 5:
            rotm = np.array([[0,0,-1],[0,1,0],[1,0,0]])

        return T.mat2quat(rotm)  # (xyzw)  

    def get_target_position(self, action, box_size_after_rotate, feasible_points=None):
        # Sample x position
        x_logits = action[self.N_visible_boxes+6: self.N_visible_boxes+6+int(self.pallet_size_discrete[0])]
        x = x_logits.argmax()

        # Sample y position
        y_logits = action[self.N_visible_boxes+6+int(self.pallet_size_discrete[0]):]
        y = y_logits.argmax()

        # Apply action mask
        if feasible_points is not None:
            x, y = self.get_nearest_xy(x, y, feasible_points)
        
        # Determine z position based on pallet situation
        place_area = self.obs["pallet_obs_density"][x: int(min(x+box_size_after_rotate[0], self.pallet_size_discrete[0])), y: int(min(y+box_size_after_rotate[1], self.pallet_size_discrete[1])), :]
        non_zero_mask = np.any(place_area > 0, axis=(0, 1))
        z = np.max(np.nonzero(non_zero_mask)) + 1 if np.any(non_zero_mask) else 0

        # Determine the box center(position in world frame)
        target_x = self.pallet_position[0] - self.pallet_size[0]/2 + x * self.bin_size + box_size_after_rotate[0] * self.bin_size /2
        target_y = self.pallet_position[1] - self.pallet_size[1]/2 + y * self.bin_size + box_size_after_rotate[1] * self.bin_size /2
        target_z = self.pallet_position[2] + self.pallet_size[2]/2 + z * self.bin_size + box_size_after_rotate[2] * self.bin_size /2
        target_pos = np.array([target_x, target_y, target_z])

        return target_pos, (x,y,z)


    def step(self, action):
        # Sample a box from action
        sample_box_index = self.choose_index(action)
        if sample_box_index >= len(self.unplaced_box_ids):
            # Unplaced box numbers is less than buffer capacity and choose no box!
            r_nongrasp = -0.1
            done = False
            return self.obs, r_nongrasp, done, {}
        
        # Get the sampled box
        sample_box_id = self.unplaced_box_ids[sample_box_index]
        sample_box = self.id_to_box_obj[sample_box_id]
        sample_box_size = (np.array(sample_box.size) * 2 / self.bin_size).astype(int)  # discretized size
        sample_box_density = self.id_to_properties[sample_box_id][3]

        # Sample orientation from action
        orientation, target_quat = self.get_orientation(action)
        box_size_after_rotate = sample_box_size[orientation]  # box size after orientation

        # Record pallet for generate feasible data
        record_data = self.record_pallet(sample_box_id, target_quat, box_size_after_rotate)

        # Feasible set
        feasible_points = self.find_feasible_positions(box_size_after_rotate, sample_box_density, method=self.mask_type)
        if feasible_points is not None and feasible_points.shape[0] == 0:
            done = True
            reward, info = self.reward_func(termination_reason=1)
            info["record_data"] = record_data
            return self.obs, reward, done, info
        
        # Sample target place on the pallet
        target_pos, (x,y,z) = self.get_target_position(action, box_size_after_rotate, feasible_points)

        # Place the box on the pallet
        self.place_box(sample_box, target_pos, target_quat)

        # Run simulation forward
        self.sim_forward(40)

        # Check stability(current box and previous boxes on the pallet)
        cur_pos = self.get_box_pose(sample_box_id)[:3]
        is_stable = self.check_stable() and np.linalg.norm(cur_pos-target_pos) < self.stable_thres # check if the placement is stable

        if is_stable:
            # Remove box from buffer and add it to pallet
            self.unplaced_box_ids.pop(sample_box_index)
            self.boxes_on_pallet_id.append(sample_box_id)
            self.boxes_on_pallet_target_pose[sample_box_id] = np.concatenate([target_pos, target_quat])

            # Update obs
            self.obs["pallet_obs_density"][x: int(min(x+box_size_after_rotate[0], self.pallet_size_discrete[0])), 
                                       y: int(min(y+box_size_after_rotate[1], self.pallet_size_discrete[1])),
                                       z: z + int(box_size_after_rotate[2])] = self.id_to_properties[sample_box_id][3]
            self.update_obs_buffer()

            done = len(self.boxes_on_pallet_id) == self.total_box_number
            termination_reason = 3 if done else 0
            reward, info = self.reward_func(termination_reason, box_size_after_rotate, (x,y,z))

        else:
            done = True
            reward, info = self.reward_func(termination_reason=2)
        
        if self.save_video:
            self.save_frame()
        
        info["record_data"] = record_data
        return self.obs, reward, done, info

    def place_box(self, box_obj, target_pos, target_quat):
        # quat in xyzw format
        self.sim.data.set_joint_qpos(box_obj.joints[0], np.concatenate([target_pos, T.convert_quat(target_quat, to="wxyz")]))
        self.sim.data.set_joint_qvel(box_obj.joints[0], np.zeros(6))

    def sim_forward_per_frame(self, n_frames):
        for i in range(n_frames):
            for j in range(int(1/(self.control_freq * SIM_TIMESTEP))):
                self.sim.forward()
                self.sim.step()
                #self._update_observables()

            if self.save_video:
                self.save_frame()

    def reset(self):
        # Call the reset method of the super class
        _ = super().reset()
        self.unplaced_box_ids = copy.copy(self.boxes_ids)
        # random.shuffle(self.unplaced_box_ids) # for every episode, randomly shuffle the order at the beginning

        self.obs = {}
        self.obs["pallet_obs_density"] = np.zeros((int(self.pallet_size_discrete[0]), int(self.pallet_size_discrete[1]), self.max_pallet_height), dtype=np.float32)
        self.update_obs_buffer()

        self.boxes_on_pallet_id = []
        self.boxes_on_pallet_target_pose = {}

        return self.obs
    
    def init_box_pose(self):
        for i in range(self.total_box_number):
            box_id = self.boxes_ids[i]
            init_pose = np.array(self.box_init_pose[i])
            box_obj = self.id_to_box_obj[box_id]
            self.place_box(box_obj, init_pose[:3], init_pose[3:])

    def reinit(self, random_generator:np.random.Generator):
        self.init_box_pose()
        self.unplaced_box_ids = copy.copy(self.boxes_ids)
        if self.random_generator is None:
            self.random_generator = random_generator

        self.random_generator.shuffle(self.unplaced_box_ids) # for every episode, randomly shuffle the order at the beginning
        random.shuffle(self.unplaced_box_ids)
        self.obs = {}
        self.obs["pallet_obs_density"] = np.zeros((int(self.pallet_size_discrete[0]), int(self.pallet_size_discrete[1]), self.max_pallet_height), dtype=np.float32)
        self.update_obs_buffer()

        self.boxes_on_pallet_id = []
        self.boxes_on_pallet_target_pose = {}

        return self.obs

    def _setup_references(self):
        # Sets up references to important components
        super()._setup_references()

        self.boxes_body_id = [[] for _ in range(self.n_box_types)]
        self.boxes_id_to_index = {}
        self.boxes_names = []
        self.boxes_ids = []
        self.id_to_box_obj = {}
        self.id_to_properties = {}
        for i in range(self.n_box_types):
            for j in range(len(self.box_obj_list[i])):
                box_id = self.sim.model.body_name2id(self.box_obj_list[i][j].root_body)
                self.boxes_names.append(self.box_obj_list[i][j].root_body[:-5])
                self.boxes_body_id[i].append(box_id)
                self.boxes_id_to_index[box_id] = [i, j]
                self.boxes_ids.append(box_id)
                self.id_to_box_obj[box_id] = self.box_obj_list[i][j]
                # Scale the property
                box_property = list(np.array(self.box_obj_list[i][j].size) * 100) + [self.box_obj_list[i][j].density / 1000]
                self.id_to_properties[box_id] = np.array(box_property, dtype=np.float32)

    def check_stable(self):
        for box_id in self.boxes_on_pallet_id:
            box_cur_position = self.get_box_pose(box_id)[:3]
            box_target_position = self.boxes_on_pallet_target_pose[box_id][:3]
            if np.linalg.norm(box_cur_position - box_target_position) > self.stable_thres:
                return False
        return True
    
    def get_box_pose(self, box_id):
        """ input box id, get box pose"""
        # import pdb; pdb.set_trace()
        box_pos = np.array(self.sim.data.body_xpos[box_id])
        box_quat = convert_quat(np.array(self.sim.data.body_xquat[box_id]), to="xyzw")
        return np.hstack((box_pos, box_quat))

    def save_frame(self):
        self._update_observables()
        video_img = self.sim.render(height=720, width=1280, camera_name="frontview")[::-1]
        self.writer.append_data(video_img)  # 添加帧到视频

    def sim_forward(self, steps):
        for _ in range(steps):
            self.sim.forward()
            self.sim.step()

    def update_obs_buffer(self):
        boxes_in_buffer = np.zeros(self.N_visible_boxes * self.n_properties, dtype=np.float32)
        for i in range(min(len(self.unplaced_box_ids), self.N_visible_boxes)):
            box_property = self.id_to_properties[self.unplaced_box_ids[i]]
            boxes_in_buffer[self.n_properties * i : self.n_properties * (i+1)] = box_property
        self.obs["buffer"] = boxes_in_buffer
        return boxes_in_buffer

    def reward_func(self, termination_reason, box_size_discrete=None, position_discrete=None):
        """
            termination_reason: 
                0 -> done is False
                1 -> no feasible point
                2 -> not stable
                3 -> success
        """
        if termination_reason == 0 or termination_reason == 3:
            r1 = self.reward_box_size(box_size_discrete)
        elif termination_reason == 1:
            r1 = 0
        elif termination_reason == 2:
            r1 = 0
        info = {"reward_box_size":r1, "termination_reason":termination_reason}
        reward =  r1
        return reward, info
    
    def reward_box_size(self, box_size_discrete):
        box_vol = np.prod(box_size_discrete)
        return box_vol / (self.pallet_size_discrete[0] * self.pallet_size_discrete[1] * self.max_pallet_height)

    def find_feasible_positions(self, box_size, box_density, method):

        if method == "heuri":
            feasible_map = self.heuri.heuri_annotation(self.obs["pallet_obs_density"], box_size, box_density)
        elif method == "nn":
            if len(self.boxes_on_pallet_id) == 0:
                feasible_map = np.zeros(self.pallet_size_discrete.astype(int))
                feasible_map[:self.pallet_size_discrete[0] - box_size[0] + 1, 
                             :self.pallet_size_discrete[1] - box_size[1] + 1] = 1
            else:
                self.nn_mask.eval()
                obs_density = torch.tensor(self.obs["pallet_obs_density"])
                lengths = torch.ones_like(obs_density) * box_size[0]
                widths = torch.ones_like(obs_density) * box_size[1]
                heights = torch.ones_like(obs_density) * box_size[2]
                box_density = torch.ones_like(obs_density) * box_density
                
                image = torch.stack([obs_density, lengths, widths, heights, box_density], axis=0).unsqueeze(0).to(self.device)
                image = (image - self.mean[:, None, None, None]) / self.std[:, None, None, None]
                feasible_map = self.nn_mask(image).cpu().squeeze()
                feasible_map = torch.where(feasible_map > 0.5, 1, 0).numpy()


        else:
            return None
        coordinates = np.argwhere(feasible_map.squeeze() == 1)
        coordinates_list = [tuple(coord) for coord in coordinates]
        return np.array(coordinates_list).astype(int)
        
    def get_nearest_xy(self, x, y, feasible_xys):
        if feasible_xys.shape[0] == 0 or [x, y] in feasible_xys.tolist():
            return x, y

        pt = np.array([x, y])[None, :]
        distances = np.linalg.norm(pt - feasible_xys, axis=1)
        min_indice = np.where(distances == distances.min())[0]
        chosen_ind = self.random_generator.choice(min_indice)
        chosen_pt = feasible_xys[chosen_ind, :]
    
        return chosen_pt[0], chosen_pt[1]
    
    def save_box_pose(self, path):
        box_pose_list = []
        for i in range(self.n_box_types):
            for j in range(len(self.box_obj_list[i])):
                box_id = self.sim.model.body_name2id(self.box_obj_list[i][j].root_body)
                box_pose_list.append(self.get_box_pose(box_id))

        np.save(path, np.array(box_pose_list))

    def record_pallet(self, sample_boxid, target_quat, size_after_rotate):
        # record the current pallet and clusters for learning feasible set
        record_data = {}
        record_data["pallet_config"] = self.boxes_on_pallet_target_pose
        record_data["pallet_obs_density"] = self.obs["pallet_obs_density"].copy()
        record_data["to_place_id"] = sample_boxid
        record_data["to_place_quat"] = target_quat.copy() # xyzw format
        record_data["size_after_rotate"] = size_after_rotate.copy()
        record_data["to_place_density"] = self.id_to_properties[sample_boxid][3]

        return record_data
    
    def reward(self, action):
        return 0

class BoxPlanningEnvWrapper(gym.Env):
    def __init__(self, save_video_path=None, mask_type=None, nn_mask_path=None, device=torch.device("cuda:0")):
        super().__init__()
        self.env = BoxPlanning(save_video_path=save_video_path, mask_type=mask_type, nn_mask_path=nn_mask_path ,device=device, init_box_pose_path="./helpers/box_init_pose.npy")
        if mask_type == 'nn':
            self.nn_mask_state_dict = self.env.nn_mask.state_dict()
        action_dim = int(self.env.N_visible_boxes + 6 + self.env.pallet_size_discrete[0] + self.env.pallet_size_discrete[1])
        action_lower = np.array([-10.] * action_dim)
        action_upper = np.array([10.] * action_dim)
        self.action_space = spaces.Box(low=action_lower, high=action_upper)
        self.observation_space = gym.spaces.Dict(
            {
                "pallet_obs_density": spaces.Box(low=0, high=10, shape=(int(self.env.pallet_size_discrete[0]),int(self.env.pallet_size_discrete[1]),self.env.max_pallet_height)),
                "buffer": spaces.Box(low=0, high=10, shape=(self.env.N_visible_boxes * self.env.n_properties,))
                }
            )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # obs = self.env.reset()
        obs = self.env.reinit(random_generator=self.np_random)
        return (obs, {})
    
    def update_nn_mask(self, nn_mask_state_dict):
        self.env.nn_mask.load_state_dict(nn_mask_state_dict)
        self.nn_mask_state_dict = copy.deepcopy(nn_mask_state_dict)
    
    # def update_mean_std(self, mean, std):
    #     self.env.mean = torch.from_numpy(copy.deepcopy(mean)).to(self.env.device)
    #     self.env.std = torch.from_numpy(copy.deepcopy(std)).to(self.env.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_box_pose', type=bool, default=False, help='Save box pose as init pose or not')
    args = parser.parse_args()

    env = BoxPlanning(save_video_path="video/init_pose.mp4", mask_type=None, nn_mask_path=None, init_box_pose_path=None)
    env.reset()
    env.sim_forward_per_frame(20)

    if args.save_box_pose is True:
        env.save_box_pose("./helpers/box_init_pose.npy")

# Physics-Aware Robotic Palletization with Online Masking Inference

<img src=".\assets\overview.png" alt="overview" style="zoom:75%;" />

![test_policy_OL_mask](.\assets\test_policy_OL_mask.gif)



## Getting Started

**Folder structure**

```
├── env.py
├── generate_annotation.py
├── helpers
│   ├── box_init_pose.npy
│   ├── controller.json
│   ├── __init__.py
│   ├── material.py
│   ├── train_mask.py
│   └── utils.py
├── __init__.py
├── logs
│   ├── heuri1
		...
│   ├── nn1
		...
│   └── nomask1
		...
├── environment.yaml
├── task_config.py
├── test.py
├── train.py
├── Unet3D
	...
└── video
```

### Installation

Create conda environment

```
conda env create -f environment.yaml
```

To test if the environment is installed properly

```
conda activate palletization
python env.py
```

It should generate a video called ```init_pose.mp4.mp4``` in the video folder.



## Train

Change the configuration in ```task_config.py```, and run:

```
python train.py
```

Note that the annotation generation process will cost lots of time. We have implemented multi-threading in code. It is best to set ```TaskConfig.train.n_gen``` in ```task_config.py``` to the number of computer CPU cores during training to speed up the annotation generation process.



Here we provide the box settings as follows:

| Dimension | Density | Rigidity | Counts |
| --------- | ------- | -------- | ------ |
| 6 × 6 × 4 | 500     | 0.5      | 10     |
| 6 × 6 × 6 | 500     | 0.5      | 10     |
| 6 × 6 × 6 | 5000    | 3        | 10     |
| 8 × 6 × 6 | 5000    | 3        | 10     |

You can modify the box settings in the `task_config`, such as changing the dimension, density, rigidity, or adding new types of boxes. However, after modifying the box settings, please regenerate the `helpers/box_init_pose.npy` file by running:

```
python env.py --save_box_pose
```

It should generate a new ```box_init_pose.npy``` and replace the old one.



## Test

We provide the checkpoints obtained from training using the ```nn mask```, the ```heuristic mask```, and the ```nomask```, and you can use them directly for testing. Run ```test.py```:

```
python test.py --device cuda:0 --mask_type nn --exp_id 1 --check_point_name model_4000000_steps
```



## Related Work

We implement the 3D UNet using the method from [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).


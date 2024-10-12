### Folder structure

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
├── palletization.yaml
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
conda env create -f palletization.yaml
```

To test if the environment is installed properly

```
conda activate palletization
python test.py --device cuda:0 --exp_id nn1 --check_point_name model_4000000_steps
```

It should generate a video called ```test_policy_nn1.mp4``` in the video folder, using ```logs/nn1/model_4000000_steps.zip```



### Train

Change the configuration in ```task_config.py```, and run:

```
python train.py
```



### Test

We provide the checkpoints obtained from training using the ```nn mask```, the ```heuristic mask```, and the ```nomask```, and you can use them directly for testing. Run ```test.py```:

```
python test.py --device cuda:0 --exp_id nn1 --check_point_name model_4000000_steps
```


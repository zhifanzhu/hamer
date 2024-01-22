# Run

e.g. to run batch inference on all P01_01 frames in GetaGrip dataset
```
CUDA_VISIBLE_DEVICES=1 python infer_epic/infer.py --vid P01_01 --step_size 5 --dump_dir data/hamer --out_folder epic_out --side_view

# If needs to visualize 
CUDA_VISIBLE_DEVICES=1 python infer_epic/infer.py --vid P01_01 --step_size 5 --dump_dir data/hamer --out_folder epic_out --side_view --save_mesh --viz
```

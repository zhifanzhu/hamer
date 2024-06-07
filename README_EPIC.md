# Run

`infer_hov1.py` and `grip_dataset_hov1.py` were specifically using hoa_cache.

e.g. to run batch inference on all P01_01 frames in GetaGrip dataset
```
CUDA_VISIBLE_DEVICES=1 python infer_epic/infer.py --vid P01_01 --step_size 5 --dump_dir data/hamer --out_folder epic_out --side_view

# If needs to visualize 
CUDA_VISIBLE_DEVICES=1 python infer_epic/infer.py --vid P01_01 --step_size 5 --dump_dir data/hamer --out_folder epic_out --side_view --save_mesh --viz
```


Use infer_general.py for known (vid, st, ed).
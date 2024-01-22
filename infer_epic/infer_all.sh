#!/bin/bash

# Correct array assignment
all_vids=(P01_07 P01_09 P01_103 P01_104 P01_107 P01_14 P02_01 P02_03 P02_07 P02_09 P02_101 P02_102 P02_107 P02_109 P02_12 P02_121 P02_122 P02_124 P02_128 P02_130 P02_132 P02_135 P03_03 P03_04 P03_05 P03_10 P03_101 P03_112 P03_113 P03_120 P03_123 P03_13 P03_14 P03_17 P03_22 P03_23 P03_24 P04_02 P04_03 P04_04 P04_05 P04_06 P04_101 P04_109 P04_11 P04_110 P04_114 P04_121 P04_21 P04_24 P04_25 P04_33 P05_01 P05_08 P06_01 P06_03 P06_05 P06_07 P06_09 P06_101 P06_102 P06_103 P06_106 P06_108 P06_110 P06_12 P06_13 P06_14 P07_08 P07_101 P07_103 P07_110 P08_09 P08_16 P08_17 P08_21 P09_02 P09_103 P09_104 P09_106 P10_04 P11_103 P11_104 P11_107 P11_16 P12_02 P12_03 P12_04 P12_101 P13_10 P14_05 P15_02 P17_01 P18_01 P18_02 P18_03 P18_06 P18_07 P20_03 P21_01 P22_01 P22_07 P22_107 P22_117 P23_02 P23_05 P24_05 P24_08 P24_09 P25_09 P25_101 P25_107 P26_02 P26_108 P26_110 P27_101 P27_105 P28_05 P28_06 P28_101 P28_103 P28_109 P28_112 P28_113 P28_13 P29_04 P30_05 P30_07 P30_101 P30_107 P30_110 P30_111 P30_112 P32_01 P35_105 P35_109 P37_101 P37_102)

# Correct iteration over array
for vid in "${all_vids[@]}"; do
    echo "Running ${vid}"
    CUDA_VISIBLE_DEVICES=1 python infer_epic/infer.py \
        --vid $vid --step_size 1 --dump_dir data/hamer --out_folder epic_out --side_view --batch_size 24 --num_workers 4
done

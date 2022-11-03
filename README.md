# CCCR
Container Character Code Recognition, a final year project continuation from bachelor degree of electrical and electronics.

## Presumed Machine Specifications
OS    : Ubuntu 20.04.5 LTS
Kernel: Linux 5.15.0-52-generic(x86_64)
CPU   : AMD Ryzen 9 5900X 12-Core Processor; 12 Cores, 24 threads
GPU   : NVIDIA GEFORE RTX 3090
RAM   : 64GB

## Usages
1. cd to PaddleOCR if not already,
2. To predict with testdata (./firebase_pull):
```
    - GPU: RUN "python3 tools/infer/predict_system.py --gpu_mem=21000 --det_limit_side_len=1080 --use_mp=True --total_process_num=8 --warmup=True"   - 53.01fps
    - CPU: RUN "python3 tools/infer/predict_system.py --use_gpu=False --det_limit_side_len=1080 --enable_mkldnn=True --cpu_threads=12 --warmup=True" -  4.05fps
```

## For OPTIMAL usage
please consider experimenting and changing the following paramters:

### Prediction Settings
1. --gpu_mem (default = 500)
2. --use_gpu (default = True)
    - To use cpu for prediction, set --use_gpu=False.
3. --cpu_threads (default = 10)
4. --use_mp (default = False)
    - For single-thread prediction, set --use_mp=False.
5. --total_process_num (default = 1)
    - Use not more than 1 when --use_mp=False.
6. --det_limit_side_len (default = 960)
    - The reduce in parameter --det_limit_side_len will increase the fps greatly at the tradeoff of accuracy.

### I/O Directory Settings
5. --image_dir (default="./firebase_pull/")
    - The image to be processed, input images directory.
6. --draw_img_save_dir (default = ./inference_result)
    - Directory to save the output images with anchor boxes drawn when --save_as_image=True. 
7. --save_log_path (default = ./log_output/)
    - Directory to save the output log. 
8. --ground_truth_path (default = ./train_data/BP_CCC_Rec/test/rec_gt_test.txt)
    - Text file for the groundtruth of the provided test/input dataset (images) for computing the overall precision. 

### INFO and DEBUG Settings
9. --save_as_image (default = False)
    - To save prediction in images, where saving as images will reduce fps drastically.
10. --show_log (default = False)
    - To show debug messages.
11. --benchmark (default = False)
    - To benchmark the machine's inference speed, memory usage, and etc.

## NOTES
To display more optional arguments, RUN 
```
    python3 tools/infer/predict_system.py [-h OR --help]"
```

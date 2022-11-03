# CCCR
Container Character Code Recognition, a final year project continuation from bachelor degree of electrical and electronics.

## Presumed Machine Specifications
| Aspects     | Specifications                                              |
|-------------|-------------------------------------------------------------|
| OS          | Ubuntu 20.04.5 LTS                                          |
| Kernel      | Linux 5.15.0-52-generic(x86_64)                             |
| CPU         | AMD Ryzen 9 5900X 12-Core Processor; 12 Cores, 24 threads   |
| GPU         | NVIDIA GEFORE RTX 3090                                      |
| RAM         | 64GB                                                        |


## Usages
1. cd to PaddleOCR if not already,
2. To predict with testdata (./firebase_pull):
```
    - Multi-thread GPU: RUN "python3 tools/infer/predict_system.py --test_run=True --gpu_mem=21000 --det_limit_side_len=1080 --use_mp=True --total_process_num=8 --warmup=True"
    - Single-thread GPU: RUN "python3 tools/infer/predict_system.py --test_run=True --gpu_mem=21000 --det_limit_side_len=1080 --warmup=True"
    - CPU: RUN "python3 tools/infer/predict_system.py --test_run=True --use_gpu=False --det_limit_side_len=1080 --enable_mkldnn=True --cpu_threads=12 --warmup=True"
```

## Tested Results
| Command           | Accuracy | FPS   |
|-------------------|----------|-------|
| Multi-thread GPU  | 91.566%  | 53.01 |
| Single-thread GPU | 92.771%  | 22.54 |
| CPU               | 92.711%  | 3.758 |

Command refers to the commands in Usage.

## For OPTIMAL usage
please consider experimenting and changing the following parameters:

| Optional Arguments                                                            | Descriptions                                                                                                    |
|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
|                                                                         **PREDICTION**                                                                                                          |
| --use_gpu            (default = True)                                         | Whether to enable GPU for prediction.                                                                           |
| --gpu_mem            (default = 500)                                          | The limit on GPU memory usage.                                                                                  |
| --cpu_threads        (default = 10)                                           | The limit of the no. of cpu threads to use in prediction.                                                       |
| --use_mp             (default = False)                                        | To enable multi-process prediction.                                                                             |
| --total_process_num  (default = 1)                                            | The no. of threads to use in prediction, DO NOT set when --use_mp=False.                                        |
| --det_limit_side_len (default = 960)                                          | The reduce in parameter --det_limit_side_len will increase the fps greatly at the tradeoff of accuracy.         |
|                                                                        **I/O DIRECTORY**                                                                                                        |
| --image_dir          (default="./test_data/")                             | The image to be processed, input images directory.                                                              |
| --draw_img_save_dir  (default = ./inference_result)                           | Directory to save the output images with anchor boxes drawn when --save_as_image=True.                          |
| --save_log_path      (default = ./log_output/)                                | Directory to save the output log.                                                                               |
| --ground_truth_path  (default = ./train_data/BP_CCC_Rec/test/rec_gt_test.txt) | Text file for the groundtruth of the provided test/input dataset (images) for computing the overall precision.  |
|                                                                        **INFO and DEBUG**                                                                                                       |
| --test_run           (default = False)                                        | Whether to provide statistical comparison between the predicted and expected output from --ground_truth_path.   |
| --save_as_image      (default = False)                                        | To save prediction in images, where saving as images will reduce fps drastically.                               |
| --show_log           (default = False)                                        | To show debug messages.                                                                                         |
| --benchmark          (default = False)                                        | To benchmark the machine's inference speed, memory usage, and etc.                                              |

## NOTES
To display more optional arguments, RUN 
```
    python3 tools/infer/predict_system.py [-h OR --help]"
```

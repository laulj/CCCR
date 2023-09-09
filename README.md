# CCCR
Container Character Code Recognition, a final year project continuation from bachelor degree of electrical and electronics.

## Training and Testing Machine Specification
| Aspects     | Specifications                                              |
|-------------|-------------------------------------------------------------|
| OS          | Ubuntu 20.04.5 LTS                                          |
| Kernel      | Linux 5.15.0-52-generic(x86_64)                             |
| CPU         | AMD Ryzen 9 5900X 12-Core Processor; 12 Cores, 24 threads   |
| GPU         | NVIDIA GEFORE RTX 3090                                      |
| RAM         | 64GB                                                        |

## Command-line Usages
1. cd to PaddleOCR if not already,

2. To predict with testdata (./test_data), RUN:
```
    - Multi-thread GPU  : python3 tools/infer/predict_system.py --gpu_mem=21000 --input_type="img" --det_limit_side_len=1280 --use_mp=True --total_process_num=8 --warmup=True --test_run=True
    - Single-thread GPU : python3 tools/infer/predict_system.py  --gpu_mem=21000 --input_type="img" --det_limit_side_len=1280 --warmup=True --test_run=True
    - CPU               : python3 tools/infer/predict_system.py --test_run=True --use_gpu=False --det_limit_side_len=1280 --enable_mkldnn=True --cpu_threads=12 --warmup=True
```
3. To predict with video, RUN:


## Tested Results
| Command           | Accuracy | FPS   |
|-------------------|----------|-------|
| Multi-thread GPU  | 89.157%  | 58.37 |
| Single-thread GPU | 90.361%  | 15.47 |
| CPU               | 90.361%  | 2.659 |

Command refers to the commands in Usage.

## For OPTIMAL usage
please consider experimenting and changing the following parameters:

| Optional Arguments                                                            | Descriptions                                                                                                      |
|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
|                                                                         **PREDICTION**                                                                                                            |
| --input_type         (default="img")                                          | The input data type, "img" or "video" or "cam".                                                                   |
| --use_gpu            (default = True)                                         | Whether to enable GPU for prediction.                                                                             |
| --gpu_mem            (default = 500)                                          | The limit on GPU memory usage.                                                                                    |
| --cpu_threads        (default = 10)                                           | The limit of the no. of cpu threads to use in prediction.                                                         |
| --use_mp             (default = False)                                        | To enable multi-process prediction.                                                                               |
| --total_process_num  (default = 1)                                            | The no. of threads to use in prediction, DO NOT set when --use_mp=False.                                          |
| --det_limit_side_len (default = 960)                                          | The reduce in parameter --det_limit_side_len will increase the fps marginally at the tradeoff of accuracy.        |
|                                                                        **I/O DIRECTORY**                                                                                                          |
| --image_dir          (default = ./test_data/)                                 | The image to be processed, input images directory when input_type = img, video filepath when input_type = "video".|
| --draw_img_save_dir  (default = ./inference_result)                           | Directory to save the output images with anchor boxes drawn when --save_as_image=True.                            |
| --cam_pos            (default = 0)                                            | The camera position if the input_type == "cam".                                                                   |
| --save_log_path      (default = ./log_output/)                                | Directory to save the output log.                                                                                 |
| --ground_truth_path  (default = ./train_data/BP_CCC_Rec/test/rec_gt_test.txt) | Text file for the groundtruth of the provided test/input dataset (images) for computing the overall precision.    |
|                                                                        **INFO and DEBUG**                                                                                                         |
| --test_run           (default = False)                                        | Whether to provide statistical comparison between the predicted and expected output from --ground_truth_path.     |
| --save_as_image      (default = False)                                        | To save prediction in images, where saving as images will reduce fps marginally.                                  |
| --show_log           (default = False)                                        | To show debug messages.                                                                                           |
| --benchmark          (default = False)                                        | To benchmark the machine's inference speed, memory usage, and etc.                                                |

To display more optional arguments, RUN 
```
    python3 tools/infer/predict_system.py [-h OR --help]"
```

The groundtruth format is as folder/image_name_*.jpg + tab + groundtruth, e.g. img/01012021050110310403_CAIU883333.jpg	CAIU 883333 0.


## Docker Usages
Usages are referring to current docker image with tag: laulj/ccc_ocr-tensorflow-cuda11.2:1.4.0
1. To mount your dataset directory from your machine into the docker container, uncomment and complete the docker-compose.yaml in your local directory:

```
    Volumes:
        - ./dataset:<your_dataset_directory>
```

where the format is : - <container_directory>:<local_directory> and ensure that <local_directory> exists and is an absolute path.

2. To update the changes, if the container is already running, RUN

'''
    sudo docker compose down
    sudo docker compose up -d
'''

3. To use the dataset mounted, set --img_dir=<container_directory>

## NOTES


## Graphical User Interface (GUI) Usage
To inference with GUI, RUN
```
    python3 interface.py
```
The API currently employed in the GUI only utilizes single-threaded processor capability, i.e. --use_mp=False.
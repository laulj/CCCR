import cv2
import re
import os
from ppocr.utils.utility import get_image_file_list
from tools.cd_val import *
from ppocr.utils.logging import get_logger
logger = get_logger()

def m_eval(res_path, groundtruth_path):
    logger.info("Post processing step begin...")
    fps_path = res_path + "processes_time.txt"
    f1  = open(res_path + "result.txt", "r")    
    f2 = open(groundtruth_path, "r")
    f3 = open(res_path + "postprocessed_result.txt", "w")
    f4 = open(fps_path, "r")
    # write header
    f3.write(repr('Filepath').center(50)  + repr('Predicted_result').rjust(23) +repr('GroundTruth').rjust(15) + repr('Confidence').rjust(15) + repr('Status').rjust(10) + repr('Validity Check').rjust(18) + '\n')

    res_lines = f1.readlines()
    groundtruth_lines = f2.readlines()
    thread_time = f4.readlines()
    
    hit_str = 0
    acc = 0.00
    longest_thread_time = 0.00
    fps = 0.00
        
    for lines in groundtruth_lines:
        full_ground_img_path, ground_truth = lines.strip().split('\t')
        temp, ground_img_path = full_ground_img_path.split('/')
        
        for Lines in res_lines:
            full_res_img_path, res_predict, res_confidence = Lines.strip().split('\t')
            res_img_path = full_res_img_path.split('/')
            res_img_path = res_img_path[2].split('.')
            search_for_path = re.search(res_img_path[0], ground_img_path)
            if search_for_path is not None:
                # Checkdigit validation
                val_check = ccc_validate(res_predict)
                if res_predict == ground_truth:
                    hit_str += 1
                    if len(ground_truth) <12:
                        val_check = "PASSED"

                    f3.write(repr(full_res_img_path).ljust(55) + repr(res_predict).center(20) + repr(ground_truth).rjust(15) + repr(res_confidence).rjust(10) + repr("HIT!").rjust(13) + repr(val_check).rjust(14) + '\n')
                else:
                    if len(ground_truth) <12:
                        val_check = "FAILED"
                    f3.write(repr(full_res_img_path).ljust(55) + repr(res_predict).center(20) + repr(ground_truth).rjust(15) + repr(res_confidence).rjust(10) + repr("MISSED!").rjust(13) + repr(val_check).rjust(14) + '\n')
    
    # Determine the longest_thread_time
    for lines in thread_time:
        time = lines.strip()
        if float(time) > longest_thread_time:
            longest_thread_time = float(time)
    
    fps = len(groundtruth_lines)/longest_thread_time
    acc = hit_str/len(groundtruth_lines)

    f3.write(f'Accuracy:\t {acc*100:{4}.{5}}% at {fps:{4}.{4}} fps\n')
    
    f1.close()
    f2.close()
    f3.close()
    f4.close()
        
    logger.info(f"Successful! post-processsed file is saved in {res_path + 'postprocessed_result.txt'}")
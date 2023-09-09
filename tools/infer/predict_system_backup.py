# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import subprocess
import collections
import datetime
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import math
import cv2
import copy
import numpy as np
import json
import time
import paddle
import logging
from PIL import Image
from IPython import display
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop

from tools.eval_manual import m_eval
from tools.firebase_read import fb_pull
from tools.cd_val import *
logger = get_logger()
fps_var = []

class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'csl': 0, 'all': 0}
        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict

def cleartempfiles(save_log_path):
    # Remove present result log file
    filelist = [f for f in os.listdir(save_log_path) if f.endswith(".txt") ]
    for f in filelist:
        os.remove(os.path.join(save_log_path, f))

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def run_paddle_ocr_as_api(source=0, flip=False, use_popup=False, skip_first_frames=0, text_sys=None, args=[]):
    """
    Main function to run the paddleOCR video/ cam inference:
    1. Create a video player to play with target fps (utils.VideoPlayer).
    2. Prepare a set of frames for text detection and recognition.
    3. Run AI inference for both text detection and recognition.
    4. Visualize the results.

    Parameters:
        source: The webcam number to feed the video stream with primary webcam set to "0", or the video path.
        flip: To be used by VideoPlayer function for flipping capture image.
        use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
        skip_first_frames: Number of frames to skip at the beginning of the video.
    """
    # Create a video player to play with target fps.
    player = None
    frameCount = 0
    try:
        player = utility.VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start video capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        results_log_lines = []
        Filtered_results_log_lines = []
        results_log_lines_skip_to_index = 0
        saveImage = False
        processing_times = collections.deque()
        
        while True:
            # Grab the frame.
            frame = player.next()
            frame_time = datetime.datetime.now()
            frameCount += 1
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(src=frame, dsize=None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)

            # Measure processing time for text detection and recognition.
            start_time = time.time()
            # Perform the inference step.
            dt_boxes, rec_res, time_dict = text_sys(frame)
            stop_time = time.time()
            #print("rec_res:", rec_res)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()
            processing_time_det = np.mean(processing_times) * 1000
            
            valid_rec_res = []
            valid_dt_boxes = []
            _predict = [("NULL", 0)]
            
            for id, (text, score) in enumerate(rec_res):
                temp_text = text
                temp_score = score

                # Concatenate the results if they are two-lines, i.e. text length < 5
                if _predict[0][0] != "NULL":
                    temp_text = _predict[0][0] + " " + text
                    temp_score = (score + _predict[0][1]) / 2
                
                # Check digit validation
                valid = ccc_validate(temp_text)
                #print("text:", temp_text, "is valid:", valid)
                if valid == "PASSED":
                    # Append the valid results
                    valid_rec_res.append((temp_text, temp_score))
                    valid_dt_boxes.append(dt_boxes[id])

                    #frame_time = datetime.datetime.now()
                    #print("append:", now.strftime("%Y-%m-%d %H:%M:%S") + "\t" + "{}\t{:.3f}".format(temp_text, temp_score) + "\n")
                    results_log_lines.append(frame_time.strftime("%Y-%m-%dT%H-%M-%S") + "\t" + "{}\t{:.3f}".format(temp_text, temp_score) + "\n")

                #print("rec_res:", valid_rec_res)
                # Should the prediction be concatenated if they are possibly two-lines
                if len(text) < 5:
                    _predict[0] = (text, score)
                else:
                    _predict[0] = ("NULL", 0)

                '''
                    Filter the duplicate recognized resutls.
                    Filter steps:
                        1. Compare the first recognized text with the next,
                        2. If they differ, save the first, get the next id,
                        3. Repeat step 1.
                '''
                skip_to = None

                # When results_log_lines_skip_to_index is at the last valid recognized text
                if results_log_lines_skip_to_index == len(results_log_lines) - 1:
                    saveImage = False

                for id, res in enumerate(results_log_lines[results_log_lines_skip_to_index:], results_log_lines_skip_to_index):
                    if id == (len(results_log_lines) - 2):
                        break
                    if skip_to != None:
                        if id != skip_to:
                            continue
                    # Base case, i.e. the first valid recognized text
                    if len(results_log_lines) == 1:
                        Filtered_results_log_lines.append(res)
                        saveImage = True
                        break
                        
                    for id_next, res_next in enumerate(results_log_lines[id + 1:], id + 1):
                        dateTime, res_text, res_score = res.strip().split('\t')
                        dateTime_next, res_text_next, res_score_next = res_next.strip().split('\t')
                        if res_text != res_text_next:
                            #print("id:", id, "res_text:", res_text)
                            #print("next id:", id_next, "rest_text_next:", res_text_next)
                            skip_to = id_next
                            results_log_lines_skip_to_index = skip_to
                            #print("skip to:", skip_to)
                            Filtered_results_log_lines.append(res_next)

                            saveImage = True
                            break


            #print("valid rec_res:", valid_rec_res, "saveImage:", saveImage)
                
            # For storing recognition results, include two parts:
            # txts are the recognized text results, scores are the recognition confidence level.
            txts = []
            scores = []
            for i in range(len(valid_dt_boxes)):
                txts.append(valid_rec_res[i][0])
                scores.append(valid_rec_res[i][1])

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Draw text recognition results beside the image.
            draw_img = draw_ocr_box_txt(
                image,
                valid_dt_boxes,
                txts,
                scores,
                drop_score=args.drop_score,
                font_path=args.vis_font_path)


            _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR))
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                

            # Save only the first recognized result as image, filter the duplicates
            if args.save_as_image and saveImage and len(valid_dt_boxes) != 0:
                _date = results_log_lines[results_log_lines_skip_to_index].strip().split('\t')[0]
                _text = results_log_lines[results_log_lines_skip_to_index].strip().split('\t')[1]
                _score = str(math.ceil(float(results_log_lines[results_log_lines_skip_to_index].strip().split('\t')[2]) * 100))
                # Save the valid recognized image
                image_path = _date + "_" + _text + "_" + _score + ".jpg"
                
                logger.info(f"filename: {image_path}, result: {valid_rec_res}")
                cv2.imwrite(
                    os.path.join(args.draw_img_save_dir,
                                image_path),
                    draw_img[:, :, ::-1])
                # Toggle saveImage state to ensure the next frame is not saved
                saveImage = False
            
            # Visualize the PaddleOCR results.
            f_height, f_width = draw_img.shape[:2]
            fps = 1000 / processing_time_det
            cv2.putText(img=draw_img, text=f"{frame_time.strftime('%Y-%m-%d %H:%M:%S')} Inference time: {processing_time_det:.1f}ms ({fps:.1f} FPS)",
                        org=(20, 30),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 2000,
                        color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            
            # Use this workaround if there is flickering.
            if use_popup:
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                cv2.imshow(winname=title, mat=draw_img)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                _, encoded_img = cv2.imencode(ext=".jpg", img=draw_img,
                                              params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)

        # Write the filtered valid recognized text to results.txt at save_log_path
        with open(args.save_log_path + "result.txt", "a") as fout:
            fout.writelines(Filtered_results_log_lines)
        ##return frameCount
    
    # ctrl-c
    except KeyboardInterrupt:
        logger.info("Interrupted")
    # any different error
    except RuntimeError as e:
        logger.info(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()
        ##return frameCount

def run_paddle_ocr(source=0, flip=False, use_popup=False, skip_first_frames=0, text_sys=None, args=[]):
    """
    Main function to run the paddleOCR video/ cam inference:
    1. Create a video player to play with target fps (utils.VideoPlayer).
    2. Prepare a set of frames for text detection and recognition.
    3. Run AI inference for both text detection and recognition.
    4. Visualize the results.

    Parameters:
        source: The webcam number to feed the video stream with primary webcam set to "0", or the video path.
        flip: To be used by VideoPlayer function for flipping capture image.
        use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
        skip_first_frames: Number of frames to skip at the beginning of the video.
    """
    # Create a video player to play with target fps.
    player = None
    frameCount = 0
    try:
        player = utility.VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start video capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        results_log_lines = []
        Filtered_results_log_lines = []
        results_log_lines_skip_to_index = 0
        saveImage = False
        processing_times = collections.deque()
        
        while True:
            # Grab the frame.
            frame = player.next()
            frame_time = datetime.datetime.now()
            frameCount += 1
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(src=frame, dsize=None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)

            # Measure processing time for text detection and recognition.
            start_time = time.time()
            # Perform the inference step.
            dt_boxes, rec_res, time_dict = text_sys(frame)
            stop_time = time.time()
            #print("rec_res:", rec_res)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()
            processing_time_det = np.mean(processing_times) * 1000
            
            valid_rec_res = []
            valid_dt_boxes = []
            _predict = [("NULL", 0)]
            
            for id, (text, score) in enumerate(rec_res):
                temp_text = text
                temp_score = score

                # Concatenate the results if they are two-lines, i.e. text length < 5
                if _predict[0][0] != "NULL":
                    temp_text = _predict[0][0] + " " + text
                    temp_score = (score + _predict[0][1]) / 2
                
                # Check digit validation
                valid = ccc_validate(temp_text)
                #print("text:", temp_text, "is valid:", valid)
                if valid == "PASSED":
                    # Append the valid results
                    valid_rec_res.append((temp_text, temp_score))
                    valid_dt_boxes.append(dt_boxes[id])

                    #frame_time = datetime.datetime.now()
                    #print("append:", now.strftime("%Y-%m-%d %H:%M:%S") + "\t" + "{}\t{:.3f}".format(temp_text, temp_score) + "\n")
                    results_log_lines.append(frame_time.strftime("%Y-%m-%dT%H-%M-%S") + "\t" + "{}\t{:.3f}".format(temp_text, temp_score) + "\n")

                #print("rec_res:", valid_rec_res)
                # Should the prediction be concatenated if they are possibly two-lines
                if len(text) < 5:
                    _predict[0] = (text, score)
                else:
                    _predict[0] = ("NULL", 0)

                '''
                    Filter the duplicate recognized resutls.
                    Filter steps:
                        1. Compare the first recognized text with the next,
                        2. If they differ, save the first, get the next id,
                        3. Repeat step 1.
                '''
                skip_to = None

                # When results_log_lines_skip_to_index is at the last valid recognized text
                if results_log_lines_skip_to_index == len(results_log_lines) - 1:
                    saveImage = False

                for id, res in enumerate(results_log_lines[results_log_lines_skip_to_index:], results_log_lines_skip_to_index):
                    if id == (len(results_log_lines) - 2):
                        break
                    if skip_to != None:
                        if id != skip_to:
                            continue
                    # Base case, i.e. the first valid recognized text
                    if len(results_log_lines) == 1:
                        Filtered_results_log_lines.append(res)
                        saveImage = True
                        break
                        
                    for id_next, res_next in enumerate(results_log_lines[id + 1:], id + 1):
                        dateTime, res_text, res_score = res.strip().split('\t')
                        dateTime_next, res_text_next, res_score_next = res_next.strip().split('\t')
                        if res_text != res_text_next:
                            #print("id:", id, "res_text:", res_text)
                            #print("next id:", id_next, "rest_text_next:", res_text_next)
                            skip_to = id_next
                            results_log_lines_skip_to_index = skip_to
                            #print("skip to:", skip_to)
                            Filtered_results_log_lines.append(res_next)

                            saveImage = True
                            break


            #print("valid rec_res:", valid_rec_res, "saveImage:", saveImage)
                
            # For storing recognition results, include two parts:
            # txts are the recognized text results, scores are the recognition confidence level.
            txts = []
            scores = []
            for i in range(len(valid_dt_boxes)):
                txts.append(valid_rec_res[i][0])
                scores.append(valid_rec_res[i][1])

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Draw text recognition results beside the image.
            draw_img = draw_ocr_box_txt(
                image,
                valid_dt_boxes,
                txts,
                scores,
                drop_score=args.drop_score,
                font_path=args.vis_font_path)

            # Save only the first recognized result as image, filter the duplicates
            if args.save_as_image and saveImage and len(valid_dt_boxes) != 0:
                _date = results_log_lines[results_log_lines_skip_to_index].strip().split('\t')[0]
                _text = results_log_lines[results_log_lines_skip_to_index].strip().split('\t')[1]
                _score = str(math.ceil(float(results_log_lines[results_log_lines_skip_to_index].strip().split('\t')[2]) * 100))
                # Save the valid recognized image
                image_path = _date + "_" + _text + "_" + _score + ".jpg"
                
                logger.info(f"filename: {image_path}, result: {valid_rec_res}")
                cv2.imwrite(
                    os.path.join(args.draw_img_save_dir,
                                image_path),
                    draw_img[:, :, ::-1])
                # Toggle saveImage state to ensure the next frame is not saved
                saveImage = False
            
            # Visualize the PaddleOCR results.
            f_height, f_width = draw_img.shape[:2]
            fps = 1000 / processing_time_det
            cv2.putText(img=draw_img, text=f"{frame_time.strftime('%Y-%m-%d %H:%M:%S')} Inference time: {processing_time_det:.1f}ms ({fps:.1f} FPS)",
                        org=(20, 30),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 2000,
                        color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            
            # Use this workaround if there is flickering.
            if use_popup:
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                cv2.imshow(winname=title, mat=draw_img)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                _, encoded_img = cv2.imencode(ext=".jpg", img=draw_img,
                                              params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)

        # Write the filtered valid recognized text to results.txt at save_log_path
        with open(args.save_log_path + "result.txt", "a") as fout:
            fout.writelines(Filtered_results_log_lines)
        ##return frameCount
    
    # ctrl-c
    except KeyboardInterrupt:
        logger.info("Interrupted")
    # any different error
    except RuntimeError as e:
        logger.info(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()
        ##return frameCount

def reader_wrapper(g):
    yield from g

def main(args):
    print('line580')
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    os.makedirs(args.draw_img_save_dir, exist_ok=True)
    save_results = []
    print('line585')
    '''logger.info(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
    )'''

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    total_time = 0
    #cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    imgCount = 0
    #temp_ccc = ""
    print(args)
    '''
        Predicting with images, with a set of groundtruths provided.    
    '''
    
    print('api:', args.asApi)
    if args.asApi:
        if args.input_type == "video":
            video_file = "inputs/CCCRVideoData1.mp4"
            print("asAPI: input type:", "video", video_file)
            wrap = reader_wrapper(run_paddle_ocr_as_api(source=video_file, flip=False, use_popup=False, skip_first_frames=0, text_sys=text_sys, args=args))
            for i in wrap:
                print(i)

            #yield from (run_paddle_ocr_as_api(source=video_file, flip=False, use_popup=False, skip_first_frames=0, text_sys=text_sys, args=args))
            #print("lines:", lines)
        
        elif args.input_type == "cam":
            print("input type:", "cam", 0)
            run_paddle_ocr_as_api(source=0, flip=False, use_popup=True, skip_first_frames=0, text_sys=text_sys, args=args)
    else:
        if args.input_type == "img":
            for idx, image_file in enumerate(image_file_list):
                img, flag_gif, flag_pdf = check_and_read(image_file)
                if not flag_gif and not flag_pdf:
                    img = cv2.imread(image_file)
                if not flag_pdf:
                    if img is None:
                        logger.debug("error in loading image:{}".format(image_file))
                        continue
                    imgs = [img]
                else:
                    page_num = args.page_num
                    if page_num > len(img) or page_num == 0:
                        page_num = len(img)
                    imgs = img[:page_num]
                for index, img in enumerate(imgs):
                    starttime = time.time()
                    dt_boxes, rec_res, time_dict = text_sys(img)
                    #print("386:",dt_boxes,rec_res,time_dict)
                    elapse = time.time() - starttime
                    now = datetime.datetime.now()
                    total_time += elapse
                    if len(imgs) > 1:
                        logger.debug(
                            str(idx) + '_' + str(index) + "  Predict time of %s: %.3fs"
                            % (image_file, elapse))
                    else:
                        logger.debug(
                            str(idx) + "  Predict time of %s: %.3fs" % (image_file,
                                                                        elapse))
                    _predict = [("NULL", 0)]
                    for text, score in rec_res:
                        logger.debug("{}, {:.3f}".format(text, score))

                        # Append the results
                        with open(args.save_log_path + "result.txt", "a") as fout:
                            fout.write(now.strftime("%Y-%m-%dT%H-%M-%S") + "\t" + image_file + "\t" + "{}\t{:.3f}".format(text, score) + "\n")
                            # Concatenate the results if they are two-lines, i.e. text length < 5
                            if _predict[0][0] != "NULL":
                                fout.write(now.strftime("%Y-%m-%dT%H-%M-%S") + "\t" + image_file + "\t" + "{}\t{:.3f}".format(_predict[0][0] + " " + text, (score + _predict[0][1]) / 2) + "\n")
                        
                        # Should the prediction be concatenated if they are possibly two-lines
                        if len(text) < 5:
                            _predict[0] = (text, score)
                        else:
                            _predict[0] = ("NULL", 0)
                    
                            
                    res = [{
                        "transcription": rec_res[i][0],
                        "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                    } for i in range(len(dt_boxes))]
                    #print("res:", res)
                    if len(imgs) > 1:
                        save_pred = os.path.basename(image_file) + '_' + str(
                            index) + "\t" + json.dumps(
                                res, ensure_ascii=False) + "\n"
                    else:
                        save_pred = os.path.basename(image_file) + "\t" + json.dumps(
                            res, ensure_ascii=False) + "\n"
                    save_results.append(save_pred)
                    
                    #logger.debug(f"Det: {text}, check digit validation: {validity}")
                    if args.save_as_image:
                        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        boxes = dt_boxes
                        txts = [rec_res[i][0] for i in range(len(rec_res))]
                        scores = [rec_res[i][1] for i in range(len(rec_res))]

                        draw_img = draw_ocr_box_txt(
                            image,
                            boxes,
                            txts,
                            scores,
                            drop_score=args.drop_score,
                            font_path=args.vis_font_path)
                        if flag_gif:
                            save_file = image_file[:-3] + "png"
                        elif flag_pdf:
                            save_file = image_file.replace('.pdf',
                                                        '_' + str(index) + '.png')
                        else:
                            save_file = image_file

                        image_path = now.strftime("%Y-%m-%dT%H-%M-%S") + "_" + "{}_{}".format(txts[0], math.ceil(scores[0] * 100))
                        save_file = image_path + '_' +  os.path.basename(save_file).split('.')[0] + '.' + os.path.basename(save_file).split('.')[1]

                        cv2.imwrite(
                            os.path.join(args.draw_img_save_dir,
                                        os.path.basename(save_file)),
                            draw_img[:, :, ::-1])
                        logger.debug("The visualized image saved in {}".format(
                            os.path.join(args.draw_img_save_dir, save_file)))
                # The number of images
                imgCount += 1
        
            '''
                Predicting with video, or cam without a set of groudtruths provided.
            '''
        elif args.input_type == "video":
            video_file = "inputs/CCCRVideoData1.mp4"
            print("input type:", "video", video_file)
            var = run_paddle_ocr(source=video_file, flip=False, use_popup=True, skip_first_frames=0, text_sys=text_sys, args=args)
            print("var:", var)
        
        elif args.input_type == "cam":
            print("input type:", "cam", 0)
            run_paddle_ocr(source=0, flip=False, use_popup=True, skip_first_frames=0, text_sys=text_sys, args=args)
    
    # not averaged time_dict
    #logger.info(f"Thread {thread_id}: Total prediction time for (detection, recognition, total, total + reading + writing)) = ({time_dict['det']:.3}, {time_dict['rec']:.3}, {time_dict['all']:.3}, {time.time() - _st:.3})s")
    
    # Stash the results
    thread_totaltime = float(time.time() - _st)
    '''
    thread_fps = (imgCount * args.total_process_num)/thread_totaltime

    logger.info("The frame per second is {:.2f}.".format(thread_fps))'''
    
    # Stash all the thread's running time
    if args.test_run == True:
        fps_var.append(str(thread_totaltime) +"\n")
        with open(args.save_log_path + "processes_time.txt", "a") as fout:  
            for line in fps_var:
                fout.write(line)
    
    if args.benchmark:
        text_sys.text_detector.autolog.report()
        text_sys.text_recognizer.autolog.report()

    '''with open(
            os.path.join(args.draw_img_save_dir, "system_results.txt"),
            'w',
            encoding='utf-8') as f:
        f.writelines(save_results)'''

if __name__ == "__main__":
    args = utility.parse_args()
    
    # Disable multi-threads if input type is video or cam
    if args.input_type != "img":
        args.use_mp = False
        args.test_run = False

    # Parse agruments
    save_log_path = args.save_log_path
    ground_truth_path = args.ground_truth_path
    total_process_num = args.total_process_num
    thread_id = args.process_id
    
    
    # Clear temp files
    cleartempfiles(args.save_log_path)

    try:       
        if args.use_mp:
            p_list = []
            total_process_num = args.total_process_num
            for args.process_id in range(1, args.total_process_num + 1):
                cmd = [sys.executable, "-u"] + sys.argv + [
                    "--process_id={}".format(args.process_id),
                    "--use_mp={}".format(False)
                ]
                p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
                p_list.append(p)
            for p in p_list:
                p.wait()
            
        else:
            print('main')
            main(args)
            
        if thread_id == 0 and args.test_run == True:
            # Post-processing the results
            m_eval(args.save_log_path, args.ground_truth_path)

    # ctrl-c
    except KeyboardInterrupt:
        logger.info("Interrupted")
    # any different error
    except RuntimeError as e:
        logger.info(e)

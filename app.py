# Copyright (c) 2020-2023 Lok Jing Lau PaddlePaddle Authors. All Rights Reserved.
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

import os, sys, time, subprocess, dotenv
import numpy as np
import docker, time, cv2, base64
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from tools.infer import utility, predict_system
from tools.eval_manual import m_eval


app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_interval=30, logger=False, engineio_logger=False)

encodings = ["utf-8", "latin-1", "ascii"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ping")
def ping():
    return Response("pong", 200)


@app.route("/pong")
def pong():
    return Response("Hello world.")


def args_constructor(target_type):
    use_gpu = os.environ.get("USE_GPU")
    gpu_mem = os.environ.get("GPU_MEM")
    cpu_threads = os.environ.get("CPU_THREADS")
    det_limit_side_len = os.environ.get("DET_LIMIT_SIDE_LEN")
    drop_score = os.environ.get("DROP_SCORE")
    det_db_thresh = os.environ.get("DET_DB_THRESH")
    ground_truth_path = os.environ.get("GROUND_TRUTH_PATH")
    show_log = os.environ.get("SHOW_LOG")

    default_args = [
        "--save_as_image", "True",
        "--warmup", "True",
        "--input_type", target_type,
    ]

    """
    Parse arguments
    """
    # ----- Processor -----
    if use_gpu == "gpu":
        default_args.extend(["--use_gpu", "True", "--gpu_mem", str(gpu_mem)])
    else:
        default_args.extend(["--use_gpu", "False", "--cpu_threads", str(cpu_threads)])
    # ----- Input path -----
    # General args
    if det_limit_side_len != "" and det_limit_side_len != None:
        default_args.extend(["--det_limit_side_len", str(det_limit_side_len)])

    if drop_score != "" and drop_score != None:
        default_args.extend(["--drop_score", str(drop_score)])

    if det_db_thresh != "" and det_db_thresh != None:
        default_args.extend(["--det_db_thresh", str(det_db_thresh)])

    if show_log != "" and show_log != None:
        if show_log == "true":
            default_args.extend(["--show_log", "True"])

    # Conditional args
    if target_type == "img":
        default_args.extend(["--image_dir", "./myInput/"])

        if ground_truth_path != "" and ground_truth_path != None and os.path.basename(ground_truth_path).endswith('.txt'):
            default_args.extend(
                ["--ground_truth_path", "./otherInputs/" + os.path.basename(ground_truth_path), "--test_run", "True"]
            )

    print("default args:", default_args)
    return default_args


@app.route("/img", methods=["GET", "POST"])
def handle_image():
    default_args = args_constructor("img")

    input_type = os.environ.get("INPUT_TYPE")
    image_dir = os.environ.get("IMAGE_DIR")
    print("img: default_args:", default_args)

    # Invalid configuration handling
    if input_type != "img":
        return Response("Invalid configuration.", 400)
    elif image_dir == "" or image_dir == None:
        return Response("Missing image directory.", 400)
    elif os.path.dirname(image_dir) == "":
        return Response("Invalid image directory.", 400)
    
    # Parse agruments
    parser = utility.init_args()
    args = parser.parse_args(default_args)
    # for groundtruth evaluation
    args.process_id = 0

    # Disable multi-threads if input type is video or cam
    if args.input_type != "img":
        args.use_mp = False
        args.test_run = False

    # Clear temp files
    predict_system.cleartempfiles(args.save_log_path)

    text_sys = predict_system.TextSystem(args)
    os.makedirs(args.draw_img_save_dir, exist_ok=True)

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    predict_system.main(args)

    if args.test_run:
        print("Running post evaluation.")
        m_eval(args.save_log_path, args.ground_truth_path)

    return Response("Executed", 201)


@app.route("/video", methods=["GET", "POST"])
def handle_video():
    """if request.method == 'POST':
    if request.is_json:
        try:
            data = request.get_json()
            print("route ('/video'), received data:", data)
            #filename = data['filename']
            for key, value in data.items():
                print(key, ":", value)

                dotenv_file = dotenv.find_dotenv()
                dotenv.load_dotenv(dotenv_file)

                os.environ[key] = value

                # Write changes to .env file.
                dotenv.set_key(dotenv_file, key, os.environ[key])

        except Exception as e:
            print("Error parsing JSON data:", e)
            return Response("Bad Request", 400)
    #else:
    #    return Response("Bad Request, not JSON", 400)
    """

    input_type = os.environ.get("INPUT_TYPE")
    image_dir = os.environ.get("IMAGE_DIR")
    video_filepath = "myInput/" + os.path.basename(image_dir)

    # Invalid configuration handling
    if input_type != "video":
        return Response("Invalid configuration.", 400)
    elif image_dir == "" or image_dir == None:
        return Response("Missing video filepath.", 400)
    elif video_filepath == "":
        return Response("Invalid video filepath.", 400)

    default_args = args_constructor("video")

    parser = utility.init_args()
    args = parser.parse_args(default_args)

    # Parse agruments
    thread_id = args.process_id

    # Disable multi-threads if input type is video or cam
    if args.input_type != "img":
        args.use_mp = False
        args.test_run = False

    # Clear temp files
    predict_system.cleartempfiles(args.save_log_path)

    text_sys = predict_system.TextSystem(args)
    os.makedirs(args.draw_img_save_dir, exist_ok=True)

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    resp = Response(
        predict_system.run_paddle_ocr_as_api(
            source=video_filepath,
            flip=False,
            use_popup=False,
            skip_first_frames=0,
            text_sys=text_sys,
            args=args,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

    return resp


@app.route("/cam")
def handle_cam():
    input_type = os.environ.get("INPUT_TYPE")

    if input_type != "cam":
        return Response("Invalid configuration.", 400)

    default_args = args_constructor("cam")

    parser = utility.init_args()
    args = parser.parse_args(default_args)

    # Parse agruments
    thread_id = args.process_id
    # print('\nargs:', args)

    # Disable multi-threads if input type is video or cam
    if args.input_type != "img":
        args.use_mp = False
        args.test_run = False

    # Clear temp files
    predict_system.cleartempfiles(args.save_log_path)

    text_sys = predict_system.TextSystem(args)
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    resp = Response(
        predict_system.run_paddle_ocr_as_api(
            source=0,
            flip=False,
            use_popup=False,
            skip_first_frames=0,
            text_sys=text_sys,
            args=args,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

    resp.headers["Access-Control-Allow-Origin"] = "*"

    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)

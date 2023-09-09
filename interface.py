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

import docker, time, requests, dotenv
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from tools.infer import utility, predict_system
from tools.eval_manual import m_eval
import os, sys, time, subprocess, cv2
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = "ASJCIASDNDU128431713491ADJNADK3ds3ASDKJ"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_interval=30, logger=False, engineio_logger=False)

encodings = ["utf-8", "latin-1", "ascii"]

defaultConfig = {
    "CONFIGURED": "false",
    "USE_GPU": "gpu",
    "GPU_MEM": "21000",
    "CPU_THREADS": "10",
    "CAM_POS": "0",
    "inputs": {
        "INPUT_TYPE": "img",
        "IMAGE_DIR": "/home/acery/tmp/inputs/video/CCCRVideoData1.mp4",
        "DRAW_IMG_SAVE_DIR": "/home/acery/tmp/outputs/",
        "DET_LIMIT_SIDE_LEN": "1280",
        "DROP_SCORE": "0.5",
        "DET_DB_THRESH": "0.3",
        "GROUND_TRUTH_PATH": "",
        "SAVE_LOG_PATH": "/home/acery/tmp/logs/",
        "SHOW_LOG": "false",
    },
    "requirements": {
        "REQ_DOCKER": "false",
        "REQ_IMAGE": "false",
        "REQ_NVIDIA": "false",
        "REQ_CUDA": "false",
        "REQ_CAM": "false",
    },
    "status": {
        "STAT_CONTAINER": "false",
        "STAT_SERVER": "false",
    },
    "message": {
        "MSG_DOCKER": "",
        "MSG_IMAGE": "",
        "MSG_NVIDIA": "",
        "MSG_CUDA": "",
        "MSG_CAM": "",
    },
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reset_env", methods=["POST"])
def reset_env():
    """
    Reset env. variables to default value.
    """
    for key, value in defaultConfig.items():
        # print(key, ":", value)

        dotenv_file = dotenv.find_dotenv()
        dotenv.load_dotenv(dotenv_file)

        # print(value, "is dict?", isinstance(value, dict), type(value))
        if isinstance(value, dict):
            nestedDict = value
            for key, value in nestedDict.items():
                os.environ[key] = value

                # Write changes to .env file.
                dotenv.set_key(dotenv_file, key, os.environ[key])
        else:
            os.environ[key] = value

            # Write changes to .env file.
            dotenv.set_key(dotenv_file, key, os.environ[key])

    return Response("Envs reset to default.", 201)


@app.route("/run_container", methods=["POST"])
def run_container():
    """
    Run docker container from image "laulj/ccc_ocr-tensorflow-cuda11.2:1.4.0"
    """

    # Create a Docker client
    client = docker.from_env()

    # ----- GPU configuration -----
    device_requests = []
    use_gpu = os.environ.get("USE_GPU")
    if use_gpu == "gpu":
        device_requests = [
            docker.types.DeviceRequest(
                driver="nvidia", count=-1, capabilities=[["gpu"]]
            )
        ]

    # ----- Docker volume configuration -----
    volumeConfigs = {}
    input_key = None
    input_type = os.environ.get("INPUT_TYPE")
    image_dir = os.environ.get("IMAGE_DIR")
    draw_img_save_dir = os.environ.get("DRAW_IMG_SAVE_DIR")
    save_log_path = os.environ.get("SAVE_LOG_PATH")
    ground_truth_path = os.environ.get("GROUND_TRUTH_PATH")

    if input_type == "img":
        if image_dir is not None and image_dir != "":
            input_key = image_dir

    elif input_type == "video":
        # if (image_dir is not None and image_dir != "") and (video_filename is not None and video_filename != ""):
        if image_dir is not None and image_dir != "":
            # dir, filename = os.path.dirname(image_dir), os.path.basename(image_dir)
            input_key = os.path.dirname(image_dir)

    if input_key is not None:
        volumeConfigs[input_key] = {
            "bind": "/home/PaddleOCR/myInput/",
            "mode": "ro",
        }
    if draw_img_save_dir is not None and draw_img_save_dir != "":
        volumeConfigs[draw_img_save_dir] = {
            "bind": "/home/PaddleOCR/inference_results/",
            "mode": "rw",
        }
    if save_log_path is not None and save_log_path != "":
        volumeConfigs[save_log_path] = {
            "bind": "/home/PaddleOCR/log_output/",
            "mode": "rw",
        }
    if ground_truth_path is not None and ground_truth_path != "" and os.path.basename(ground_truth_path).endswith('.txt'):
        volumeConfigs[os.path.dirname(ground_truth_path)+ "/"] = {
            "bind": "/home/PaddleOCR/otherInputs/",
            "mode": "ro",
        }
    print("volume config.:", volumeConfigs)

    # ----- Camera Configuration -----
    devices = []
    cam_pos = os.environ.get("CAM_POS")
    cam_status = cv2.VideoCapture(int(cam_pos)) != None and cv2.VideoCapture(int(cam_pos)).isOpened()
    if cam_pos is not None and cam_pos != "" and cam_status:
        devices = [f"/dev/video{cam_pos}:/dev/video{cam_pos}"]

    # ----- Environment Variables -----
    dotenv_file = dotenv.find_dotenv()
    if dotenv_file != "":
        dotenv.load_dotenv(dotenv_file)
    else:
        return Response("Environment file not found", 500)

    try:
        container = client.containers.get("paddleocr")
        print("container status:", container.status)

        # Remove the slept or shutted down docker container
        if container.status == "exited":
            container.remove()
        else:
            container.remove(v=True, force=True)
    except:
        print("No existing container.")
    
    try:
        # If there is no existing docker container
        client.containers.run(
            "laulj/ccc_ocr-tensorflow-cuda11.2:1.4.0",
            command=["/bin/bash"],
            devices=devices,
            device_requests=[
                docker.types.DeviceRequest(
                    driver="nvidia", count=-1, capabilities=[["gpu"]]
                )
            ],
            detach=True,
            environment=dict(dotenv.dotenv_values(dotenv_file)),
            ports={"5005/tcp": 5555},
            stdin_open=True,
            tty=True,
            name="paddleocr",
            volumes=volumeConfigs,
        )
    except docker.errors.APIError:
        print("Docker container failed to start.")
    # Verify container status
    return container_status()


@socketio.on("run_containerServer")
def run_containerServer():
    """
    Assuming the docker container from image "laulj/ccc_ocr-tensorflow-cuda11.2:1.4.0" is running,
    run a flask server 'app.py', in the container.
    """
    isServerStarted = False

    try:
        response = requests.get("http://172.17.0.2:5005/ping")
        print("\n\nresponse:", response)
        if response.status_code == 200:
            print("Connection is good.")
            isServerStarted = True
        else:
            print("Connection error:", response.status_code)

    except requests.exceptions.RequestException as e:
        print("Exception:", e)

    if not isServerStarted:
        print("Trying to start server...")
        input_data = "python3 app.py"
        container = client.containers.get("paddleocr")

        # Start the server in the docker container
        code = container.exec_run(cmd=input_data, stdin=True, socket=True)

        for chunk in code.output:
            print("output:", chunk)
            for encoding in encodings:
                try:
                    decoded_string = chunk.decode(encoding)
                    print(f"Decoded string using {encoding}: {decoded_string}")
                    break
                except UnicodeDecodeError:
                    continue
            socketio.emit("server_logs", decoded_string)


@app.route("/pull-image")
def pull_image(image_name, image_tag):
    """
    Pull a docker image from DockerHub
    """

    # Create a Docker client
    client = docker.from_env()

    try:
        # Pull the image
        response = client.api.pull(f"{image_name}:{image_tag}", stream=True)

        for line in response:
            # Delay 0.1s
            time.sleep(0.1)

            # Stream the output to the frontend
            line_data = line.decode("utf-8").strip()
            socketio.emit("progress", {"data": line_data})

        socketio.emit("progress", {"status": "Image pull completed"})

    except Exception as e:
        print("error:", e)
        socketio.emit("progress", {"status": e})


@socketio.on("start_pull")
def start_pull_image():
    # Specify the image name and tag laulj/ccc_ocr-tensorflow-cuda11.2:1.4.0
    image_name = "laulj/ccc_ocr-tensorflow-cuda11.2"
    image_tag = "1.4.0"

    pull_image(image_name, image_tag)


@app.route("/read_env", methods=["POST"])
def read_env():
    """
    Return the existing env. variables as configuration
    """
    # filename = os.environ.get("VIDEO_FILENAME")
    configured = os.environ.get("CONFIGURED")
    use_gpu = os.environ.get("USE_GPU")
    gpu_mem = os.environ.get("GPU_MEM")
    cpu_threads = os.environ.get("CPU_THREADS")
    cam_pos = os.environ.get("CAM_POS")
    input_type = os.environ.get("INPUT_TYPE")
    image_dir = os.environ.get("IMAGE_DIR")
    draw_img_save_dir = os.environ.get("DRAW_IMG_SAVE_DIR")
    det_limit_side_len = os.environ.get("DET_LIMIT_SIDE_LEN")
    drop_score = os.environ.get("DROP_SCORE")
    det_db_thresh = os.environ.get("DET_DB_THRESH")
    ground_truth_path = os.environ.get("GROUND_TRUTH_PATH")
    save_log_path = os.environ.get("SAVE_LOG_PATH")
    show_log = os.environ.get("SHOW_LOG")
    req_docker = os.environ.get("REQ_DOCKER")
    req_image = os.environ.get("REQ_IMAGE")
    req_nvidia = os.environ.get("REQ_NVIDIA")
    req_cuda = os.environ.get("REQ_CUDA")
    req_cam = os.environ.get("REQ_CAM")
    stat_container = os.environ.get("STAT_CONTAINER")
    stat_server = os.environ.get("STAT_SERVER")
    msg_docker = os.environ.get("MSG_DOCKER")
    msg_image = os.environ.get("MSG_IMAGE")
    msg_nvidia = os.environ.get("MSG_NVIDIA")
    msg_cuda = os.environ.get("MSG_CUDA")
    msg_cam = os.environ.get("MSG_CAM")

    data = {
        "configured": configured,
        "processor": use_gpu,
        "gpu_mem": gpu_mem,
        "cpu_threads": cpu_threads,
        "cam_pos": cam_pos,
        "inputs": {
            "type": input_type,
            "image_dir": image_dir,
            "output_dir": draw_img_save_dir,
            "img_side_limit": det_limit_side_len,
            "drop_score": drop_score,
            "det_db_thresh": det_db_thresh,
            "ground_truth_path": ground_truth_path,
            "log_output": save_log_path,
            "show_log": show_log,
        },
        "requirements": {
            "isDocker": req_docker,
            "isImage": req_image,
            "isNvidia": req_nvidia,
            "isCUDA": req_cuda,
            "isCam": req_cam,
        },
        "status": {
            "container": stat_container,
            "server": stat_server,
        },
        "message": {
            "isDocker": msg_docker,
            "isImage": msg_image,
            "isNvidia": msg_nvidia,
            "isCUDA": msg_cuda,
            "isCam": msg_cam,
        },
    }
    print("config data:", data)
    return data, 200


@app.route("/write_env", methods=["POST"])
def write_env():
    """
    Saving the configuration as env. variables
    """
    if request.is_json:
        try:
            data = request.get_json()
            print("route ('/write_env'), received data:", data)

            dotenv_file = dotenv.find_dotenv()
            if dotenv_file != "":
                dotenv.load_dotenv(dotenv_file)
            else:
                return Response("Environment file not found", 500)

            for key, value in data.items():
                # print(key, ":", value)

                os.environ[key] = value

                # Write changes to .env file.
                dotenv.set_key(dotenv_file, key, os.environ[key])
            # print("dotenv_values:", dotenv.dotenv_values(dotenv_file))
            print("dict:", dict(dotenv.dotenv_values(dotenv_file)))

            return Response("Environment variable changed", 201)

        except Exception as e:
            print("Error parsing JSON data:", e)
            return Response("Bad Request", 400)
    else:
        return Response("Bad Request", 400)


# -------- Requirements Verification Route --------
@app.route("/docker_check", methods=["POST"])
def docker_check():
    try:
        # Create a Docker client
        client = docker.from_env()

        container = client.containers.run(
            "docker/getting-started",
            detach=True,
        )
        print("/docker_check container:", container)
        container.remove(force=True)
        return {
            "message": "Container' docker/getting-started' started and removed successfully"
        }, 200

    except Exception as e:
        return {"message": "Failed to start 'docker/getting-started' container"}, 400


@app.route("/image_check", methods=["POST"])
def image_check():
    image = None
    try:
        # Create a Docker client
        client = docker.from_env()
        try:
            # image = client.images.get("laulj/ccc_ocr-tensorflow-cuda11.2:1.4.0")
            image = client.images.get("laulj/ccc_ocr-tensorflow-cuda11.2:1.4.0")
            return {"message": "Image exists"}, 200

        except docker.errors.ImageNotFound:
            return {"message": "Image not found"}, 400

        except docker.errors.APIError:
            return {"message": "Docker API errors"}, 500

        except Exception as e:
            print("error:", e)
            return {"message": "Server errors."}, 500

    except Exception as e:
        return {"message": "Failed to create docker client"}, 500


@app.route("/nvidia_check", methods=["POST"])
def nvidia_check():
    try:
        # Create a Docker client
        client = docker.from_env()
        container = client.containers.run(
            "laulj/ccc_ocr-tensorflow-cuda11.2:1.4.0",
            detach=True,
            stdin_open=True,
            tty=True,
            device_requests=[
                docker.types.DeviceRequest(
                    driver="nvidia", count=-1, capabilities=[["gpu"]]
                )
            ],
        )

        container.remove(force=True)
        return {"message": "Container started and removed successfully"}, 200

    except Exception as e:
        return {"message": "Failed to start container with nvidia driver"}, 400


@app.route("/cuda_check", methods=["POST"])
def cuda_check():
    try:
        # Create a Docker client
        client = docker.from_env()
        container = client.containers.run(
            "laulj/ccc_ocr-tensorflow-cuda11.2:1.4.0",
            command=["/bin/bash"],
            detach=True,
            stdin_open=True,
            tty=True,
            device_requests=[
                docker.types.DeviceRequest(
                    driver="nvidia", count=-1, capabilities=[["gpu"]]
                )
            ],
        )
        # python3 -c 'import tensorflow as tf; print(tf.__version__)'

        input_data = "python3 tools/isCUDA.py"
        # print("inputdata:", input_data)
        # Run the command
        code = container.exec_run(cmd=input_data, stdin=True)
        print("code:", code)
        print("output:", code.exit_code, code.output)

        container.remove(force=True)
        if code.exit_code == 1:
            return {"message": "Failed to import CUDA library."}, 400

        return {"message": "Container started and removed successfully"}, 201

    except Exception as e:
        print("e:", e)
        return {"message": "Failed to start container"}, 400


@app.route("/cam_check", methods=["POST"])
def cam_check():
    try:
        cam_pos = request.args.get("pos")
        print("cam pos:", cam_pos)
        # Create a Docker client
        client = docker.from_env()

        container = client.containers.run(
            "laulj/ccc_ocr-tensorflow-cuda11.2:1.4.0",
            command=["/bin/bash"],
            detach=True,
            devices=[f"/dev/video{cam_pos}:/dev/video{cam_pos}"],
            environment=[f"CAM_POS={cam_pos}"],
            stdin_open=True,
            tty=True,
        )
        # print("container:", container.logs())

        # Run the command
        code = container.exec_run(cmd="python3 tools/isCam.py", stdin=True)
        # print("code:", code)
        # print("output:", code.exit_code, code.output)

        container.remove(force=True)
        if code.exit_code == 1:
            return {"message": f"Failed to detect camera at position {cam_pos}"}, 400

        return {"message": f"Camera is detected at position {cam_pos}"}, 201

    except Exception as e:
        print("e:", e)
        return {"message": "Failed to detect the camera"}, 400


@app.route("/container_status", methods=["GET"])
def container_status():
    """
    Return 200 if the container is running, else 400.
    """
    try:
        container = client.containers.get("paddleocr")
        # print("container status:", container.status)
        return {"message": "Container is online."}, 200
    except Exception as e:
        return {"message": "Container is offline."}, 400


if __name__ == "__main__":
    # Reset env vars to default value
    reset_env()

    # Create a Docker client
    client = docker.from_env()

    app.run(port=7000, debug=True)

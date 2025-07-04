from config_provider import ConfigProvider

ConfigProvider.device = "cpu"
ConfigProvider.debug_print = True
ConfigProvider.print_args()


import cv2
from flask import Flask, Response, jsonify, send_file
from flask_model import DataModel


app = Flask(__name__)
dataModel = DataModel()

@app.route('/')
def index():
    return "suburban-env"


@app.route("/cmd/<command>", methods=['GET'])
def single_cmd(command: str):
    return dataModel.parse_command(command, [])

@app.route("/cmd/<command>/<args>", methods=['GET'])
def cmd(command: str, args: str):
    args = args.split('$')
    return dataModel.parse_command(command, args)

@app.route('/current_obs', methods=['GET'])
def current_obs():

    img = dataModel.get_obs()
    # 将图像从 BGR 转换为 RGB（因为 Flask 的默认 MIME 类型是图片格式的 JPEG/PNG/等）
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将图像编码为 JPEG 格式并转换为字节流
    ret, jpeg = cv2.imencode('.jpg', img)
    if not ret:
        return 'Failed to encode image', 500

    # 创建一个响应，将图像作为字节流发送
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/current_radius_linear')
def current_radius_linear():
    return dataModel.get_current_radius_min_max()


@app.route('/current_space_type')
def current_space_type():
    return dataModel.get_current_space_type()


@app.route('/total_state')
def total_state():
    return dataModel.get_total_state()


@app.route('/all_actions')
def history_actions():
    return jsonify(
        dataModel.get_all_actions()
    )


@app.route('/all_info')
def all_info():
    return jsonify(
        dataModel.get_all_save_info()
    )


@app.route('/cover/<name>')
def cover(name: str):
    return send_file(
        dataModel.get_specific_history_cover(name),
        mimetype='image/jpeg'
    )


@app.route('/current_obs_plus/<int:size>', methods=['GET'])
def current_obs_plus(size: int):

    img = dataModel.get_obs_plus(size)
    # 将图像从 BGR 转换为 RGB（因为 Flask 的默认 MIME 类型是图片格式的 JPEG/PNG/等）
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将图像编码为 JPEG 格式并转换为字节流
    ret, jpeg = cv2.imencode('.jpg', img)
    if not ret:
        return 'Failed to encode image', 500

    # 创建一个响应，将图像作为字节流发送
    return Response(jpeg.tobytes(), mimetype='image/jpeg')



if __name__ == '__main__':
    app.run(debug=True)

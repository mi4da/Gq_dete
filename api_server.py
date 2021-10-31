import json
import os
import sys
import time
import traceback

import flask

from detection_model import ModelPredict

app = flask.Flask(__name__)

# 设置显卡
os.environ['CUDA_VISIBLE_DEVICE'] = "0"


# 初始化服务
@app.before_first_request
def init():
    print("__init before first request__")
    global model_predict
    model_predict = ModelPredict()


# index
@app.route("/", methods=['GET', 'POST'])
def index():
    return "YOLO-V4 目标检测服务已启动"


# 推理路由设置
@app.route("/ai/detection_predict", methods=['POST'])
def get_result():
    if flask.request.method == "POST":
        try:
            s = time.time()
            res = model_predict.generator_result(flask.request.data)
            result = {
                "code": 200,
                "msg": "推理成功",
                "result": res,
                "time_cost": time.time() - s
            }

        except Exception as e:
            print(e)
            result_error = {
                "errcode": -1
            }
            result = json.dumps(result_error, indent=4, ensure_ascii=False)
            # 这里用于捕获更详细的异常信息
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            # 提前退出请求
            flask.abort(flask.Response("Failed!\n" + '\n\r\n'.join('' + line for line in lines)))

        return flask.Response(str(result), mimetype="application/json")  # 指定回复类型, result一定要转换成str


if __name__ == '__main__':
    app.run(debug=False, port=8888)

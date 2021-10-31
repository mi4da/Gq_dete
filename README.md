# Gq_dete
Trash detective project, which base on Yolov4.
# updata2021/10/31
垃圾检测服务暂时部署在学校的内网上了
请求地址：http://192.168.3.10:8888/ai/detection_predict
请求方法：POST
请求数据：二进制图片

返回数据

{
                "code": 200,
                "msg": "推理成功",
                "result": [种类，置信度，坐标1，坐标2，坐标3，坐标],
                "time_cost": *
            }

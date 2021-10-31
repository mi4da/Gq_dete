# coding:utf-8
import cv2
import numpy as np

import args


class ModelPredict:
    def __init__(self):
        # 初始化一些参数
        self.LABELS = open(args.labelsPath, encoding='utf-8').read().strip().split("\n")
        self.boxes = []
        self.confidences = []
        self.classIDs = []
        self.net = cv2.dnn.readNetFromDarknet(args.configPath, args.weightsPath)

    def generator_result(self, img_data: bytes) -> list:
        # 将二进制数据转换成图象
        img_data = np.frombuffer(img_data, np.uint8)
        # cv2.IMREAD_COLOR 加载彩色图象， 必要的参数
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        # 读入待检测的图像
        # image = cv2.imread(img_data)
        # 得到图像的高和宽
        (H, W) = image.shape[0:2]

        # 得到 YOLO需要的输出层
        ln = self.net.getLayerNames()
        out = self.net.getUnconnectedOutLayers()  # 得到未连接层得序号  [[200] /n [267]  /n [400] ]
        x = []
        for i in out:  # 1=[200]
            x.append(ln[i[0] - 1])  # i[0]-1    取out中的数字  [200][0]=200  ln(199)= 'yolo_82'
        ln = x
        # 从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True,
                                     crop=False)  # 构造了一个blob图像，对原图像进行了图像的归一化，缩放了尺寸 ，对应训练模型

        self.net.setInput(blob)  # 将blob设为输入？？？ 具体作用还不是很清楚
        layerOutputs = self.net.forward(ln)  # ln此时为输出层名称  ，向前传播  得到检测结果

        for output in layerOutputs:  # 对三个输出层 循环
            for detection in output:  # 对每个输出层中的每个检测框循环
                scores = detection[5:]  # detection=[x,y,h,w,c,class1,class2] scores取第6位至最后
                classID = np.argmax(scores)  # np.argmax反馈最大值的索引
                confidence = scores[classID]

                if confidence > 0.5:  # 过滤掉那些置信度较小的检测结果
                    box = detection[0:4] * np.array([W, H, W, H])
                    # print(box)
                    (centerX, centerY, width, height) = box.astype("int")
                    # 边框的左上角
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # 更新检测出来的框
                    self.boxes.append([x, y, int(width), int(height)])
                    self.confidences.append(float(confidence))
                    self.classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.2, 0.3)
        box_seq = idxs.flatten()  # [ 2  9  7 10  6  5  4]
        if len(idxs) > 0:
            for seq in box_seq:
                (x, y) = (self.boxes[seq][0], self.boxes[seq][1])  # 框左上角
                (w, h) = (self.boxes[seq][2], self.boxes[seq][3])  # 框宽高
                if self.classIDs[seq] == 0:  # 根据类别设定框的颜色
                    color = [0, 0, 255]
                else:
                    color = [0, 255, 0]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # 画框
                # text = "{}: {:.4f}".format(self.LABELS[self.classIDs[seq]], self.confidences[seq])
                cv2.putText(image, 'Classs', (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)  # 写字
                # 显示具体信息，顺序是 类别与置信度，方框xywh位置,（打中文会乱码，所以一律输出Classs
                return [self.LABELS[self.classIDs[seq]], round(self.confidences[seq], 4), x, y, w, h]


if __name__ == '__main__':
    predictor = ModelPredict()
    print(predictor.generator_result('Test_Pic/fimg_0.jpg'))

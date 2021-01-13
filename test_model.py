import torch
import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import Data_Loader.Dataset as data
import math as math
import argparse

batch_size = 1

img_root = './agedb_30_masked/images'
test_txt = './agedb_30_masked/lables_test.txt'
module_file = './models/mask_detection.pkl'

transform = transforms.Compose([
    transforms.Resize(300),  # 图像缩小
    # transforms.CenterCrop(128),  # 中心剪裁
    # transforms.RandomHorizontalFlip(),  # 依概率p水平翻转
    transforms.ToTensor(),  # 转tensor 并归一化
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # 标准化
                         std=[0.5, 0.5, 0.5])
])

# 加载模型
module = torch.load(module_file)
module.eval()


# 通过摄像头
def CatchPICFromVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)

    # 从摄像头读取图像
    cap = cv2.VideoCapture(camera_idx)

    # 告诉OpenCV使用人脸识别分类器
    eye_classifier = cv2.CascadeClassifier("./opencv/haarcascade_eye.xml")
    # 识别出人脸后要画的边框的颜色，RGB格式, color是一个不可增删的数组
    no_color = (0, 0, 255)
    color = (0, 255, 0)
    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像

        # 人脸检测，1.2和3分别为图片缩放比例和需要检测的有效点数
        eyeRects = eye_classifier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3)
        if len(eyeRects) > 0: # 检测到了人眼
            eye_tags = []
            for eyeRect in eyeRects:
                x, y, w, h = eyeRect
                eye_tag = [x, y, w, h]
                eye_tags.append(eye_tag)
            if len(eye_tags) == 2:
                # 计算眼睛位置眼睛
                x1 = int(eye_tags[0][0] + eye_tags[0][2]/2)
                x2 = int(eye_tags[1][0] + eye_tags[1][2]/2)
                y1 = int(eye_tags[0][1] + eye_tags[0][3]/2)
                y2 = int(eye_tags[1][1] + eye_tags[1][3]/2)
                # cv2.circle(frame, (x1, y1),2,color)
                # cv2.circle(frame, (x2, y2), 2, color)
                # 根据眼睛位置推测脸部位置
                eyes_dis = int(math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2)))  # 眼距
                eyes_mid = (int((x1+x2)/2), int((y1+y2)/2))                         # 眼睛中点
                face_height = int(eyes_dis * 2.8)                                   # 脸长=眼距 * 2.8
                # 计算脸部矩形位置坐标
                rec_xl = int(eyes_mid[0] - (eyes_dis/2 * 2))
                rec_yl = int(eyes_mid[1] - (face_height/3))
                rec_xr = int(eyes_mid[0] + (eyes_dis/2 * 2))
                rec_yr = int(eyes_mid[1] + (face_height * (2/3)))

                x = rec_xl
                y = rec_yl
                w = rec_xr - rec_xl
                h = rec_yr - rec_yl
                image = frame[y: y + h + 10, x: x + w + 10]
                # cv2.imshow("current", image)
                image = Image.fromarray(image)
                image = transform(image).unsqueeze(0)
                output = module(image)
                # print(output)
                _, predict = torch.max(output, 1) # 获取预测结果

                if predict == 1:
                    result = "noMask"
                else:
                    result = "Masked"

                # 绘制矩形框
                if result == "noMask":
                    cv2.rectangle(frame, (rec_xl, rec_yl), (rec_xr, rec_yr), no_color, 1)
                    cv2.rectangle(frame, (rec_xl, rec_yl), (rec_xr, rec_yl + 15), no_color, -1)
                else:

                    cv2.rectangle(frame, (rec_xl, rec_yl), (rec_xr, rec_yr), color, 1)
                    cv2.rectangle(frame, (rec_xl, rec_yl), (rec_xr, rec_yl + 15), color, -1)

                # 显示结果
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, result + ":  %.2f" % _, (x, y+10), font, 0.4, (0, 0, 0), 1)



        # 显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


def CatchPICByFileName(filepath):
    image = data.default_loader(filepath)
    image = data.transform(image).unsqueeze(0)
    image = Variable(image)

    output = module(image)

    _, predict = torch.max(output, 1)
    if predict == 1:
        print("Masked")
    else:
        print("noMask")


# 通过图片数据集
def test_by_image_set():
    print("Predicting >>>", end=" ")
    test_dataset = data.myDataset(img_dir=img_root, img_txt=test_txt, transform=data.transform)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    all_correct_num = 0

    for ii, (img, label) in enumerate(tqdm(test_dataloader)):
        img = Variable(img)
        label = Variable(label)

        output = module(img)  # 前馈计算
        _, predict = torch.max(output, 1)  # 按列取最大值

        correct_num = sum(predict == label.data.item())
        all_correct_num += correct_num.data.item()
    Accuracy = all_correct_num * 1.0 / (len(test_dataset))  # 计算正确率
    print('all_correct_num={0},Accuracy={1}'.format(all_correct_num, Accuracy))


def main(argv):
    if argv.mode == 'video':
        CatchPICFromVideo("Get Face", 0)
    elif argv.mode == 'image'and argv.filepath != '':
        CatchPICByFileName(argv.filepath)
    elif argv.mode == 'imageset':
        test_by_image_set()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test", description='test argparse')
    parser.add_argument('--mode','-m',type=str, help='mode 运行模式, 必要参数')
    parser.add_argument('--filepath', '-f', type=str, help='filepath 图片路径, 当mode为image时必要')
    args = parser.parse_args()
    main(args)
    # CatchPICFromVideo("Get Face", 0)
    # run_on_img('./CelebA/Img/img_align_celeba/202591.jpg')
    # test_by_image()


import cv2


# 对图像进行放缩

# 直接指定使用元组指定新图像的尺寸
def resize_img(image, weight, height):
    resImg = cv2.resize(image, (weight, height), interpolation=cv2.INTER_CUBIC)
    return resImg


# resize函数的输入：
# 源图像，img
# (weight,height)是变换后的图片大小，分别是图片宽和高。注意用元组的形式
# 变换的算法，interpolation就是表示插补的意思，因为变换不同的尺寸就会增减像素
# ITER_CUBIC是一种方法

if __name__ == "__main__":
    img = cv2.imread('../agedb_30_masked/0.jpg')
    print(type(img))
    cv2.imshow('img', img)
    cv2.imshow('resImg', resize_img(img, 300, 300))
    cv2.waitKey()
    cv2.destroyAllWindows()

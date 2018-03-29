
# 自定义常用功能函数
___

read_images	读取文件夹下的图像数据，返回np.array;

detect_face 输入图片，返回人脸头像区域图片，原图中头像区域的坐标,cv2.rectangle(origin_image, (x, y), (x+w, y+h), (0, 255, 0), 2);

align_images 找两张图像的特征点，生成齐次矩阵，im_aligned对齐im_reference;





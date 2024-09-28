import os
import time
import numpy as np
from torchcam.utils import overlay_mask
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as transforms
import torch
from PIL import Image
import visdom

tf = transforms.Compose([
    lambda x: Image.open(x).convert("RGB"),  # string path =>image data
    # transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),  # 仅放缩，没有剪裁；若只有一个参数则是把短边放缩到指定值
    # transforms.RandomRotation(15),  # 图片在15度范围内随机旋转，不要设置太大，不然运算量太大，网络变复杂
    # transforms.CenterCrop(self.resize),  # 中心剪裁
    transforms.ToTensor(),  # 会将各个像素值除以225进行归一化
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])  # 此处的均值方差是行业标准，统计的ImageNet的海量数据得到的
])


def gan_grad_cam(model, img_path, save_path, target_layers, fc_num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = tf(img_path).unsqueeze(0).to(device)
    cam = GradCAM(model=model, target_layers=target_layers)

    # 下面的代码是生成grayscale_cams的过程，并且保存下来备用
    grayscale_cams = []
    # 这个101是FC层的维度，是最后要分类为多少层
    for i in range(fc_num):
        targets = [ClassifierOutputTarget(i)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        grayscale_cams.append(grayscale_cam)

        # plt.imshow(grayscale_cam)
        # plt.title("Grayscale Cam：{}".format(i))
        # plt.show()

    np.save(save_path, grayscale_cams)
    print("npy文件 生成完毕！")


# 把img_path和grad_cams_save_path输入，输出所有的组合tensor，[b, c, h, w]
def visualize_grad_cam(img_path, np_save_path, img_save_path, channel=0):
    grayscales_cams = np.load(np_save_path)
    transform = transforms.ToTensor()
    img_pil = Image.open(img_path)
    visualizations = []
    for i in range(len(grayscales_cams)):
        visualization = overlay_mask(img_pil, Image.fromarray(grayscales_cams[i]), alpha=0.5)
        visualizations.append(visualization)


    # 使用 save 方法保存图像
    # 第一个参数是文件名，第二个参数是文件格式
    # save_channel_path = "/home/leishu/datasets/myocc_datasets/test/可用的/img1/" + save_path.split(os.sep)[-1].split('.')[0] + "_"+ str(channel) + ".png"
    save_channel_path = img_save_path
    print(save_channel_path)
    visualizations[channel].save(save_channel_path, 'PNG')

    visualizations_tensor = [transform(image1) for image1 in visualizations]
    visualizations_tensor = torch.stack(visualizations_tensor)

    return visualizations_tensor


def main(img_input_path, model, img_save_path, target_layers, channel=0):

    viz = visdom.Visdom()
    # 选择输入图片
    # img_input_path = "/home/leishu/datasets/myocc_datasets/test/可用的/5.jpg"
    # 会把所有通道的类激活图以npy文件的形式保存下来备用，此参数不需要改。
    grad_cam_save_path = "./matplotlib/" + img_input_path.split(os.sep)[-1].split('.')[0] + "_gray scales_cams_mask.npy"
    # 选择模型文件，此模型需要含参数和模型结构，要求单此pth文件就能加载整个模型
    # model_path = '../MegafaceEvaluate/model/model_p4_baseline_9938_8205_3610_原来1.pth'
    # 选定特定的通道，保存该通道的图片, 参数channel最大值为选定的对应模型的对应卷积层的通道数减一，
    channel = channel
    # img_save_path = img_save_path + "/" + img_input_path.split(os.sep)[-1].split('.')[0] + "_channel_" + str(channel) + ".png"
    img_save_path = os.path.join(img_save_path, img_input_path.split(os.sep)[-1].split('.')[0] + "_channel_" + str(channel) + ".png")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    # # 加载模型
    # model = torch.load(model_path).to(device)
    # model.eval()
    # print(model)
    # # 确定目标层次,根据模型确定层次，选择卷积层
    # target_layers = target_layers
    # # target_layers = [model.regress]


    # 保存grad_cam.npy文件
    gan_grad_cam(model, img_input_path, grad_cam_save_path, target_layers, 512)
    # 生成可视化热力图
    visualizations_tensor = visualize_grad_cam(img_input_path, grad_cam_save_path, img_save_path, channel)
    title = img_input_path.split(os.sep)[-1].split('.')[0] + "_" + grad_cam_save_path.split(os.sep)[-1].split('.')[0]

    viz.images(visualizations_tensor, nrow=20, win="visualizations", opts=dict(title=title))
    #
    # #保存某个热力图
    # grayscales_cams = np.load(grad_cam_save_path)
    # transform = transforms.ToTensor()
    # img_pil = Image.open(img_input_path)
    # visualizations = []
    # for i in range(len(grayscales_cams)):
    #     visualization = overlay_mask(img_pil, Image.fromarray(grayscales_cams[i]), alpha=0.5)
    #     visualizations.append(visualization)
    #
    # visualizations_tensor = [transform(image1) for image1 in visualizations]
    # visualizations_tensor = torch.stack(visualizations_tensor)
    #
    # return visualizations_tensor

    print("运行结束！")
    time.sleep(6000)


if __name__ == "__main__":
    '''
    img_input_path  选择输入图片
    model_path  选择模型文件，此模型需要含参数和模型结构，要求单此pth文件就能加载整个模型
    img_save_path  图片的保存目录
    channel  选定特定的通道，保存该通道的图片, 参数channel最大值为选定的对应模型的对应卷积层的通道数减一
    target_layers 确定目标层次,根据模型确定层次，选择卷积层
    只需要设置这五个参数
    
    然后安装visdom这个包,在代码运行前,在当前的python环境下启动:python -m visdom.server
    最后再运行此脚本
    '''
    # 选择输入图片
    img_input_path = "/home/leishu/datasets/myocc_datasets/test/可用的/5.jpg"
    # 选择模型文件，此模型需要含参数和模型结构，要求单此pth文件就能加载整个模型
    model_path = '../MegafaceEvaluate/model/model_p4_baseline_9938_8205_3610_原来1.pth'
    # 选定特定的通道，保存该通道的图片, 参数channel最大值为选定的对应模型的对应卷积层的通道数减一，
    channel = 0
    # 图片的保存目录
    img_save_path = "./img_save"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # 加载模型
    model = torch.load(model_path).to(device)
    model.eval()
    print(model)
    # 确定目标层次,根据模型确定层次，选择卷积层
    target_layers = [model.layer4[-1]]

    main(img_input_path, model, img_save_path, target_layers, channel)

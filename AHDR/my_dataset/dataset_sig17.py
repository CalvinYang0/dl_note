from torch.utils.data import Dataset
import os.path as osp
import sys
import os
import glob
import numpy as np
import cv2
import imageio
import torch
imageio.plugins.freeimage.download()

def ldr_to_hdr(image, exposure, gamma):
    return (image ** gamma) / (exposure + 1e-8)


def read_exposure_times(file_path):
    # 将文档以np的形式读取
    exposure_times = (np.loadtxt(file_path))
    # 转换为2次方的形式
    return np.power(2, exposure_times)


def read_images(file_path):
    images = []
    for image_path in file_path:
        # 读取图像
        img = cv2.imread(image_path, -1)
        # 对图像进行一些预处理
        img = img / (2 ** 16)
        img = np.float32(img)
        img.clip(0, 1)
        images.append(img)
    return np.array(images)


def list_all_images(folder_path, extension):
    # 读取文件夹下所有的图像,glob.glob()返回所有匹配的文件路径列表
    return sorted(glob.glob(osp.join(folder_path, '*' + extension)))


def read_label(file_path, file_name):
    # 读取标签
    label = imageio.imread(osp.join(file_path, file_name), 'hdr')
    # 返回长宽rgb，按照列表顺序读取第三个维度
    label = label[:, :, [2, 1, 0]]
    return label


class SIG17_Training_Dataset(Dataset):
    """
    用于加载SIG17数据集
    """

    def __init__(self, root_dir, sub_set, is_training=True):
        scenes_dir = osp.join(root_dir, sub_set)
        scenes_list = sorted(os.listdir(scenes_dir))
        self.image_list = []
        for scene_num in range(len(scenes_list)):
            exposure_file_path = osp.join(scenes_dir, scenes_list[scene_num], 'exposure.txt')
            ldrs_path = list_all_images(osp.join(scenes_dir, scenes_list[scene_num]), '.tif')
            lable_path = osp.join(scenes_dir, scenes_list[scene_num])
            self.image_list += [[exposure_file_path, ldrs_path, lable_path]]

    def __getitem__(self, index):
        # 读取曝光时间
        exposure_times = read_exposure_times(self.image_list[index][0])
        # 读取ldr图像
        ldr_images = read_images(self.image_list[index][1])
        # 读取hdr图像
        label = read_label(self.image_list[index][2], 'label.hdr')
        # 对ldr图像进行处理，转换为hdr
        pre_image0 = ldr_to_hdr(ldr_images[0], exposure_times[0], 2.2)
        pre_image1 = ldr_to_hdr(ldr_images[1], exposure_times[1], 2.2)
        pre_image2 = ldr_to_hdr(ldr_images[2], exposure_times[2], 2.2)

        # 将ldr图像和hdr图像合并
        pre_image0 = np.concatenate((pre_image0, ldr_images[0]), 2)
        pre_image1 = np.concatenate((pre_image1, ldr_images[1]), 2)
        pre_image2 = np.concatenate((pre_image2, ldr_images[2]), 2)
        # 读进来的图像的通道数在第三维，需要提前到第一维
        img0 = pre_image0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_image1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_image2.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)
        # 转为tensor类型
        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {
            'input0': img0,
            'input1': img1,
            'input2': img2,
            'label': label
        }
        return sample

    def __len__(self):
        return len(self.image_list)


class SIG17_Validation_Dataset(Dataset):
    """
    加载SIG17测试集
    """

    def __init__(self, root_dir, is_training=False, crop=True, crop_size=512):
        self.root_dir = root_dir
        self.is_training = is_training
        # 是否进行裁剪
        self.crop = crop
        self.crop_size = crop_size

        self.scenes_dir = osp.join(root_dir, 'Test')
        self.scenes_list = sorted(os.listdir(self.scenes_dir))
        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_images(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        expoTimes = read_exposure_times(self.image_list[index][0])
        ldr_images = read_images(self.image_list[index][1])
        label = read_label(self.image_list[index][2], 'HDRImg.hdr')  # 'HDRImg.hdr' for test data
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)
        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)
        # 是否需要剪裁
        if self.crop:
            # 如果需要裁剪
            x = 0
            y = 0
            # 将图像的大小限制在512*512以内
            img0 = pre_img0[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            label = label[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
        else:
            img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
            label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {
            'input0': img0,
            'input1': img1,
            'input2': img2,
            'label': label
        }
        return sample

    def __len__(self):
        return len(self.scenes_list)


class Img_Dataset(Dataset):
    def __init__(self, ldr_path, label_path, exposure_path, patch_size):
        self.ldr_images = read_images(ldr_path)
        self.label = read_label(label_path, 'HDRImg.hdr')
        self.ldr_patches = self.get_ordered_patches(patch_size)
        self.expo_times = read_exposure_times(exposure_path)
        self.patch_size = patch_size
        self.result = []

    def __getitem__(self, index):
        pre_img0 = ldr_to_hdr(self.ldr_patches[index][0], self.expo_times[0], 2.2)
        pre_img1 = ldr_to_hdr(self.ldr_patches[index][1], self.expo_times[1], 2.2)
        pre_img2 = ldr_to_hdr(self.ldr_patches[index][2], self.expo_times[2], 2.2)
        pre_img0 = np.concatenate((pre_img0, self.ldr_patches[index][0]), 2)
        pre_img1 = np.concatenate((pre_img1, self.ldr_patches[index][1]), 2)
        pre_img2 = np.concatenate((pre_img2, self.ldr_patches[index][2]), 2)
        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        sample = {
            'input0': img0,
            'input1': img1,
            'input2': img2}
        return sample

    def get_ordered_patches(self, patch_size):
        ldr_patch_list = []
        # h,w,c为图像的高，宽，通道数
        h, w, c = self.label.shape
        # tmp_h,tmp_w,将图像的大小限制在patch_size以内
        n_h = h // patch_size + 1
        n_w = w // patch_size + 1
        tmp_h = n_h * patch_size
        tmp_w = n_w * patch_size
        # 如果图像的大小小于patch_size，则将图像的大小扩充到patch_size
        tmp_label = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr0 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr1 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr2 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_label[:h, :w] = self.label
        tmp_ldr0[:h, :w] = self.ldr_images[0]
        tmp_ldr1[:h, :w] = self.ldr_images[1]
        tmp_ldr2[:h, :w] = self.ldr_images[2]
        # 将图像分割成patch_size*patch_size的小块
        for x in range(n_w):
            for y in range(n_h):
                if (x + 1) * patch_size <= tmp_w and (y + 1) * patch_size <= tmp_h:
                    temp_patch_ldr0 = tmp_ldr0[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size]
                    temp_patch_ldr1 = tmp_ldr1[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size]
                    temp_patch_ldr2 = tmp_ldr2[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size]
                    ldr_patch_list.append([temp_patch_ldr0, temp_patch_ldr1, temp_patch_ldr2])
        # 如果ldr_patch_list的长度不等于n_h*n_w，则报错
        assert len(ldr_patch_list) == n_h * n_w
        return ldr_patch_list

    def __len__(self):
        return len(self.ldr_patches)
    def rebuild_result(self):
        #将图像的大小扩充到patch_size的整数倍
        h, w, c = self.label.shape
        n_h = h // self.patch_size + 1
        n_w = w // self.patch_size + 1
        tmp_h = n_h * self.patch_size
        tmp_w = n_w * self.patch_size
        pred = np.empty((c, tmp_h, tmp_w), dtype=np.float32)
        #将pred的值用分块推导的结果填充
        for x in range(n_w):
            for y in range(n_h):
                pred[:, y*self.patch_size:(y+1)*self.patch_size, x*self.patch_size:(x+1)*self.patch_size] = self.result[x*n_h+y]
        #仅选用lable的大小的图像
        return pred[:, :h, :w], self.label.transpose(2, 0, 1)

    def update_result(self, tensor):
        self.result.append(tensor)

def SIG17_Test_Dateset(root_dir, patch_size):
    scene_dir = osp.join(root_dir, 'Test')
    scenes_list = sorted(os.listdir(scene_dir))
    ldr_list = []
    label_list = []
    expo_time_list = []
    for scene in range(len(scenes_list)):
        exposure_file_path = os.path.join(scene_dir, scenes_list[scene], 'exposure.txt')
        ldr_file_path = list_all_images(os.path.join(scene_dir, scenes_list[scene]), '.tif')
        label_path = os.path.join(scene_dir, scenes_list[scene])
        ldr_list.append(ldr_file_path)
        label_list.append(label_path)
        expo_time_list.append(exposure_file_path)
    for ldr_dir, label_dir, expo_time_dir in zip(ldr_list, label_list, expo_time_list):
        yield Img_Dataset(ldr_dir, label_dir, expo_time_dir, patch_size)

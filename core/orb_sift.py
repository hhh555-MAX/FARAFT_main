import cv2
import numpy as np
import torch.nn as nn
import torch
from utils_flow.flow_and_mapping_operations import convert_mapping_to_flow
from utils_data.geometric_transformation_sampling.aff_homo_tps_generation import HomographyGridGen

class ORB(nn.Module):
    def __init__(self, args):
        super(ORB, self).__init__()


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, flow_gt=None, test_mode=False):
        image1 = image1.permute(0, 2, 3, 1).squeeze(0)
        image2 = image2.permute(0, 2, 3, 1).squeeze(0)
        image1, image2 = image1.cpu().numpy().astype(np.uint8), image2.cpu().numpy().astype(np.uint8)
        H, W, C = image1.shape
        self.homo_grid_sample = HomographyGridGen(out_h=H, out_w=W, normalized=False, use_cuda=False)
        M, warped_img, img_matches = orb(image1, image2)
        theta_hom = torch.Tensor(M.astype(np.float32)).unsqueeze(0).reshape(1, 9)
        mapping = self.homo_grid_sample.forward(theta_hom)
        flow_est = convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 1, 2,h,w
        flow_est = flow_est.cuda()
        return {'final': flow_est, 'flow':[flow_est]}
    
class SIFT(nn.Module):
    def __init__(self, args):
        super(SIFT, self).__init__()


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, flow_gt=None, test_mode=False):
        image1 = image1.permute(0, 2, 3, 1).squeeze(0)
        image2 = image2.permute(0, 2, 3, 1).squeeze(0)
        image1, image2 = image1.cpu().numpy().astype(np.uint8), image2.cpu().numpy().astype(np.uint8)
        H, W, C = image1.shape
        self.homo_grid_sample = HomographyGridGen(out_h=H, out_w=W, normalized=False, use_cuda=False)
        M, warped_img, img_matches = sift(image1, image2)
        theta_hom = torch.Tensor(M.astype(np.float32)).unsqueeze(0).reshape(1, 9)
        mapping = self.homo_grid_sample.forward(theta_hom)
        flow_est = convert_mapping_to_flow(mapping, output_channel_first=True)  # should be 1, 2,h,w
        flow_est = flow_est.cuda()
        return {'final': flow_est, 'flow':[flow_est]}

    

def orb(image1, image2):
    # 转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # print(type(gray1))
    # 使用ORB检测器检测特征点和描述符
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # 使用BFMatcher进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 根据特征点匹配结果，选取最好的匹配点
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:1000]  # 选取前20个最好的匹配点


    # 绘制匹配结果
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)

    # 提取匹配点的坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算仿射变换矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w, _ = image1.shape
    warped_img = cv2.warpPerspective(image2, M, (w, h))
    # 展示结果
    return M, warped_img, img_matches

def sift(image1, image2):
    # 转换为灰度图像
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    # sift = cv2.ORB_create()
    
    # 使用SIFT找到关键点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    # 使用FLANN matcher进行匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
    search_params = dict(checks=10)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)
    
    # 找到匹配点的位置
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 计算仿射变换矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 应用仿射变换
    h, w = img1.shape
    warped_img = cv2.warpPerspective(image2, M, (w, h))    
    return M, warped_img, img_matches


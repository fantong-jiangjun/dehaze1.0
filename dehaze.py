# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse


def dark_channel(img, size = 15):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img,kernel)
    return dc_img


# 用暗通道最高区域估计三通道大气光，避免偏色
def get_atmo(img, top_percent=0.001, size=15):
    h, w, _ = img.shape
    dc = dark_channel(img, size)
    n = max(1, int(h * w * top_percent))

    dc_flat = dc.reshape(-1)
    top_idx = np.argsort(dc_flat)[-n:]
    I_flat = img.reshape(-1, 3)
    brightest_idx = top_idx[np.argmax(I_flat[top_idx].sum(axis=1))]
    A = I_flat[brightest_idx]  # (3,)
    return A


# 使用三通道A按通道归一化，温和的w降低过度去雾
def get_trans(img, A, w = 0.85, size=21):
    x = img / A
    t = 1 - w * dark_channel(x, size)
    return np.clip(t, 0.0, 1.0)


def guided_filter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """
    #1
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    #2
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    #3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    #4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    #5
    q = mean_a * i + mean_b
    return q


def refine_transmission(t, guide_bgr, r=40, eps=1e-3):
    """优先用 ximgproc 彩色引导滤波；无则回退到灰度 guided_filter"""
    try:
        if hasattr(cv2, 'ximgproc') and cv2.ximgproc is not None:
            # ximgproc 接受 float32
            guide = guide_bgr.astype(np.float32)
            src = t.astype(np.float32)
            t_ref = cv2.ximgproc.guidedFilter(guide, src, r, eps)
            return t_ref.astype(np.float64)
    except Exception:
        pass
    # 回退：灰度引导
    gray = cv2.cvtColor((guide_bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    return guided_filter(t, gray, r, eps)


def gray_world_white_balance(img):
    """简单灰世界白平衡，缓解整体偏色"""
    mean_rgb = img.reshape(-1, 3).mean(axis=0) + 1e-8
    scale = mean_rgb.mean() / mean_rgb
    balanced = img * scale
    return np.clip(balanced, 0.0, 1.0)


def dehaze(path, output = None):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    img = im.astype('float64') / 255

    # 1) A 与初始 t
    A = get_atmo(img, top_percent=0.001, size=15)
    t = get_trans(img, A, w=0.85, size=21)

    # 2) 轻度平滑 + 引导滤波细化，减少块效应与噪声
    t = cv2.bilateralFilter(t.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=15).astype(np.float64)
    t = refine_transmission(t, (img * 255).astype(np.uint8), r=40, eps=1e-3)

    # 3) 下限更低更自然
    t0 = 0.06
    t = np.maximum(t, t0)

    # 4) 按通道重建
    result = (img - A) / t[:, :, np.newaxis] + A
    result = np.clip(result, 0.0, 1.0)

    # 5) 基于透射率的自适应融合，减轻边缘光晕/过强对比
    alpha = (t - t0) / (1.0 - t0)
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, np.newaxis]
    result = result * alpha + img * (1.0 - alpha)

    # 6) 轻度白平衡
    result = gray_world_white_balance(result)

    # 展示（保存行为由 -o 参数控制）
    result_u8 = (result * 255).astype(np.uint8)
    cv2.imshow("source",im)
    cv2.imshow("result", result_u8)
    cv2.waitKey()

    # 仅当提供 -o 时保存（保持你左侧 output.png 的工作方式）
    if output is not None:
        cv2.imwrite(output, result_u8)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
args = parser.parse_args()


if __name__ == '__main__':
    if args.input is None:
        dehaze('image/canon3.bmp')
    else:
        dehaze(args.input, args.output)
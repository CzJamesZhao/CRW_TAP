import logging
import sys
from collections import OrderedDict
from types import SimpleNamespace

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from dgl.geometry import farthest_point_sampler
from itertools import repeat
from tqdm import tqdm

from utils.metrics import end_point_error

# def farthest_point_sampler(points, num_samples):
#     """
#     朴素的最远点采样实现，用于替代 dgl 的 farthest_point_sampler。

#     参数:
#         points (np.ndarray): 输入点集，形状为 (1, N, 2)，N 是点的数量。
#         num_samples (int): 需要采样的点的数量。

#     返回:
#         np.ndarray: 采样点的索引，形状为 (num_samples,)。
#     """
#     points = points.squeeze(0)  # 去除第一个维度，形状变为 (N, 2)
#     N = points.shape[0]
    
#     if num_samples >= N:
#         return np.arange(N)  # 如果采样数大于等于点数，直接返回所有点的索引
    
#     selected_indices = [np.random.randint(0, N)]  # 随机选择一个初始点
#     distances = np.linalg.norm(points - points[selected_indices[0]], axis=1)

#     for _ in range(1, num_samples):
#         farthest_index = np.argmax(distances)  # 找到最远的点
#         selected_indices.append(farthest_index)
#         new_distances = np.linalg.norm(points - points[farthest_index], axis=1)
#         distances = np.minimum(distances, new_distances)  # 更新距离

#     return np.array(selected_indices)


# logger = logging.getLogger(__name__)


def strnum(x):
    s = "%g" % x
    if "." in s:
        if x < 1.0:
            s = s[s.index(".") :]
    return s


def save_col_major_nd_plot(images, titles, file_path):
    assert len(images) == len(titles)

    num_cols = len(images)
    num_rows = 1
    for i in images:
        if type(i) == list:
            num_rows = max(num_rows, len(i))

    fig, axs = plt.subplots(
        figsize=(int(2 * num_cols), int(2 * num_rows)), ncols=num_cols, nrows=num_rows
    )
    fig.tight_layout(pad=3.0)

    for j in range(num_cols):
        if type(images[j]) == list:
            for i in range(len(images[j])):
                if type(images[j][i]) == tuple and images[j][i][1]:
                    im1 = axs[i][j].imshow(images[j][i][0], vmin=0, vmax=1)
                    divider = make_axes_locatable(axs[i][j])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im1, cax=cax, orientation="vertical")
                else:
                    axs[i][j].imshow(images[j][i])
                axs[i][j].set(title=titles[j][i])
        else:
            axs[0][j].imshow(images[j])
            axs[0][j].set(title=titles[j])

    plt.savefig(file_path, bbox_inches="tight", dpi=200)
    plt.close()


def save_row_major_nd_plot(images, titles, file_path):
    assert len(images) == len(titles)

    num_rows = len(images)
    num_cols = 1
    for i in images:
        if type(i) == list:
            num_cols = max(num_cols, len(i))

    if isinstance(images[0], list):
        img_size_h = images[0][0].shape[0]
        img_size_w = images[0][0].shape[1]
    else:
        img_size_h = images[0].shape[0]
        img_size_w = images[0].shape[1]

    fig_size_h = img_size_h * num_rows / 100 + 1
    fig_size_w = img_size_w * num_cols / 100

    fig, axs = plt.subplots(
        figsize=(int(fig_size_w), int(fig_size_h)), ncols=num_cols, nrows=num_rows
    )
    fig.tight_layout(pad=1.0)

    for i in range(num_rows):
        if type(images[i]) == list:
            for j in range(len(images[i])):
                if type(images[i][j]) == tuple:
                    image_to_plot = images[i][j][0]
                    plot_it = False
                    if len(images[i][j]) > 1:
                        if images[i][j][1]:
                            plot_it = True
                    if plot_it:
                        if len(images[i][j]) == 3:
                            vmin = images[i][j][2][0]
                            vmax = images[i][j][2][1]
                        else:
                            vmin = 0
                            vmax = 1

                        im1 = axs[i][j].imshow(image_to_plot, vmin=vmin, vmax=vmax)
                        divider = make_axes_locatable(axs[i][j])
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        fig.colorbar(im1, cax=cax, orientation="vertical")
                        axs[i][j].set(title=titles[i][j])
                else:
                    axs[i][j].imshow(images[i][j])
                    axs[i][j].set(title=titles[i][j])
            for j in range(len(images[i]), num_cols):
                fig.delaxes(axs[i][j])

        else:
            axs[i][0].imshow(images[i])
            axs[i][0].set(title=titles[i])
            for j in range(1, num_cols):
                fig.delaxes(axs[i][j])

    plt.savefig(file_path, bbox_inches="tight", dpi=240)
    plt.close()


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip flow values to [-clip_flow, clip_flow]. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def draw_trajs_on_rgbs(
    rgbs,
    trajs,
    visibilities=None,
    valids=None,
    texts=None,
    show_dots=True,
    cmap="coolwarm",
    linewidth=1,
    num_samples=None, # 采样轨迹数量
):
    # rgbs: (S, H, W, C), numpy  形状为 (S, H, W, C) 的 NumPy 数组，表示一系列帧。
    # trajs: (S, N, 2), numpy (long) 形状为 (S, N, 2) 的 NumPy 数组，表示 N 个点在 S 帧中的轨迹。
    # valids: (S, N) 
    S, H, W, C = rgbs.shape
    S2, N, D = trajs.shape
    assert S == S2 # 轨迹的长度和视频的长度应该是一致的

    if visibilities is None: # 如果没有可见性，就默认全1
        visibilities = np.ones_like(trajs[:, :, 0])  # S, N

    if valids is None: # 如果没有valid，就默认全1
        valids = np.ones_like(trajs[:, :, 0])  # S, N

    rgbs_color = rgbs

    # 全部轨迹都展示太密集，选择性的画
    # 使用最远采样点选择轨迹
    if num_samples is None or num_samples > N:
        selected_trajs = max(1, N // 8)
        # 随机采样
        # selected_trajs = np.random.choice(N, N // 4, replace=False)
    else:
        selected_trajs = num_samples
        # 随机采样
        # selected_trajs = np.random.choice(N, num_samples, replace=False)

    sample_indices = farthest_point_sampler(torch.from_numpy(trajs[0:1]).float(),selected_trajs) # 输入维度是（B，N，C）批次，数量，维度，[0:1]的目的是为了留住第一个维度
    sample_indices = sample_indices[0].cpu().numpy()
    # 随机采样
    # sample_indices = selected_trajs


    # for i in range(N): # 一个一个轨迹地画
    for i in sample_indices:
        if cmap == "onediff" and i == 0:
            cmap_ = "spring"
        elif cmap == "onediff":
            cmap_ = "winter"
        else:
            cmap_ = cmap
        traj = trajs[:, i]  # S, 2
        valid = valids[:, i]  # S
        visibility = visibilities[:, i]  # S

        for t in range(S):
            if valid[t]:
                rgbs_color[t] = draw_traj_on_image_py(
                    rgbs_color[t],
                    traj[: t + 1],
                    visibilities=visibility[: t + 1], 
                    S=S,
                    show_dots=show_dots,
                    cmap=cmap_,
                    linewidth=linewidth,
                )

    # (S, H, W, C)
    rgbs = rgbs_color
    if texts is not None:
        len(texts) == S
        for t in range(S):
            rgbs_color[t] = draw_text_on_img(rgbs_color[t], texts[t])
    return rgbs_color


def draw_trajs_on_single_rgb(
    rgb,
    trajs,
    valids=None,
    text=None,
    show_dots=True,
    cmap="coolwarm",
    linewidth=1,
):
    # rgb: (H, W, C), numpy
    # trajs: (S, N, 2), numpy (long)
    # valids: (S, N)

    H, W, C = rgb.shape
    S, N, D = trajs.shape

    if valids is None:
        valids = np.ones_like(trajs[:, :, 0])  # S, N

    # using maxdist will dampen the colors for short motions
    # norms = torch.sqrt(1e-4 + torch.sum((trajs[-1] - trajs[0]) ** 2, dim=1))  # N
    # maxdist = torch.quantile(norms, 0.95).detach().cpu().numpy()
    maxdist = None

    rgb_color = rgb

    for i in range(N):
        if cmap == "onediff" and i == 0:
            cmap_ = "spring"
        elif cmap == "onediff":
            cmap_ = "winter"
        else:
            cmap_ = cmap
        traj = trajs[:, i]  # S, 2
        valid = valids[:, i]  # S

        if valid[0] == 1:
            traj = traj[valid > 0]
            rgb_color = draw_traj_on_image_py(
                rgb_color,
                traj,
                S=S,
                show_dots=show_dots,
                cmap=cmap_,
                maxdist=maxdist,
                linewidth=linewidth,
            )

    # (S, H, W, C)
    rgb = rgb_color
    if text is not None:
        rgb_color = draw_text_on_img(rgb_color, text)
    return rgb_color


def draw_traj_on_image_py(
    rgb,
    traj,
    visibilities=None,
    S=50,
    linewidth=1,
    show_dots=False,
    cmap="coolwarm",
    maxdist=None,
):
    # all inputs are numpy tensors
    # rgb is 3 x H x W
    # traj is S x 2

    H, W, C = rgb.shape
    assert C == 3

    rgb = rgb.astype(np.uint8).copy()

    S1, D = traj.shape
    assert D == 2

    color_map = cm.get_cmap(cmap)
    S1, D = traj.shape

    for s in range(S1 - 1):
        if maxdist is not None:
            val = (np.sqrt(np.sum((traj[s] - traj[0]) ** 2)) / maxdist).clip(0, 1)
            color = np.array(color_map(val)[:3]) * 255  # rgb
        else:
            # 根据轨迹的每个段的位置，动态生成颜色值。每个轨迹段的位置越靠后，其颜色值会不同
            color = np.array(color_map((s) / max(1, float(S - 2)))[:3]) * 255  # rgb

        cv2.line(
            rgb,
            (int(traj[s, 0]), int(traj[s, 1])),
            (int(traj[s + 1, 0]), int(traj[s + 1, 1])),
            color,
            linewidth,
            cv2.LINE_AA,
        )
        if show_dots:
            # 这里总报错，因为轨迹是浮点，不是整形，四舍五入round改为整形
            try:
                center = (round(traj[s, 0]), round(traj[s, 1]))  # 使用四舍五入
                # cv2.circle(rgb, (traj[s, 0], traj[s, 1]), linewidth, color, -1)
                cv2.circle(rgb, center, linewidth, color, -1)
            except:
                print(f"traj[s, 0]: {traj[s, 0]}, traj[s, 1]: {traj[s, 1]}")
                import ipdb; ipdb.set_trace()


        # if visibilities is not None:
        #     if visibilities[s].item() is False:
        #         cv2.circle(rgb, (traj[s, 0], traj[s, 1]), linewidth + 1, (255, 0, 0), 1)

    if maxdist is not None:
        val = (np.sqrt(np.sum((traj[-1] - traj[0]) ** 2)) / maxdist).clip(0, 1)
        color = np.array(color_map(val)[:3]) * 255  # rgb
    else:
        color = np.array(color_map((S1 - 1) / max(1, float(S - 2)))[:3]) * 255  # rgb

    # color = np.array(color_map(1.0)[:3]) * 255

    # 这里总报错，因为轨迹是浮点，不是整形，四舍五入round改为整形
    try:
        center = (round(traj[-1, 0]), round(traj[-1, 1]))  # 使用四舍五入
        # cv2.circle(rgb, (traj[-1, 0], traj[-1, 1]), linewidth * 2, color, -1)
        cv2.circle(rgb, center, linewidth * 2, color, -1)

    except:
        print(f"traj[-1, 0]: {traj[-1, 0]}, traj[-1, 1]: {traj[-1, 1]}")
        import ipdb; ipdb.set_trace()


    if visibilities is not None:
        for s in range(S1):
            if visibilities[s].item() in [False, 0]:
                # 这里总报错，因为轨迹是浮点，不是整形，四舍五入round改为整形
                try:
                    center = (round(traj[s, 0]), round(traj[s, 1]))  # 使用四舍五入
                    # cv2.circle(rgb, (traj[s, 0], traj[s, 1]), linewidth + 1, (255, 255, 255), 1)
                    cv2.circle(rgb, center, linewidth + 1, (255, 255, 255), 1)
                except:
                    print(f"traj[s, 0]:{traj[s, 0]}; traj[s, 1]:{traj[s, 1]}")
                    import ipdb; ipdb.set_trace()

    return rgb


def draw_text_on_img(image, text, scale=0.5, left=5, top=20):
    rgb = image
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = (255, 255, 255)
    # print('putting frame id', frame_id)

    frame_str = strnum(text)

    cv2.putText(
        rgb,
        frame_str,
        (left, top),  # from left, from top
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,  # font scale (float)
        color,
        1,
    )  # font thickness (int)
    rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return rgb


def visualize_kinetics(
    step,
    images,
    flows,
    overall_flows,
    args,
    viz_model=None,
    use_html=True,
    split="train",
):
    """
    images: (B, 2*T, C, H, W)
    flows: (B, T-1, 2, H, W)
    overall_flows: (B, T-1, 2, H, W)
    """
    MAX_POINTS_TO_SHOW = 500

    B, T, C, H, W = images.shape

    T = int(T / 2)

    save_path = f"{args.output_dir}/predictions/{split}"

    assert use_html is True

    html_dict = {"images": [], "texts": [], "links": []}
    html_gif_dict = {"images": [], "texts": [], "links": []}

    max_viz_per_batch = args.max_viz_per_batch

    if args.disable_transforms:
        viz_images = images
    else:
        viz_images = 255 * images

    forward_images = viz_images[:, :T]
    backward_images = viz_images[:, T:]

    # (B, T, H, W, 3)
    forward_images = (
        forward_images.permute(0, 1, 3, 4, 2).to(torch.uint8).detach().cpu().numpy()
    )
    backward_images = (
        backward_images.permute(0, 1, 3, 4, 2).to(torch.uint8).detach().cpu().numpy()
    )
    # (B, T-1, H, W, 2)
    flows = flows.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    overall_flows = overall_flows.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    for batch_idx in tqdm(range(min(B, max_viz_per_batch)), desc="Visualizing"):
        file_path = f"{save_path}/step{step}_batch{batch_idx}_all_tracks.png"
        html_png_path = f"step{step}_batch{batch_idx}_all_tracks.png"

        # (T, H, W, 3)
        forward_images_idx = forward_images[batch_idx]
        backward_images_idx = backward_images[batch_idx]

        # (2*T-1, H, W, 3)
        full_cycle_images = np.concatenate(
            [forward_images_idx, backward_images_idx[1:]], axis=0
        )

        # (T-1, H, W, 2)
        flows_idx = flows[batch_idx]
        overall_flows_idx = overall_flows[batch_idx]

        max_gt_flow_value = 255.0

        images_mean = np.mean(forward_images_idx, axis=0)

        output_images = []
        output_titles = []

        row_1 = []
        titles_1 = []

        for t in range(T):
            t_ = t + 1
            image_to_add = forward_images_idx[t]
            row_1.append(image_to_add)
            titles_1.append(f"F{t_}")

        for t in range(1, T):
            t_ = T - t
            image_to_add = backward_images_idx[t]
            row_1.append(image_to_add)
            titles_1.append(f"B{t_}")

        output_images.append(row_1)
        output_titles.append(titles_1)

        cycle_titles = []
        for t in range(2 * T - 2):
            if t <= T - 2:
                im_title = f"F{t+1}->F{t+2}"
            elif t == T - 1:
                im_title = f"F{t+1}->B{t}"
            else:
                t_ = 2 * T - t - 1
                im_title = f"B{t_}->B{t_-1}"
            cycle_titles.append(im_title)

        row_2 = []
        titles_2 = cycle_titles.copy()

        for t in range(2 * T - 2):
            flow_t = flows_idx[t]
            flow_t_image = flow_to_image(flow_t, clip_flow=max_gt_flow_value)
            row_2.append(flow_t_image)

        output_images.append(row_2)
        output_titles.append(titles_2)

        row_3 = []
        titles_3 = cycle_titles.copy()

        for t in range(2 * T - 2):
            flow_t = overall_flows_idx[t]
            flow_t_image = flow_to_image(flow_t, clip_flow=max_gt_flow_value)
            row_3.append(flow_t_image)
            titles_3[t] = f"Overall {titles_3[t]}"

        output_images.append(row_3)
        output_titles.append(titles_3)

        save_row_major_nd_plot(output_images, output_titles, file_path)

        html_dict["images"].append(html_png_path)
        html_dict["texts"].append(html_png_path)
        html_dict["links"].append(html_png_path)

    return html_dict


def visualize_kubric(
    step,
    images,
    flows,
    overall_flows,
    gt_flows,
    visible_flow_mask,
    occluded_flow_mask,
    trajs,
    gt_trajs,
    gt_visible_points,
    args,
    use_html=True,
    split="train",
):
    """
    images: (B, 2*T, C, H, W)
    flows: (B, T-1, 2, H, W)
    overall_flows: (B, T-1, 2, H, W)
    gt_flows: (B, T-1, 2, H, W)
    visible_flow_mask: (B, T-1, 1, H, W)
    occluded_flow_mask: (B, T-1, 1, H, W)
    trajs: (B, T, N, 2)
    gt_trajs: (B, T, N, 2)
    gt_visible_points: (B, T, N)
    """
    MAX_POINTS_TO_SHOW = 500

    B, T, C, H, W = images.shape

    T = int(T / 2)

    save_path = f"{args.output_dir}/predictions/{split}"

    assert use_html is True

    html_dict = {"images": [], "texts": [], "links": []}

    max_viz_per_batch = args.max_viz_per_batch

    if args.disable_transforms:
        viz_images = images
    else:
        viz_images = 255 * images

    forward_images = viz_images[:, :T]
    backward_images = viz_images[:, T:]

    # (B, T, H, W, 3)
    forward_images = (
        forward_images.permute(0, 1, 3, 4, 2).to(torch.uint8).detach().cpu().numpy()
    )
    backward_images = (
        backward_images.permute(0, 1, 3, 4, 2).to(torch.uint8).detach().cpu().numpy()
    )
    # (B, T-1, H, W, 2)
    flows = flows.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    overall_flows = overall_flows.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    gt_flows = gt_flows.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    # visible_flow_mask: (B, T-1, H, W)
    visible_flow_mask = visible_flow_mask[:, :, 0].detach().cpu().numpy()
    occluded_flow_mask = occluded_flow_mask[:, :, 0].detach().cpu().numpy()

    for batch_idx in tqdm(range(min(B, max_viz_per_batch)), desc="Visualizing"):
        file_path = f"{save_path}/step{step}_batch{batch_idx}_all_tracks.png"
        html_png_path = f"step{step}_batch{batch_idx}_all_tracks.png"

        # (T, H, W, 3)
        forward_images_idx = forward_images[batch_idx]
        backward_images_idx = backward_images[batch_idx]

        # (2*T-1, H, W, 3)
        full_cycle_images = np.concatenate(
            [forward_images_idx, backward_images_idx[1:]], axis=0
        )

        # (T-1, H, W, 2)
        flows_idx = flows[batch_idx]
        overall_flows_idx = overall_flows[batch_idx]
        # (T-1, H, W, 2)
        gt_flows_idx = gt_flows[batch_idx]

        # (T-1, H, W)
        visible_flow_mask_idx = visible_flow_mask[batch_idx]
        occluded_flow_mask_idx = occluded_flow_mask[batch_idx]

        # (T, N, 2)
        trajs_idx = trajs[batch_idx]
        # (T, N, 2)
        gt_trajs_idx = gt_trajs[batch_idx]
        # (T, N)
        gt_visible_points_idx = gt_visible_points[batch_idx]

        if gt_trajs_idx.shape[1] > MAX_POINTS_TO_SHOW:
            # inds = farthest_point_sampler(gt_trajs_idx[0:1], MAX_POINTS_TO_SHOW)[0]
            inds = farthest_point_sampler(gt_trajs_idx[0:1].cpu(), MAX_POINTS_TO_SHOW)[0]
            inds.to('cuda')
            trajs_idx = trajs_idx[:, inds]
            gt_trajs_idx = gt_trajs_idx[:, inds]
            gt_visible_points_idx = gt_visible_points_idx[:, inds]

        trajs_idx = trajs_idx.long().detach().cpu().numpy()
        gt_trajs_idx = gt_trajs_idx.long().detach().cpu().numpy()

        max_gt_flow_value = np.max(np.abs(gt_flows_idx))

        images_mean = np.mean(forward_images_idx, axis=0)

        pred_tracks = draw_trajs_on_rgbs(
            full_cycle_images.copy(),
            trajs_idx,
            # visibilities=gt_visible_points_idx,
            cmap="spring",
            linewidth=1,
        )
        gt_tracks = draw_trajs_on_rgbs(
            forward_images_idx.copy(),
            gt_trajs_idx,
            visibilities=gt_visible_points_idx,
            cmap="winter",
            linewidth=1,
        )
        avg_im_tracks = draw_trajs_on_single_rgb(
            images_mean.copy(), trajs_idx, cmap="spring", linewidth=1
        )
        avg_im_tracks = draw_trajs_on_single_rgb(
            avg_im_tracks, gt_trajs_idx, cmap="winter", linewidth=1
        )

        output_images = []
        output_titles = []

        row_1 = []
        titles_1 = []

        for t in range(T):
            t_ = t + 1
            image_to_add = forward_images_idx[t]
            row_1.append(image_to_add)
            titles_1.append(f"F{t_}")

        for t in range(1, T):
            t_ = T - t
            image_to_add = backward_images_idx[t]
            row_1.append(image_to_add)
            titles_1.append(f"B{t_}")

        output_images.append(row_1)
        output_titles.append(titles_1)

        cycle_titles = []
        for t in range(2 * T - 2):
            if t <= T - 2:
                im_title = f"F{t+1}->F{t+2}"
            elif t == T - 1:
                im_title = f"F{t+1}->B{t}"
            else:
                t_ = 2 * T - t - 1
                im_title = f"B{t_}->B{t_-1}"
            cycle_titles.append(im_title)

        row_2 = []
        titles_2 = cycle_titles.copy()

        for t in range(2 * T - 2):
            flow_t = flows_idx[t]
            flow_t_image = flow_to_image(flow_t, clip_flow=max_gt_flow_value)
            row_2.append(flow_t_image)

        row_2.append(avg_im_tracks)
        titles_2.append(f"Tracks 1 to {T}")

        output_images.append(row_2)
        output_titles.append(titles_2)

        row_3 = []
        titles_3 = cycle_titles.copy()

        for t in range(2 * T - 2):
            flow_t = overall_flows_idx[t]
            flow_t_image = flow_to_image(flow_t, clip_flow=max_gt_flow_value)
            row_3.append(flow_t_image)
            titles_3[t] = f"Overall {titles_3[t]}"

        output_images.append(row_3)
        output_titles.append(titles_3)

        row_3 = []
        titles_3 = cycle_titles.copy()

        for t in range(T - 1):
            flow_t = flows_idx[t]
            visible_flow_mask_t = visible_flow_mask_idx[t]
            visible_flow_t = visible_flow_mask_t[..., None] * flow_t
            flow_t_image = flow_to_image(visible_flow_t, clip_flow=max_gt_flow_value)
            row_3.append(flow_t_image)
            titles_3[t] = f"Visible {titles_3[t]}"

        output_images.append(row_3)
        output_titles.append(titles_3)

        row_3 = []
        titles_3 = cycle_titles.copy()

        for t in range(T - 1):
            flow_t = flows_idx[t]
            occluded_flow_mask_t = occluded_flow_mask_idx[t]
            occluded_flow_t = occluded_flow_mask_t[..., None] * flow_t
            flow_t_image = flow_to_image(occluded_flow_t, clip_flow=max_gt_flow_value)
            row_3.append(flow_t_image)
            titles_3[t] = f"Occluded {titles_3[t]}"

        output_images.append(row_3)
        output_titles.append(titles_3)

        row_5 = []
        titles_5 = cycle_titles.copy()
        titles_5 = [x for item in titles_5 for x in repeat(item, 2)]

        for t in range(T - 1):
            visible_flow_mask_t = visible_flow_mask_idx[t]
            gt_flow_t = gt_flows_idx[t]
            visible_flow_t = visible_flow_mask_t[..., None] * gt_flow_t
            gt_visible_flow_t_image = flow_to_image(
                visible_flow_t, clip_flow=max_gt_flow_value
            )
            visible_flow_mask_t = visible_flow_mask_t.astype(float)
            row_5.append((visible_flow_mask_t, True))
            row_5.append(gt_visible_flow_t_image)
            titles_5[2 * t] = f"{titles_5[2*t]} Visible Flow Mask"
            titles_5[2 * t + 1] = f"{titles_5[2*t+1]} Visible GT Flow"

        output_images.append(row_5)
        output_titles.append(titles_5)

        row_6 = []
        titles_6 = cycle_titles.copy()
        titles_5 = [x for item in titles_6 for x in repeat(item, 2)]

        for t in range(T - 1):
            occluded_flow_mask_t = occluded_flow_mask_idx[t]
            gt_flow_t = gt_flows_idx[t]
            occluded_flow_t = occluded_flow_mask_t[..., None] * gt_flow_t
            gt_occluded_flow_t_image = flow_to_image(
                occluded_flow_t, clip_flow=max_gt_flow_value
            )
            occluded_flow_mask_t = occluded_flow_mask_t.astype(float)
            row_6.append((occluded_flow_mask_t, True))
            row_6.append(gt_occluded_flow_t_image)
            titles_6[2 * t] = f"{titles_6[2*t]} Occluded Flow Mask"
            titles_6[2 * t + 1] = f"{titles_6[2*t+1]} Occluded GT Flow"

        output_images.append(row_6)
        output_titles.append(titles_6)

        row_7 = []
        titles_7 = []

        for t in range(T - 1):
            gt_flow_t = gt_flows_idx[t]
            gt_flow_t_image = flow_to_image(gt_flow_t, clip_flow=max_gt_flow_value)
            row_7.append(gt_flow_t_image)
            im_title = f"GT F{t+1}->F{t+2}"
            titles_7.append(im_title)

        output_images.append(row_7)
        output_titles.append(titles_7)

        row_8 = [pred_tracks[t] for t in range(2 * T - 1)]
        titles_8 = [f"Pred T={t+1}" for t in range(2 * T - 1)]

        output_images.append(row_8)
        output_titles.append(titles_8)

        row_9 = [gt_tracks[t] for t in range(T)]
        titles_9 = [f"GT T={t+1}" for t in range(T)]

        output_images.append(row_9)
        output_titles.append(titles_9)

        save_row_major_nd_plot(output_images, output_titles, file_path)

        html_dict["images"].append(html_png_path)
        html_dict["texts"].append(html_png_path)
        html_dict["links"].append(html_png_path)

    return html_dict


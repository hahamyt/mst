from functools import lru_cache
import cv2
import numpy as np


def visualize_instances(imask, bg_color=255,
                        boundaries_color=None, boundaries_width=1, boundaries_alpha=0.8):
    num_objects = imask.max() + 1
    palette = get_palette(num_objects)
    if bg_color is not None:
        palette[0] = bg_color

    result = palette[imask].astype(np.uint8)
    if boundaries_color is not None:
        boundaries_mask = get_boundaries(imask, boundaries_width=boundaries_width)
        tresult = result.astype(np.float32)
        tresult[boundaries_mask] = boundaries_color
        tresult = tresult * boundaries_alpha + (1 - boundaries_alpha) * result
        result = tresult.astype(np.uint8)

    return result


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def visualize_mask(mask, num_cls):
    palette = get_palette(num_cls)
    mask[mask == -1] = 0

    return palette[mask].astype(np.uint8)


def visualize_proposals(proposals_info, point_color=(255, 0, 0), point_radius=1):
    proposal_map, colors, candidates = proposals_info

    proposal_map = draw_probmap(proposal_map)
    for x, y in candidates:
        proposal_map = cv2.circle(proposal_map, (y, x), point_radius, point_color, -1)

    return proposal_map


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            marker = {
                0: cv2.MARKER_CROSS,
                1: cv2.MARKER_DIAMOND,
                2: cv2.MARKER_STAR,
                3: cv2.MARKER_TRIANGLE_UP
            }[p[2]] if p[2] <= 3 else cv2.MARKER_SQUARE
            image = cv2.drawMarker(image, (int(p[1]), int(p[0])),
                                   color, marker, 4, 1)
        else:
            pradius = radius
            image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def draw_instance_map(x, palette=None):
    num_colors = x.max() + 1
    if palette is None:
        palette = get_palette(num_colors)

    return palette[x].astype(np.uint8)


def blend_mask(image, mask, alpha=0.6):
    if mask.min() == -1:
        mask = mask.copy() + 1

    imap = draw_instance_map(mask)
    result = (image * (1 - alpha) + alpha * imap).astype(np.uint8)
    return result


def get_boundaries(instances_masks, boundaries_width=1):
    boundaries = np.zeros((instances_masks.shape[0], instances_masks.shape[1]), dtype=np.bool)

    for obj_id in np.unique(instances_masks.flatten()):
        if obj_id == 0:
            continue

        obj_mask = instances_masks == obj_id
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(obj_mask.astype(np.uint8), kernel, iterations=boundaries_width).astype(np.bool)

        obj_boundary = np.logical_xor(obj_mask, np.logical_and(inner_mask, obj_mask))
        boundaries = np.logical_or(boundaries, obj_boundary)
    return boundaries


def draw_with_blend_and_clicks(img, mask=None, alpha=0.6, clicks_list=None, pos_color=(0, 255, 0),
                               neg_color=(0, 0, 255), radius=4):
    result = img.copy()

    if mask is not None:
        palette = get_palette(np.max(mask) + 1)
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
                 (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)

        # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result


def vis_mask_on_image(image, mask, vis_trimap=False, mask_color=(255, 0, 0), trimap_color=(0, 255, 255)):
    mask = mask.astype(np.float32)
    mask_3 = np.repeat(mask[..., np.newaxis], 3, 2)

    color_mask = np.zeros_like(mask_3)
    color_mask[mask > 0] = mask_color

    fusion_mask = image * 0.3 + color_mask * 0.7
    fusion_mask = image * (1 - mask_3) + fusion_mask * mask_3

    if vis_trimap:
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        fusion_mask = cv2.drawContours(fusion_mask, contours, -1, trimap_color, 2)

    return fusion_mask


def add_tag(image, tag='nodefined', tag_h=40):
    image = image.astype(np.uint8)
    H, W = image.shape[0], image.shape[1]
    tag_blanc = np.ones((tag_h, W, 3)).astype(np.uint8) * 255
    cv2.putText(tag_blanc, tag, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    image = cv2.vconcat([image, tag_blanc])
    return image


def vis_result_base(image, pred_mask, instances_mask, iou, num_clicks, clicks_list, text, last_y,
                    last_x):
    mask_color = (132, 52, 84)
    gt_color = (63, 196, 63)
    trimap_color = (4, 172, 244)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    fusion_pred = vis_mask_on_image(image, pred_mask, vis_trimap=True, mask_color=mask_color, trimap_color=trimap_color)
    fusion_gt = vis_mask_on_image(image, instances_mask, vis_trimap=True, mask_color=gt_color,
                                  trimap_color=trimap_color)

    for i in range(len(clicks_list)):
        click_tuple = clicks_list[i]

        if click_tuple.is_positive:
            color = (0, 255, 0)
        else:
            color = (255, 0, 255)

        coord = click_tuple.coords
        x, y = coord[1], coord[0]
        if x < 0 or y < 0:
            continue
        cv2.circle(fusion_pred, (x, y), 12, color, -1)
        # cv2.putText(fusion_pred, str(i+1), (x-10, y-10),  cv2.FONT_HERSHEY_COMPLEX, 0.6 , color,1 )
    if last_x != -1:
        cv2.circle(fusion_pred, (last_x, last_y), 2, (255, 255, 255), -1)

    fusion_gt = cv2.copyMakeBorder(fusion_gt, 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    position = (30, 60)
    # 定义字体类型和其他属性
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.7
    color = (0, 0, 0)  # 红色
    font_thickness = 3
    bg_color = (255, 255, 255)  # 白色

    # 获取文本的宽度和高度
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    # 计算背景矩形的坐标
    x, y = position
    rect_start = (x, y - text_height - baseline)  # 矩形左上角
    rect_end = (x + text_width, y + baseline)  # 矩形右下角
    # 绘制白色矩形背景
    cv2.rectangle(fusion_pred, rect_start, rect_end, bg_color, -1)  # -1 表示填充矩形
    # 在图像上添加文本
    cv2.putText(fusion_pred, text, position, font, font_scale, color, font_thickness)

    h, w = image.shape[0], image.shape[1]
    if h < w:
        out_image = cv2.hconcat(
            [fusion_gt.astype(np.float32), fusion_pred.astype(np.float32), ])
    else:
        out_image = cv2.hconcat(
            [fusion_gt.astype(np.float32), fusion_pred.astype(np.float32), ])

    return out_image

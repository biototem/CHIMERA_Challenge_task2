'''
openslide image tools
and
tiffslide image tools
'''


import numpy as np
import cv2
from typing import Union, Tuple
try:
    from . import im_tool
    from .numpy_tool import round_int
    from . import image_over_scan_wrapper
    from . import coords_over_scan_gen
    from .void_cls import VC
except (ModuleNotFoundError, ImportError):
    from my_py_lib import im_tool
    from my_py_lib.numpy_tool import round_int
    from my_py_lib import image_over_scan_wrapper
    from my_py_lib import coords_over_scan_gen
    from void_cls import VC

try:
    import openslide as opsl
except (ModuleNotFoundError, ImportError):
    opsl = VC()
    opsl.OpenSlide = None

try:
    import tiffslide as tisl
except (ModuleNotFoundError, ImportError):
    tisl = VC()
    tisl.TiffSlide = None


def get_level0_mpp(opsl_im: Union[opsl.OpenSlide, tisl.TiffSlide]):
    if isinstance(opsl_im, tisl.TiffSlide):
        prop_x, prop_y = tisl.PROPERTY_NAME_MPP_X, tisl.PROPERTY_NAME_MPP_Y
    elif isinstance(opsl_im, opsl.OpenSlide):
        prop_x, prop_y = opsl.PROPERTY_NAME_MPP_X, opsl.PROPERTY_NAME_MPP_Y
    else:
        raise NotImplementedError('Error! Unsupported slide type.')
    x = opsl_im.properties[prop_x]
    y = opsl_im.properties[prop_y]
    return y, x


def read_region_any_ds(opsl_im: Union[opsl.OpenSlide, tisl.TiffSlide],
                       ds_factor: float,
                       level_0_start_yx: Tuple[int, int],
                       level_0_region_hw: Tuple[int, int],
                       close_thresh: float=0.0):
    '''
    从多分辨率图中读取任意尺度图像
    :param opsl_im:             待读取的 OpenSlide 或 tisl.TiffSlide 图像
    :param ds_factor:           下采样尺度
    :param level_0_start_yx:    所读取区域在尺度0上的位置
    :param level_0_region_hw:   所读取区域在尺度0上的高宽
    :param close_thresh:        如果找到足够接近的下采样尺度，则直接使用最接近的，不再对图像进行缩放，默认为0，即为关闭
    :return:
    '''
    level_downsamples = opsl_im.level_downsamples

    level_0_start_yx = round_int(level_0_start_yx)
    level_0_region_hw = round_int(level_0_region_hw)

    assert ds_factor > 0, f'Error! Not allow ds_factor <= 0. ds_factor={ds_factor}'

    # base_level = None
    # ori_patch_hw = None
    # 这里需要使用round，因为大小需要比较准确，避免 511.99 被压到 511
    target_patch_hw = round_int(level_0_region_hw / ds_factor, dtype=np.int64)

    is_close_list = np.isclose(ds_factor, level_downsamples, rtol=close_thresh, atol=0)
    if np.any(is_close_list):
        # 如果有足够接近的
        level = np.argmax(is_close_list)
        base_level = level
        ori_patch_hw = target_patch_hw
    else:
        # 没有足够接近的，则寻找最接近，并且分辨率更高的，然后再缩放。
        # 增加ds_factor超过level_downsamples边界的支持
        if ds_factor > max(opsl_im.level_downsamples):
            # 如果ds_factor大于图像自身包含的最大的倍率
            level = opsl_im.level_count - 1
        elif ds_factor < min(opsl_im.level_downsamples):
            # 如果ds_factor小于图像自身最小的倍率
            level = 0
        else:
            level = np.argmax(ds_factor < np.array(opsl_im.level_downsamples)) - 1
        assert level >= 0, 'Error! read_im_mod found unknow level {}'.format(level)
        base_level = level
        level_ds_factor = level_downsamples[level]
        ori_patch_hw = round_int(target_patch_hw / level_ds_factor * ds_factor, dtype=np.int64)

    # 读取图块，如果不是目标大小则缩放到目标大小
    # 使用 int64 避免 tiffslide 坐标溢出导致的错位的问题
    im = opsl_im.read_region(np.int64(level_0_start_yx[::-1]), base_level, ori_patch_hw[::-1])
    im = np.uint8(im)[:, :, :3]
    if np.any(ori_patch_hw != target_patch_hw):
        im = im_tool.resize_image(im, target_patch_hw, cv2.INTER_AREA)
    return im


opsl_read_region_any_ds = read_region_any_ds


def read_region_any_ds_v2(opsl_im: Union[opsl.OpenSlide, tisl.TiffSlide],
                          ds_factor: float,
                          start_yx: Tuple[int, int],
                          region_hw: Tuple[int, int]
                          ):
    '''
    从多分辨率图中读取任意尺度图像。
    与 read_region_any_ds 区别为，这里的坐标和宽高需要是目标下采样层级的坐标，而不是0层级的坐标。
    也不支持就近选取。
    :param opsl_im:     待读取的 OpenSlide 或 tisl.TiffSlide 图像
    :param ds_factor:   下采样尺度
    :param start_yx:    所读取区域在目标下采样等价的层级的位置
    :param region_hw:   所读取区域在目标下采样等价的层级的高宽
    :return:
    '''
    level_downsamples = opsl_im.level_downsamples

    start_yx = round_int(start_yx)
    region_hw = round_int(region_hw)

    assert ds_factor > 0, f'Error! Not allow ds_factor <= 0. ds_factor={ds_factor}'

    ds_level = opsl_im.get_best_level_for_downsample(ds_factor)

    lv0_start_yx = round_int(start_yx * ds_factor)

    # 这里需要使用round，因为大小需要比较准确，避免 511.99 被压到 511
    read_region_hw = round_int(region_hw * (ds_factor / level_downsamples[ds_level]), dtype=np.int64)

    if isinstance(opsl_im, tisl.TiffSlide):
        patch = opsl_im.read_region(lv0_start_yx[::-1], ds_level, read_region_hw[::-1], as_array=True)
    else:
        patch = np.asarray(opsl_im.read_region(lv0_start_yx[::-1], ds_level, read_region_hw[::-1]))[..., :3]

    assert isinstance(patch, np.ndarray)
    patch = im_tool.resize_image(patch, region_hw, cv2.INTER_AREA)
    return patch


def read_region_any_ds_mpp(opsl_im: Union[opsl.OpenSlide, tisl.TiffSlide],
                           mpp: float,
                           level_0_start_yx: Tuple[int, int],
                           level_0_region_hw: Tuple[int, int]
                           ):
    '''
    从多分辨率图中读取任意尺度图像。
    使用mpp做基准。
    :param opsl_im:             待读取的 OpenSlide 或 tisl.TiffSlide 图像
    :param mpp:                 目标倍镜的mpp。例如：40x的mpp为0.25，20x的mpp为0.5，10x的mpp为1.0，依此类推
    :param level_0_start_yx:    所读取区域在0层级的位置
    :param level_0_region_hw:   所读取区域在0层级的高宽
    :return:
    '''
    lv0_mpp = get_level0_mpp(opsl_im)
    ds_factor = mpp / np.float32(lv0_mpp)
    ds_factor = np.mean(ds_factor)
    patch = read_region_any_ds(opsl_im, ds_factor, level_0_start_yx, level_0_region_hw, 0.)
    return patch


def read_region_any_ds_mpp_v2(opsl_im: Union[opsl.OpenSlide, tisl.TiffSlide],
                              mpp: float,
                              start_yx: Tuple[int, int],
                              region_hw: Tuple[int, int]
                              ):
    '''
    从多分辨率图中读取任意尺度图像。
    使用mpp做基准。
    :param opsl_im:             待读取的 OpenSlide 或 tisl.TiffSlide 图像
    :param mpp:                 目标倍镜的mpp。例如：40x的mpp为0.25，20x的mpp为0.5，10x的mpp为1.0，依此类推
    :param start_yx:            所读取区域在指定mpp层级的位置
    :param region_hw:           所读取区域在指定mpp层级的高宽
    :return:
    '''
    lv0_mpp = get_level0_mpp(opsl_im)
    ds_factor = mpp / np.float32(lv0_mpp)
    ds_factor = np.mean(ds_factor)
    patch = read_region_any_ds_v2(opsl_im, ds_factor, start_yx, region_hw)
    return patch


def get_slide_hw_by_mpp(opsl_im: Union[opsl.OpenSlide, tisl.TiffSlide], mpp: float):
    '''
    获得slide在指定mpp值时的高宽
    :param opsl_im:
    :param mpp:
    :return:
    '''
    lv0_mpp = get_level0_mpp(opsl_im)
    ds_factor = mpp / np.asarray(lv0_mpp, np.float32)
    lv0_hw = opsl_im.level_dimensions[0][::-1]
    hw = round_int(lv0_hw / ds_factor, np.int64)
    return hw


def get_ds_factor_by_mpp(opsl_im: Union[opsl.OpenSlide, tisl.TiffSlide], mpp: float):
    '''
    获得slide在指定mpp值时的下采样倍率
    :param opsl_im:
    :param mpp:
    :return:
    '''
    lv0_mpp = get_level0_mpp(opsl_im)
    ds_factor = mpp / np.asarray(lv0_mpp, np.float32)
    return ds_factor.mean()


def make_thumb_any_level(bim: Union[opsl.OpenSlide, tisl.TiffSlide],
                         ds_level=None,
                         thumb_size=2048,
                         tile_hw=(512, 512),
                         *,
                         lv0_region_bbox=None,
                         ):
    '''
    从任意层级创建缩略图
    :param bim:             大图像，允许为 OpenSlide 或 TiffSlide
    :param ds_level:        指定层级
    :param thumb_size:      缩略图最长边的长度
    :param tile_hw:         采样时图块大小
    :param lv0_region_bbox: 生成指定区域的缩略图，格式为 y1x1y2x2。 例如 [100, 100, 400, 400]
    :return:
    '''
    # 支持自适应索引
    if ds_level is None:
        ds_level = bim.get_best_level_for_downsample(min(bim.level_dimensions[0]) / thumb_size)

    # 支持逆向索引
    if ds_level < 0:
        ds_level = bim.level_count + ds_level

    # 支持区域缩略图生成
    if lv0_region_bbox is None:
        lv0_region_bbox = [0, 0, *bim.level_dimensions[0][::-1]]
    else:
        assert len(lv0_region_bbox) == 4
    lv0_region_bbox = np.int64(lv0_region_bbox)

    lv0_hw = bim.level_dimensions[0][::-1]
    level_hw = bim.level_dimensions[ds_level][::-1]

    to_lv0_factor = lv0_hw / np.float32(level_hw)

    level_region_bbox = round_int(lv0_region_bbox / np.concatenate([to_lv0_factor, to_lv0_factor]))
    level_region_pos = level_region_bbox[:2]
    level_region_hw = level_region_bbox[2:]-level_region_bbox[:2]

    factor = np.min(thumb_size / np.float32(level_region_hw))

    thumb_hw = round_int(np.float32(level_region_hw) * factor)

    thumb_im = np.zeros([*thumb_hw, 3], np.uint8)
    thumb_im_wrap = image_over_scan_wrapper.ImageOverScanWrapper(thumb_im)

    for yx_start, yx_end in coords_over_scan_gen.n_step_scan_coords_gen_v2(level_region_hw, window_hw=tile_hw, n_step=1):

        lv0_pos = round_int((yx_start+level_region_pos)[::-1] * to_lv0_factor)

        tile = np.asarray(bim.read_region(lv0_pos, ds_level, tile_hw[::-1]))[..., :3]

        assert tile.shape[0] == tile_hw[0] and tile.shape[1] == tile_hw[1]

        yx_start = round_int(np.float32(yx_start) * factor)
        yx_end = round_int(np.float32(yx_end) * factor)

        new_hw = yx_end - yx_start

        tile = im_tool.resize_image(tile, new_hw, interpolation=cv2.INTER_AREA)

        thumb_im_wrap.set(yx_start, yx_end, tile)

    return thumb_im


if __name__ == '__main__':
    import imageio

    opsl_im = opsl.OpenSlide(r"#142.ndpi")
    im = opsl_read_region_any_ds(opsl_im, ds_factor=4, level_0_start_yx = (20000, 20000), level_0_region_hw = (5120, 5120))
    imageio.imwrite('1.jpg', im)

    opsl_im = tisl.TiffSlide(r"#142.ndpi")
    im = opsl_read_region_any_ds(opsl_im, ds_factor=4, level_0_start_yx = (20000, 20000), level_0_region_hw = (5120, 5120))
    imageio.imwrite('2.jpg', im)

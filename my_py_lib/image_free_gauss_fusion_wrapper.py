'''
自由高斯融合包装器，主要用于大图进行高斯融合
默认实现为硬盘读取。可以自行继承或替代读取函数，以实现从任意方式读取

实现分区，如果图块特别多，例如超过8000个图块，分区可以实现加速
'''

import numpy as np
from functools import lru_cache
from my_py_lib.bbox_tool import calc_bbox_occupancy_ratio_1toN
from my_py_lib.list_tool import list_multi_get_with_ids
from my_py_lib.image_over_scan_wrapper import ImageOverScanWrapper
from my_py_lib import draw_tool
import imageio.v3 as imageio



class ImageFreeGaussFusionWrapper:
    def __init__(self, coords, paths, im_ch=3, zone_size=None):
        '''
        :param coords:      坐标列表，要求格式为 [[y1x1y2x2], ...]
        :param paths:       图块的路径
        :param im_ch:       图像通道数
        :param zone_size:   分区大小，None代表不分区
        '''
        coords = np.atleast_2d(coords).astype(np.int32)
        assert coords.ndim == 2 and coords.shape[-1] == 4, 'Error! Bad coords format.'
        assert len(coords) == len(paths), 'Error! The len(paths) must be equal with len(coords).'
        assert zone_size is None or int(zone_size) >= 1, 'Error! The zone_size must be is None or (int type and zone_size >= 1) .'
        self.zone_size = zone_size

        self.coords = coords
        self.paths = paths
        self.im_ch = im_ch

        self.zones = None
        if zone_size is not None:
            self.zones = self.build_zones(coords, zone_size)

    @staticmethod
    def build_zones(coords, zone_size):
        zones = {}
        for coord_i, coord in enumerate(coords):
            q_coord = np.int32(coord // zone_size)
            for q_y in range(q_coord[0], q_coord[2]+1):
                for q_x in range(q_coord[1], q_coord[3]+1):
                    zones.setdefault(q_y, {}).setdefault(q_x, []).append(coord_i)
        return zones

    def get_relation_patch_ids_by_zones(self, bbox):
        bbox = np.int32(bbox)
        q_bbox = np.int32(bbox // self.zone_size)

        need_patch_ids = set()

        for q_y in range(q_bbox[0], q_bbox[2] + 1):
            for q_x in range(q_bbox[1], q_bbox[3] + 1):
                z = self.zones.get(q_y, None)
                if z is not None:
                    z = z.get(q_x, None)
                    if z is not None:
                        need_patch_ids.update(z)

        return list(need_patch_ids)

    @lru_cache(12)
    def get_image(self, im_path):
        '''
        从硬盘中读取图像
        可以替换为自定义读取方式，从而实现任意方式读取图像
        :param im_path:
        :return:
        '''
        return imageio.imread(im_path)

    @lru_cache(10)
    def make_gauss_map(self, hw):
        im = np.zeros(hw, dtype=np.float32)
        center = [hw[0] // 2, hw[1] // 2]
        im = draw_tool.draw_gradient_circle(im, center, int(center[0] * 1.4), 1, 0.01, 'sqrt')
        return im

    def check_bbox_is_cross(self, bbox, coords):
        inter_y1 = np.maximum(bbox[..., 0], coords[..., 0])
        inter_x1 = np.maximum(bbox[..., 1], coords[..., 1])
        inter_y2 = np.minimum(bbox[..., 2], coords[..., 2])
        inter_x2 = np.minimum(bbox[..., 3], coords[..., 3])

        bools = np.all([inter_y2 > inter_y1, inter_x2 > inter_x1], axis=0)
        return bools

    def get_block(self, bbox):
        '''
        获取指定位置的图块
        :param bbox: 要求格式为 y1x1y2x2
        :return:
        '''
        # bbox y1x1y2x2
        bbox = np.asarray(bbox, np.int32)
        assert bbox.ndim == 1 and bbox.shape[0] == 4 and np.all(bbox[2:] >= bbox[:2]), 'Error! Bad bbox format.'

        coords = self.coords
        paths = self.paths
        if self.zones is not None:
            ids = self.get_relation_patch_ids_by_zones(bbox)
            coords = coords[ids]
            paths = list_multi_get_with_ids(paths, ids)

        # oc = calc_bbox_occupancy_ratio_1toN(bbox, coords)
        # ids = np.argwhere(oc > 0).reshape(-1)
        bools = self.check_bbox_is_cross(bbox, coords)
        ids = np.argwhere(bools).reshape(-1)
        paths = list_multi_get_with_ids(paths, ids)
        coords = coords[ids]

        hw = (bbox[2] - bbox[0], bbox[3] - bbox[1])

        if len(paths) == 0:
            return np.zeros([*hw, 3], dtype=np.uint8)

        else:
            cur_im = np.zeros([*hw, 3], dtype=np.float32)
            cur_mask = np.zeros([*hw, 1], dtype=np.float32)

            w_cur_im = ImageOverScanWrapper(cur_im)
            w_cur_mask = ImageOverScanWrapper(cur_mask)

            for path, coord in zip(paths, coords):
                hw = (coord[2]-coord[0], coord[3]-coord[1])
                gm = self.make_gauss_map(hw)[:,:,None]
                im = self.get_image(path)

                new_coord = [coord[0] - bbox[0], coord[1] - bbox[1]]
                new_coord.extend([new_coord[0] + hw[0], new_coord[1] + hw[1]])

                temp_im = w_cur_im.get(new_coord[:2], new_coord[2:])
                temp_mask = w_cur_mask.get(new_coord[:2], new_coord[2:])

                temp_im += im * gm
                temp_mask += gm

                w_cur_im.set(new_coord[:2], new_coord[2:], temp_im)
                w_cur_mask.set(new_coord[:2], new_coord[2:], temp_mask)

            out_im = cur_im / np.clip(cur_mask, 1e-8, None)
            out_im = np.round_(out_im).clip(0, 255).astype(np.uint8)
            return out_im


if __name__ == '__main__':
    sgb = ImageFreeGaussFusionWrapper('/mnt/totem_data/totem/fengwentai/project/AI-FFPE-main/my_convert_color_slide/out_dir_froze2wax_7_step1/TCGA-B6-A0X7-01A-01-TS1.9446fbf5-34ff-4d5d-b292-ed8b129d0281.svs')

    im = sgb.get_block([2000, 2000, 4000, 4000])
    imageio.imwrite('tout.png', im)

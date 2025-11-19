# @Time    : 2022.11.21
# @Author  : kawa Yeung
# @Licence : bio-totem
import os
import tiffslide
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class Slide(tiffslide.TiffSlide):
    def __init__(self, svs_file, level=2):
        """
        open svs file with open-slide
        :param svs_file: svs file, absolute path
        :return: slide
        """
        super().__init__(svs_file)
        self._filepath = svs_file
        self._basename = os.path.basename(svs_file).split('.')[0]
        self.slide = tiffslide.TiffSlide(svs_file)
        self._level = level

    def get_basename(self):
        """
        return svs file basename, not contain file suffix
        :return:
        """

        return self._basename

    def get_filepath(self):
        """
        get absolute svs file
        :return:
        """

        return self._filepath

    def get_level(self):
        """
        return level
        :return:
        """

        return self._level

    def get_level_downsample(self, level=2):
        """
        get the expected level downsample, default level two
        :param level: level, default 2
        :return: the level downsample
        """

        return self.slide.level_downsamples[level]

    def get_level_dimension(self, level=2):
        """
        get the expected level dimension, default level two
        :param level: level, default 0
        :return:
        """

        return self.slide.level_dimensions[level]

    def get_thumb(self, level=2):
        """
        get thumb image
        :return:
        """

        level_dimension = self.get_level_dimension(level)
        # print(level_dimension)
        tile = self.slide.get_thumbnail(level_dimension)

        return tile

    def get_level_count(self):
        """
        return number of levels
        :return:
        """
        return self.slide.level_count

    def get_最靠近指定分辨率下的层级(self, target_mpp):
        最靠近目标分辨率下的层级 = 0
        jfjfp_pfpf = 99999
        min_img_lever = self.slide.level_count
        for tmp_i in range(min_img_lever):
            mpp_in_level = self.properties['tiffslide.mpp-x'] * self.slide.level_downsamples[tmp_i]
            ooooo_tmp = abs(target_mpp - mpp_in_level)
            if ooooo_tmp < jfjfp_pfpf:
                jfjfp_pfpf = ooooo_tmp
                最靠近目标分辨率下的层级 = tmp_i
        return 最靠近目标分辨率下的层级

    def svs_to_png(self, save_dir):
        """
        convert svs to png
        :return:
        """
        self.get_thumb().save(save_dir)

    def expand_img(self, im, size, value=(0, 0, 0)):
        """
        expand the image
        :param im: the image want to expand
        :param size: tuple, the size of expand
        :param value: tuple, the pixel value at the expand region
        :return: the expanded image
        """

        im_new = Image.new("RGB", size, value)
        im_new.paste(im, (0, 0))

        return im_new

    def get_mpp(self):
        """
        get the value of mpp
        :return: 0.00025
        """
        properties = self.properties
        return np.float32(properties['tiffslide.mpp-x']) / 1000

    def get_mpp_mag_ys(self):
        mpp_ys = float(self.slide.properties['tiffslide.mpp-x'])
        try:
            mag_ys = int(self.slide.properties['aperio.AppMag'])
        except:
            if 0.2 < mpp_ys < 0.3:
                # 指定缩放倍率
                mag_ys = 40
            elif 0.4 < mpp_ys < 0.6:
                # 指定缩放倍率
                mag_ys = 20
            else:
                assert False, '请确认这批数据的mag具体的值'
        return mpp_ys, mag_ys
    # def __del__(self):
    #     self.slide.close()


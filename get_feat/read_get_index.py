from tiffslide_utils import  Slide as tiffSlide
import numpy as np

def get_index_list(wsi_path,model_target_mpp,model_pred_szie,mask_01_array):

    img_slide = tiffSlide(wsi_path)
    svs_w, svs_h = img_slide.level_dimensions[0]
    wsi_mpp_um = img_slide.get_mpp()*1000
    assert 0.1 < wsi_mpp_um < 1.25, '----请检查是否获取到wsi正确的mpp: ' + str(wsi_mpp_um)
    target_w = round(svs_w * wsi_mpp_um / model_target_mpp)
    target_h = round(svs_h * wsi_mpp_um / model_target_mpp)
    最靠近目标分辨率下的层级 = img_slide.get_最靠近指定分辨率下的层级(model_target_mpp)
    mask_downsamples =  (target_w/mask_01_array.shape[1])
    pred_size_最靠近目标分辨率下的层级的预测框大小 = round(model_pred_szie * model_target_mpp / (wsi_mpp_um*img_slide.get_level_downsample(最靠近目标分辨率下的层级)))
    list_0_tmp  = []
    tmp_1um_75 = (model_pred_szie*2/mask_downsamples)**2*0.75
    for h_tmp in range(0,target_h,model_pred_szie):
        for w_tmp in range(0,target_w,model_pred_szie):
            mask_result_tmp = mask_01_array[round(h_tmp/mask_downsamples):round((h_tmp + model_pred_szie)/mask_downsamples), round(w_tmp/mask_downsamples):round((w_tmp + model_pred_szie)/mask_downsamples)]

            h_10x = round(h_tmp - (model_pred_szie / 2))
            w_10x = round(w_tmp - (model_pred_szie/ 2))
            mask_result_tmp1um_75 = mask_01_array[ round(h_10x/mask_downsamples):round((h_10x + model_pred_szie*2) / mask_downsamples), round( w_10x/mask_downsamples ):round((w_10x + model_pred_szie*2) / mask_downsamples)]

            if 1 in  mask_result_tmp:
                h1_tmp_level0 = round(h_tmp * model_target_mpp / wsi_mpp_um)
                w1_tmp_level0 = round(w_tmp * model_target_mpp / wsi_mpp_um)
                h2_tmp_level0 = round((h_tmp+model_pred_szie) * model_target_mpp / wsi_mpp_um)
                w2_tmp_level0 = round((w_tmp+model_pred_szie) * model_target_mpp / wsi_mpp_um)
                if np.sum(mask_result_tmp1um_75)>tmp_1um_75:
                    list_0_tmp.append([h1_tmp_level0, w1_tmp_level0, h2_tmp_level0, w2_tmp_level0, 最靠近目标分辨率下的层级, pred_size_最靠近目标分辨率下的层级的预测框大小, True])
                else:
                    list_0_tmp.append([h1_tmp_level0, w1_tmp_level0, h2_tmp_level0, w2_tmp_level0, 最靠近目标分辨率下的层级, pred_size_最靠近目标分辨率下的层级的预测框大小, False])

    return list_0_tmp


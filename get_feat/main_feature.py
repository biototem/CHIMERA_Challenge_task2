import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.backends.cuda as cuda
cuda.benchmark = False
cuda.deterministic = True
from tiffslide_utils import  Slide as tiffSlide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = 70746520960
from tile_enhancement import MyAugFunc
import numpy as np
import torch
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from transformers import PretrainedConfig, PreTrainedModel, BertModel, AutoConfig, AutoModel, BertConfig, AutoTokenizer
from read_get_index import get_index_list
class roi_dataset(Dataset):
    def __init__(self,img_list,transform):
        super().__init__()
        self.images_lst = img_list
        self.Cache_slide = {}
        self.trnsfrms_val = transform
        self.aug_func = MyAugFunc()
    def __len__(self):
        return len(self.images_lst)
    def __getitem__(self, idx):
        index_n  = self.images_lst[idx]
        wsi_path_tmp = index_n[-1]
        if wsi_path_tmp not in self.Cache_slide:
            img_slide = tiffSlide(wsi_path_tmp)
            self.Cache_slide[wsi_path_tmp] = img_slide
        img_slide = self.Cache_slide[wsi_path_tmp]
        最靠近目标分辨率的采样层级 = index_n[4]
        index_20x_h1,index_20x_w1,index_20x_h2,index_20x_w2,index_20x_size = index_n[0],index_n[1],index_n[2],index_n[3],index_n[5]
        image_20x = img_slide.read_region((index_20x_w1, index_20x_h1), 最靠近目标分辨率的采样层级, (index_20x_size, index_20x_size),as_array=True)[:,:,:3]
        image_20x_增强 = self.aug_func(image_20x)
        image_20x_11 = self.trnsfrms_val(Image.fromarray(image_20x))
        image_20x_22 = self.trnsfrms_val(Image.fromarray(image_20x_增强))

        return image_20x_11,image_20x_22
if __name__ == '__main__':

    dir1 = r'/media/he/14T/0719/NMBIC BRC predict/'
    out_dir = r'/media/he/14T/0719/HR-NMIBC/'
    model_path = '/home/he/2_13/tmp-2025-8-4/bioptimus_H0-mini'


    wsi_path_list = []

    list345565 = sorted(os.listdir(dir1))#['2A_007', '2B_408', '2B_250', '2B_267', '2A_035', '2A_016', '2A_050', '2A_083', '2A_041', '2B_319', '2B_225', '2A_147', '2A_093', '2A_019', '2A_025', '2B_354', '2B_338', '2A_158', '2A_053', '2A_087', '2B_328', '2A_186', '2A_004', '2A_023', '2A_140', '2A_162', '2A_126', '2B_337', '2B_302', '2A_097', '2A_074', '2A_116', '2B_288', '2A_168', '2A_015', '2A_089', '2B_385', '2A_088', '2B_230', '2A_012', '2A_161', '2B_351', '2B_357', '2B_322', '2A_135', '2B_310', '2A_141', '2A_010', '2B_277', '2A_005', '2A_028', '2A_042', '2A_006', '2B_229', '2A_145', '2B_362', '2A_029', '2A_039', '2A_123', '2A_064', '2A_068', '2A_062', '2A_036', '2B_417', '2A_129', '2B_410', '2A_113', '2B_321', '2B_428', '2A_138', '2A_130', '2A_105', '2A_024', '2B_411', '2A_165', '2A_095', '2A_108']


    for i in list345565:
        if '.csv' in i: continue
        path1 = dir1 + i + '/' + i + '_HE.tif'
        path2 = dir1 + i + '/' + i + '_HE_mask.tif'
        wsi_path_list.append((path1, path2))

    target_size = 224
    target_mpp = 0.5
    output_feat_xy_dir = out_dir + '/位置信息_224/'
    output_feat_dir = out_dir + '/H0-mini/'
    output_feat_xy_dir_可视化 = out_dir + '/weiz_v11/'
    os.makedirs(os.path.dirname(output_feat_xy_dir + '/'), exist_ok=True)
    os.makedirs(os.path.dirname(output_feat_xy_dir_可视化 + '/'), exist_ok=True)
    os.makedirs(os.path.dirname(output_feat_dir + '/'), exist_ok=True)


    weight_path = model_path + '/pytorch_model.bin'
    config = AutoConfig.from_pretrained(model_path)
    config.model_args['mlp_layer'] = timm.layers.SwiGLUPacked
    config.model_args['act_layer'] = torch.nn.SiLU

    model1 = timm.create_model(
        config.architecture,
        pretrained=False,
        pretrained_cfg=config.pretrained_cfg,
        checkpoint_path=weight_path,
        **config.model_args,
    )

    state_dict = torch.load(model_path + '/pytorch_model.bin', map_location='cpu')
    model1.load_state_dict(state_dict, strict=True)

    model1.to("cuda")
    model1.eval()

    transform = create_transform(**resolve_data_config(model1.pretrained_cfg, model=model1))


    for wsi_path0 in wsi_path_list:
        wsi_path, mask_path = wsi_path0
        file_name = os.path.basename(wsi_path)

        # title_index_n = np.load(output_feat_xy_dir + str(file_name).replace('.tif', '.npy'))

        file_name = os.path.basename(wsi_path)
        name, ext = os.path.splitext(file_name)
        mask_slide = tiffSlide(mask_path)
        svs_w, svs_h = mask_slide.level_dimensions[0]

        mask_01_array = mask_slide.read_region((0, 0), 0, (svs_w, svs_h), as_array=True)[:, :, 0]

        title_index_n = get_index_list(wsi_path, target_mpp, target_size, mask_01_array)


        list_index_n = [list(i) + [wsi_path] for i in title_index_n]
        database_loader = DataLoader(roi_dataset(list_index_n,transform), batch_size=16,num_workers=4, shuffle=False, pin_memory=True)
        all_20x_feat = []
        all_20x_aug_feat = []
        with torch.inference_mode():
            for batch_20X,batch_20X_aug in tqdm(database_loader):
                batch_20X_1,batch_20X_aug_1 = batch_20X.cuda(),batch_20X_aug.cuda()
                feat_ys = model1(batch_20X_1).cpu().numpy()
                feat_ys = feat_ys[:, 0]
                feat_aug = model1(batch_20X_aug_1).cpu().numpy()
                feat_aug = feat_aug[:, 0]
                all_20x_feat.append(feat_ys)
                all_20x_aug_feat.append(feat_aug)
        feat_tmp = np.expand_dims(np.concatenate(all_20x_feat, axis=0),axis=1)
        feat_tmp1 = np.expand_dims(np.concatenate(all_20x_aug_feat, axis=0),axis=1)
        feat_all = np.concatenate([feat_tmp,feat_tmp1], axis=1)
        np.save(output_feat_dir + os.path.splitext(file_name)[0] + '.npy', feat_all)
        all_20x_feat.clear()
        all_20x_aug_feat.clear()




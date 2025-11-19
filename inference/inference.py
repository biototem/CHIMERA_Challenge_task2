"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""
import gc
from pathlib import Path
import json
from glob import glob
from tiffslide import  TiffSlide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = 70746520960
import numpy as np
import torch
import timm
from architecture.transformer import AttnMIL6 as AttnMIL
from architecture.transformer_ys import AttnMIL6 as AttnMIL_YS

from timm.data.transforms_factory import create_transform
from transformers import PretrainedConfig, PreTrainedModel, BertModel, AutoConfig, AutoModel, BertConfig, AutoTokenizer
from timm.data import resolve_data_config

class roi_dataset(Dataset):
    def __init__(self,img_list,transform_xx):
        super().__init__()
        self.images_lst = img_list
        self.Cache_slide = {}
        self.trnsfrms_val = transform_xx
    def __len__(self):
        return len(self.images_lst)
    def __getitem__(self, idx):
        index_n  = self.images_lst[idx]
        wsi_path_tmp = index_n[-1]
        if wsi_path_tmp not in self.Cache_slide:
            img_slide = TiffSlide(wsi_path_tmp)
            self.Cache_slide[wsi_path_tmp] = img_slide
        img_slide = self.Cache_slide[wsi_path_tmp]
        最靠近目标分辨率的采样层级 = index_n[4]
        index_20x_h1,index_20x_w1,index_20x_h2,index_20x_w2,index_20x_size = index_n[0],index_n[1],index_n[2],index_n[3],index_n[5]
        image_20x = img_slide.read_region((index_20x_w1, index_20x_h1), 最靠近目标分辨率的采样层级, (index_20x_size, index_20x_size),as_array=True)[:,:,:3]
        image_20x_11 = self.trnsfrms_val(Image.fromarray(image_20x))
        return image_20x_11
def get_最靠近指定分辨率下的层级(slide_TMP, target_mpp):
    最靠近目标分辨率下的层级 = 0
    jfjfp_pfpf = 99999
    min_img_lever = slide_TMP.level_count
    for tmp_i in range(min_img_lever):
        mpp_in_level = float(slide_TMP.properties['tiffslide.mpp-x']) * slide_TMP.level_downsamples[tmp_i]
        ooooo_tmp = abs(target_mpp - mpp_in_level)
        if ooooo_tmp < jfjfp_pfpf:
            jfjfp_pfpf = ooooo_tmp
            最靠近目标分辨率下的层级 = tmp_i
    return 最靠近目标分辨率下的层级

def get_index_list(wsi_path,model_target_mpp,model_pred_szie,mask_01_array):
    img_slide = TiffSlide(wsi_path)
    svs_w, svs_h = img_slide.level_dimensions[0]
    wsi_mpp_um = float(img_slide.properties['tiffslide.mpp-x'])
    assert 0.1 < wsi_mpp_um < 1.25, '----请检查是否获取到wsi正确的mpp: ' + str(wsi_mpp_um)
    target_w = round(svs_w * wsi_mpp_um / model_target_mpp)
    target_h = round(svs_h * wsi_mpp_um / model_target_mpp)
    最靠近目标分辨率下的层级 = get_最靠近指定分辨率下的层级(img_slide,model_target_mpp)
    mask_downsamples =  (target_w/mask_01_array.shape[1])
    pred_size_最靠近目标分辨率下的层级的预测框大小 = round(model_pred_szie * model_target_mpp / (wsi_mpp_um*img_slide.level_downsamples[最靠近目标分辨率下的层级]))
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



def read_data_json(data_json_ys):
    data_json = {}
    for uuu1 in data_json_ys.keys():
        data_json[str(uuu1).lower()] = data_json_ys[uuu1]
    age_tmp = [0, 0, 0, 0, 0]
    no_instillations_tmp = [0, 0, 0, 0, 0]
    data_dict11 = {}
    data_linc_feat = []
    linc_dict = {'sex': {'NA': 0, 'Male': 1, 'Female': 2}, 'smoking': {'NA': 0, 'No': 1, 'Yes': 2}, 'tumor': {'NA': 0, 'Primary': 1, 'Recurrence': 2}, 'stage': {'NA': 0, 'TaHG': 1, 'T1HG': 2, 'T2HG': 3, 'T3HG': 4}, 'substage': {'NA': 0, 'T1m': 1, 'T1e': 2}, 'grade': {'NA': 0, 'G2': 1, 'G3': 2}, 'retur': {'NA': 0, 'No': 1, 'Yes': 2}, 'lvi': {'NA': 0, 'No': 1, 'Yes': 2}, 'variant': {'NA': 0, 'UCC': 1, 'UCC + Variant': 2}, 'eortc': {'NA': 0, 'High risk': 1, 'Highest risk': 2}}

    for ii1 in ['age', 'sex', 'smoking', 'tumor', 'stage', 'substage', 'grade', 'reTUR', 'LVI', 'variant', 'EORTC','no_instillations']:
        ii1 = ii1.lower()
        try:
            if (ii1 == 'age') | (ii1 == 'no_instillations'):
                data_dict11[ii1] = int(data_json[ii1])
            else:
                data_dict11[ii1] = data_json[ii1]
        except:
            data_dict11[ii1] = 'NA'
    data_age = data_dict11['age']
    data_no_instillations = data_dict11['no_instillations']
    if (str(data_age).upper() == 'NA') | (data_age == -1):
        data_age_feat = age_tmp.copy()
        data_age_feat[0] = 1
    else:
        data_age_feat = age_tmp.copy()
        if data_age < 50:
            data_age_feat[1] = 1
        if 50 <= data_age < 65:
            data_age_feat[2] = 1
        if 65 <= data_age < 75:
            data_age_feat[3] = 1
        if 75 <= data_age:
            data_age_feat[4] = 1
    if (str(data_no_instillations).upper() == 'NA') | (data_no_instillations == -1):
        data_no_instillations_feat = no_instillations_tmp.copy()
        data_no_instillations_feat[0] = 1
    else:
        data_no_instillations_feat = no_instillations_tmp.copy()
        if data_no_instillations <= 6:
            data_no_instillations_feat[1] = 1
        if 7 <= data_no_instillations < 13:
            data_no_instillations_feat[2] = 1
        if 13 <= data_no_instillations <= 30:
            data_no_instillations_feat[3] = 1
        if 30 < data_no_instillations:
            data_no_instillations_feat[4] = 1
    data_linc_feat = data_age_feat + data_no_instillations_feat + data_linc_feat
    for ii1 in ['sex', 'smoking', 'tumor', 'stage', 'substage', 'grade', 'reTUR', 'LVI', 'variant', 'EORTC']:
        ii1 = ii1.lower()
        tmp_list1 = [0 for kk in range(len(linc_dict[ii1].keys()))]
        try:
            data_tmp22 = data_dict11[ii1]
            tmp_v = linc_dict[ii1][data_tmp22]
            tmp_list1[tmp_v] = 1
        except:
            tmp_list1[0] = 1
        data_linc_feat = data_linc_feat + tmp_list1
    return data_linc_feat




INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        (
            "bladder-cancer-tissue-biopsy-whole-slide-image",
            "chimera-clinical-data-of-bladder-cancer-patients",
            "tissue-mask",
        ): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input - use thumbnail loading for tissue mask to avoid memory issues with large WSI tissue masks
    input_tissue_mask_path = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/tissue-mask",
        max_size=1024,
    )
    # Use thumbnail loading for large WSI to avoid memory issues
    input_image_path = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi",
        max_size=1024,
    )
    input_json_data = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-patients.json",
    )


    target_size = 224
    target_mpp = 0.5
    mask_slide = TiffSlide(input_tissue_mask_path)
    svs_w, svs_h = mask_slide.level_dimensions[0]
    if max([svs_w, svs_h])>10000:
        svs_w, svs_h = mask_slide.level_dimensions[1]
        mask_01_array = mask_slide.read_region((0, 0), 1, (svs_w, svs_h), as_array=True)[:, :, 0]
    else:
        svs_w, svs_h = mask_slide.level_dimensions[0]
        mask_01_array = mask_slide.read_region((0, 0), 0, (svs_w, svs_h), as_array=True)[:, :, 0]

    list_0_tmp = get_index_list(input_image_path, target_mpp, target_size, mask_01_array)
    list_index_n = [i33 + [input_image_path] for i33 in list_0_tmp]
    del mask_01_array
    mask_01_array = 1
    del mask_01_array,mask_slide
    gc.collect()
    model_path = './resources/'
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

    state_dict = torch.load(model_path + '/pytorch_model.bin', map_location='cpu',weights_only=False)
    model1.load_state_dict(state_dict, strict=True)
    model1.eval()
    model1.cuda()
    transform_xx = create_transform(**resolve_data_config(model1.pretrained_cfg, model=model1))

    database_loader = DataLoader(roi_dataset(list_index_n,transform_xx), batch_size=8, num_workers=4, shuffle=False,pin_memory=True)
    with torch.inference_mode():
        feat_tmp = np.zeros((len(list_index_n),768),dtype=np.float32)
        idx111 = 0
        for batch_20X in tqdm(database_loader):
            batch_20X_1 = batch_20X.cuda()
            feat_ys = model1(batch_20X_1)
            feat_ys1 = (feat_ys[:, 0]).detach().cpu().numpy()
            del feat_ys
            feat_tmp[idx111:idx111+feat_ys1.shape[0],] = feat_ys1
            idx111 = idx111+feat_ys1.shape[0]
        del batch_20X_1
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        try:
            tmp_linc_feat = np.array(read_data_json(input_json_data))[None,]
            tmp_linc_feat_ys = torch.as_tensor(tmp_linc_feat.copy(), dtype=torch.float32).cuda()[None,]
            tmp_linc_feat = tmp_linc_feat.repeat([feat_tmp.shape[0]], axis=0)

            feat_LINC = np.concatenate([feat_tmp, tmp_linc_feat], axis=1)
            feat_tmp_cuda = torch.as_tensor(feat_LINC, dtype=torch.float32).cuda()[None,]
            output_brs_binary_classification_list = []
            for iii34 in ['1', '2', '3', '4', '5']:
                mil_model1 = AttnMIL(D_feat=810, n_class=2, n_token=5)
                mil_model1.load_state_dict(torch.load('./resources/' + iii34 + '.pth', weights_only=False))
                mil_model1.eval()
                mil_model1.cuda()
                sub_preds, slide_preds, attn = mil_model1(feat_tmp_cuda, tmp_linc_feat_ys, use_attention_mask=False)
                pred = torch.softmax(slide_preds, dim=-1).cpu()
                Probability_1 = float(pred[0][1])
                del mil_model1
                output_brs_binary_classification_list.append(Probability_1)
            output_brs_binary_classification = np.mean(output_brs_binary_classification_list)
        except:
            feat_tmp_cuda = torch.as_tensor(feat_tmp, dtype=torch.float32).cuda()[None,]
            output_brs_binary_classification_list = []
            for iii34 in ['1', '2', '3', '4', '5']:
                mil_model1 = AttnMIL_YS(D_feat=768, n_class=2, n_token=5)
                mil_model1.load_state_dict(torch.load('./resources/' + iii34 + '.pth', weights_only=False))
                mil_model1.eval()
                mil_model1.cuda()
                sub_preds, slide_preds, attn = mil_model1(feat_tmp_cuda, use_attention_mask=False)
                pred = torch.softmax(slide_preds, dim=-1).cpu()
                Probability_1 = float(pred[0][1])
                del mil_model1
                output_brs_binary_classification_list.append(Probability_1)
            output_brs_binary_classification = np.mean(output_brs_binary_classification_list)

        del feat_tmp_cuda

    # output_brs_binary_classification = round(random.uniform(0.0, 1.0), 4)
    print(f"Random prediction: {output_brs_binary_classification}")
    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "brs-probability.json",
        content=output_brs_binary_classification,
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    """
    Load image files using appropriate library based on file type:
    - PyVips for pathology images: .tif, .tiff, .mrxs, .svs, .ndpi
    - SimpleITK for radiology images: .mha
    """
    # Find all compatible files
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
        + glob(str(location / "*.mrxs"))
        + glob(str(location / "*.svs"))
        + glob(str(location / "*.ndpi"))
    )
    
    if not input_files:
        raise FileNotFoundError(f"No compatible image files found in {location}")
    
    file_path = input_files[0]

    return file_path



def load_image_file_as_thumbnail(*, location, max_size=1024):
    """
    Load image as a thumbnail for memory-efficient processing of WSIs
    This is recommended for actual whole slide images
    Returns the PyVips image object directly for memory efficiency
    """
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
        + glob(str(location / "*.mrxs"))
        + glob(str(location / "*.svs"))
        + glob(str(location / "*.ndpi"))
    )
    
    if not input_files:
        raise FileNotFoundError(f"No compatible image files found in {location}")
    
    file_path = input_files[0]
    print(f"Loading pathology image as thumbnail using PyVips: {file_path}")

    return file_path


def _show_torch_cuda_info():

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())

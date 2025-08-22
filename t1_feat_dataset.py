import os.path
import random
import numpy as np
import torch
import t1_cfg
from torch.utils.data.dataset import Dataset
import pickle


class FeatDataset_train(Dataset):
    def __init__(self,data_dict):
        self.data_dict = data_dict
        self.train_number = t1_cfg.train_number
        self.label_dict  = {'2A_001': 2, '2A_002': 1, '2A_003': 2, '2A_004': 2, '2A_005': 0, '2A_006': 0, '2A_007': 2, '2A_008': 2, '2A_009': 0, '2A_010': 2, '2A_011': 2, '2A_012': 0, '2A_013': 1, '2A_014': 1, '2A_015': 1, '2A_016': 0, '2A_017': 0, '2A_018': 2, '2A_019': 1, '2A_020': 2, '2A_021': 2, '2A_022': 2, '2A_023': 2, '2A_024': 2, '2A_025': 2, '2A_026': 1, '2A_027': 2, '2A_028': 2, '2A_029': 1, '2A_030': 1, '2A_031': 2, '2A_033': 1, '2A_034': 0, '2A_035': 0, '2A_036': 1, '2A_037': 2, '2A_038': 2, '2A_039': 2, '2A_040': 0, '2A_041': 2, '2A_042': 1, '2A_043': 2, '2A_044': 1, '2A_045': 2, '2A_046': 1, '2A_047': 1, '2A_049': 1, '2A_050': 1, '2A_052': 1, '2A_053': 2, '2A_055': 0, '2A_056': 1, '2A_057': 1, '2A_058': 2, '2A_059': 0, '2A_060': 1, '2A_061': 2, '2A_062': 1, '2A_063': 2, '2A_064': 1, '2A_066': 1, '2A_067': 2, '2A_068': 2, '2A_070': 0, '2A_071': 2, '2A_072': 2, '2A_073': 2, '2A_074': 2, '2A_075': 1, '2A_076': 2, '2A_077': 1, '2A_082': 0, '2A_083': 1, '2A_084': 1, '2A_086': 2, '2A_087': 2, '2A_088': 2, '2A_089': 1, '2A_091': 0, '2A_092': 2, '2A_093': 2, '2A_094': 2, '2A_095': 1, '2A_097': 1, '2A_098': 2, '2A_100': 2, '2A_104': 1, '2A_105': 0, '2A_108': 2, '2A_110': 1, '2A_111': 2, '2A_113': 2, '2A_114': 2, '2A_115': 0, '2A_116': 1, '2A_123': 0, '2A_124': 1, '2A_125': 0, '2A_126': 0, '2A_127': 0, '2A_129': 2, '2A_130': 0, '2A_134': 1, '2A_135': 0, '2A_136': 0, '2A_137': 0, '2A_138': 1, '2A_139': 1, '2A_140': 0, '2A_141': 0, '2A_142': 0, '2A_143': 0, '2A_144': 0, '2A_145': 1, '2A_146': 0, '2A_147': 1, '2A_148': 2, '2A_149': 0, '2A_153': 1, '2A_154': 0, '2A_157': 0, '2A_158': 0, '2A_160': 0, '2A_161': 1, '2A_162': 0, '2A_163': 1, '2A_165': 2, '2A_168': 2, '2A_169': 0, '2A_186': 1, '2A_190': 0, '2A_191': 0, '2B_208': 2, '2B_217': 1, '2B_225': 2, '2B_227': 0, '2B_229': 2, '2B_230': 2, '2B_250': 1, '2B_262': 2, '2B_266': 2, '2B_267': 2, '2B_277': 2, '2B_281': 2, '2B_288': 0, '2B_292': 0, '2B_302': 2, '2B_303': 1, '2B_304': 1, '2B_309': 2, '2B_310': 2, '2B_319': 2, '2B_321': 0, '2B_322': 0, '2B_328': 0, '2B_337': 0, '2B_338': 2, '2B_342': 0, '2B_351': 2, '2B_354': 2, '2B_357': 0, '2B_361': 0, '2B_362': 2, '2B_365': 2, '2B_367': 1, '2B_370': 2, '2B_385': 1, '2B_389': 0, '2B_390': 0, '2B_397': 0, '2B_399': 1, '2B_408': 0, '2B_410': 1, '2B_411': 0, '2B_413': 0, '2B_415': 1, '2B_417': 1, '2B_418': 1, '2B_426': 1, '2B_428': 2, '2B_429': 1, '2B_431': 1}

        with open('/home/he/桌面/chimera_code/全图特征提取_tiffslide_9_7/data_linc_8-8.pkl', 'rb') as file:
            self.linc_feat = pickle.load(file)

    def __len__(self):
        return self.train_number
    def __getitem__(self, item):
        cls_tmp = random.choice([0, 1])
        if cls_tmp==1:
            item_data = random.choice(self.data_dict[1])
        else:
            cls_tmp = random.choice([0, 2])
            item_data = random.choice(self.data_dict[cls_tmp])
        feat_file,coord_file, cls = item_data[0],item_data[1],item_data[2]
        assert cls_tmp==cls,'按照标签去字典拿的数据，也就意味着字典数据对应的标签都是相同的'
        if cls_tmp==2:
            cls=0

        tmp_linc_feat = np.array(self.linc_feat[os.path.basename(feat_file)][:])[None]

        tmp_linc_feat_new = torch.tensor(tmp_linc_feat,dtype=torch.float32)

        feat = np.load(feat_file+'_HE.npy', mmap_mode='c', allow_pickle=False)
        if len(feat.shape)==2:
            feat = np.expand_dims(feat, axis=1)
        coords = np.load(coord_file+'_HE.npy', mmap_mode='c', allow_pickle=False)[:,:4]
        ids1 = np.arange(feat.shape[0], dtype=np.int32)
        ids2 = np.random.randint(feat.shape[1], size=feat.shape[0], dtype=np.int32)
        feat = feat[ids1, ids2]
        if random.uniform(0, 1) < 0.3:
            keep_n = int(np.random.uniform(0.3, 1.) * feat.shape[0])
            keep_n = max(2, keep_n)
            ids = np.arange(feat.shape[0])
            np.random.shuffle(ids)
            ids = ids[:keep_n]
            feat = feat[ids]
            coords = coords[ids]
            assert len(feat) == len(coords)
            noise = np.random.normal(size=feat.shape, scale=feat.std()*np.random.uniform(0.01, 0.2)).astype(feat.dtype)
            feat += noise

        tmp_linc_feat = tmp_linc_feat.repeat([feat.shape[0]], axis=0)
        feat = np.concatenate([feat,tmp_linc_feat],axis=1)

        return torch.as_tensor(feat,dtype=torch.float32) , torch.as_tensor(coords,dtype=torch.float32) , torch.tensor(cls),tmp_linc_feat_new
        # return tmp_linc_feat_new , torch.as_tensor(coords,dtype=torch.float32) , torch.tensor(cls),tmp_linc_feat_new


class FeatDataset_test(Dataset):
    def __init__(self,data_dict):
        self.cls_list = [0,1,2]
        self.data_dict = data_dict
        self.list_all = []
        for u1 in self.cls_list:
            self.list_all = self.list_all+self.data_dict[u1]
        with open('/home/he/桌面/chimera_code/全图特征提取_tiffslide_9_7/data_linc_8-8.pkl', 'rb') as file:
            self.linc_feat = pickle.load(file)

    def __len__(self):
        return len(self.list_all)
    def __getitem__(self, item):
        item_data = self.list_all[item]
        feat_file,coord_file, cls = item_data[0],item_data[1],item_data[2]
        if cls == 2:
            cls = 0
        tmp_linc_feat = np.array(self.linc_feat[os.path.basename(feat_file)][:])[None]

        tmp_linc_feat_new = torch.tensor(tmp_linc_feat,dtype=torch.float32)

        feat = np.load(feat_file+'_HE.npy', mmap_mode='c', allow_pickle=False)[:,0,:]
        coords = np.load(coord_file+'_HE.npy', mmap_mode='c', allow_pickle=False)[:,:4]
        tmp_linc_feat = tmp_linc_feat.repeat([feat.shape[0]], axis=0)

        feat = np.concatenate([feat,tmp_linc_feat],axis=1)

        return torch.as_tensor(feat,dtype=torch.float32) , torch.as_tensor(coords,dtype=torch.float32) , torch.tensor(cls,dtype=torch.float32),tmp_linc_feat_new

        # return tmp_linc_feat_new , torch.as_tensor(coords,dtype=torch.float32) , torch.tensor(cls),tmp_linc_feat_new

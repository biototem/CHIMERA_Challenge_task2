'''
多分类评分工具
'''

import numpy as np
from .score_tool import calc_score_f05_f1_f2_prec_recall


def calc_class_score(pred_cls_set, label_cls_set, cls_list):
    '''
    通用分类评估
    将会返回一个分数字典
    结构为
    类别-权重-X
    X:
    found_pred      真阳性，预测正确的数量
    fakefound_pred  假阳性，预测失败的数量
    found_label     真阳性，标签被正确匹配到的数量
    nofound_label   假阴性，没有任何成功匹配的标签数量

    :param pred_hms:    预测类别集合，是一个整数集合
    :param label_hms:   标签类别集合，是一个整数集合
    :param cls_list:    要评估的类别列表
    :return:
    '''
    score_table = {}

    pred_cls_set = set(pred_cls_set)
    label_cls_set = set(label_cls_set)

    for cls in cls_list:
        score_table[cls] = {}

        is_in_pred = cls in pred_cls_set
        is_in_label = cls in label_cls_set

        if is_in_pred and is_in_label:
            found_pred = 1
            fakefound_pred = 0
            found_label = 1
            nofound_label = 0
        elif is_in_pred and not is_in_label:
            found_pred = 0
            fakefound_pred = 1
            found_label = 0
            nofound_label = 0
        elif not is_in_pred and is_in_label:
            found_pred = 0
            fakefound_pred = 0
            found_label = 0
            nofound_label = 1
        elif not is_in_pred and not is_in_label:
            found_pred = 0
            fakefound_pred = 0
            found_label = 0
            nofound_label = 0
        else:
            raise AssertionError('Error! 不应该来到这里！')

        # f05, f1, f2, prec, recall = calc_score_f05_f1_f2_prec_recall(found_label, nofound_label, found_pred, fakefound_pred)

        score_table[cls]['found_pred'] = found_pred
        score_table[cls]['fakefound_pred'] = fakefound_pred
        score_table[cls]['found_label'] = found_label
        score_table[cls]['nofound_label'] = nofound_label
        # score_table[cls]['f05'] = float(f05)
        # score_table[cls]['f1'] = float(f1)
        # score_table[cls]['f2'] = float(f2)
        # score_table[cls]['prec'] = float(prec)
        # score_table[cls]['recall'] = float(recall)

    return score_table


def accumulate_class_score(scores, cls_list):
    '''
    对多个分数表进行合算，得到统计分数表
    其中 found_pred, fakefound_pred, found_label, nofound_label 将会被累加
    其中 f1, prec, recall 将会基于以上累加的值重新计算
    :param scores:                      多个分数表
    :param cls_list:                    要检查的分类
    :return:
    '''
    score_table = {}
    for cls in cls_list:
        score_table[cls] = {
            'found_pred': 0,
            'fakefound_pred': 0,
            'found_label': 0,
            'nofound_label': 0,
            'f05': 0.,
            'f1': 0.,
            'f2': 0.,
            'prec': 0.,
            'recall': 0.,
        }

    for score in scores:
        for cls in cls_list:
            score_table[cls]['found_pred'] += score[cls]['found_pred']
            score_table[cls]['fakefound_pred'] += score[cls]['fakefound_pred']
            score_table[cls]['found_label'] += score[cls]['found_label']
            score_table[cls]['nofound_label'] += score[cls]['nofound_label']

    for cls in cls_list:
        f05, f1, f2, prec, recall = calc_score_f05_f1_f2_prec_recall(score_table[cls]['found_label'],
                                                                     score_table[cls]['nofound_label'],
                                                                     score_table[cls]['found_pred'],
                                                                     score_table[cls]['fakefound_pred'])

        score_table[cls]['f05'] = f05
        score_table[cls]['f1'] = f1
        score_table[cls]['f2'] = f2
        score_table[cls]['prec'] = prec
        score_table[cls]['recall'] = recall

    return score_table



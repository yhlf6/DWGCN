
from math import dist
from ordered_set import OrderedSet
from collections import defaultdict
import numpy as np
import json


def avg_degree(dataset):
    '''
    统计origin graph每个实体的邻居节点数

    统计updated graph每个实体的邻居节点数
    '''
    ent_nei_ori0 = defaultdict(list) # 实体词
    ent_nei_ori = {}  ##数量
    ent_nei_update = defaultdict(list)

    ent_set, rel_set = OrderedSet(), OrderedSet()

    for split in ['train', 'test', 'valid']:
        for line in open('./data/{}/{}.txt'.format(dataset, split)):
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            ent_set.add(sub)
            rel_set.add(rel)
            ent_set.add(obj)
    
    for line in open('./data/{}/{}.txt'.format("train_origin", split)):
        sub, rel, obj = map(str.lower, line.strip().split('\t'))
        ent_nei_ori0[sub].append((rel,obj))  ##sub的邻居

    for line in open('./data/{}/{}.txt'.format("train", split)):
        sub, rel, obj = map(str.lower, line.strip().split('\t'))
        ent_nei_update[sub].append((rel,obj))  ##sub的邻居


    for k,v in ent_nei_ori0.items():
        ent_nei_ori[k] = len(v) 
    for k,v in ent_nei_update.items():
        ent_nei_update[k] = len(v) 
    ent_nei_ori_number = []
    ent_nei_update_number = []
    for ent in ent_set:
        ent_nei_ori_number.append(ent_nei_ori.get(ent,0))
        ent_nei_update_number.append(ent_nei_update.get(ent,0))
    avg_degree_ori = np.mean(ent_nei_ori_number)
    avg_degree_update = np.mean(ent_nei_update_number)
    print("origin_degree: {}".format(avg_degree_ori))
    print("updated_degree: {}".format(avg_degree_update))

    ent_nei_ori0["origin_degree"] = avg_degree_ori
    with open('./data/{}/{}.txt'.format(dataset, "train_origin_degree"),"w") as f:
        '''
        把origin_trian中每个实体的邻居实体保存在json文件中
        '''
        json.dump(ent_nei_ori0,f,indent=4)
        print("save adj_table finished...")

def drop_edges(dataset, ratio, per_ratio):
    """
    dataset: 需要处理的数据集名称
    ratio: 保留ratio的那么多度
    删掉一些边，然后保证度减少率为多少
    per_ratio: 删掉每个节点的per_ration个邻居
    """
    with open('./data/{}/{}.txt'.format(dataset, "train_origin_degree"),'r') as load_f:
        load_dict = json.load(load_f)

    avg_degree_ori = load_dict["origin_degree"]  ## 原始训练集的平均度是多少
    num_entities = len(load_dict) - 1
    num_delet_degree = num_entities*avg_degree_ori*(1-ratio)  # 删掉这么多度，相当于删掉边

    num = 0
    for k,v in load_dict.items():
        load_dict[k] = len(v)
    temp = sorted(load_dict.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)   ##按照度 降序
    
    for k,v in temp:
        if num<num_delet_degree:
            per_del_edges = int(v*per_ratio)  ##删掉几条边
            num +=per_del_edges
        if num>num_delet_degree:
            per_del_edges = num - num_delet_degree
    
        load_dict[k] = load_dict[k][per_del_edges:]  ##删掉这么多个邻居数

        if num==num_delet_degree:
            break

    with open('./data/{}/{}_{}.txt'.format(dataset, "train",str(ratio)),'w') as f:
        for k,v in load_dict.items():
            for v1 in v:
                f.write("\t".join([k,v1[0],v1[-1]])+"\n")
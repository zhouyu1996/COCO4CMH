#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/5 18:53
# @Author  : zhouyu
# @content : 
# @File    : coco_data.py
# @Software: PyCharm

import cv2
from sklearn.feature_extraction.text import CountVectorizer
import json
import numpy as np
#from pycocotools.coco import COCO
import os.path
import h5py

coco_url = r'/data/zydata/coco2014'
# this fold contains all the training images
img_train_path = coco_url + r'/train2014/train2014'
# this fold contains all the validation images, we use validation images to generate test set
img_val_path = coco_url + r'/val2014/val2014'
# this file contains the training text data
txt_train_path = coco_url + r'/annotations_trainval2014/annotations/captions_train2014.json'
# this file contains the test text data
txt_val_path = coco_url + r'/annotations_trainval2014/annotations/captions_val2014.json'
# this file contains the category information
instance_train_path = coco_url+ r'/annotations_trainval2014/annotations/instances_train2014.json'
instance_val_path = coco_url+ r'/annotations_trainval2014/annotations/instances_val2014.json'

# sentences: n * sentences
def BOW(sentences):
    count_vec = CountVectorizer(min_df=0.0005, max_df=0.8, binary=True, stop_words='english', max_features=2000)
    a = count_vec.fit_transform(sentences)
    return a.toarray()

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx

def generate_val_set():
    annotations_file = json.load(open(instance_val_path))
    # print(annotations_file.keys())
    category = annotations_file['categories']
    category_id = {}
    for cat in category:
        category_id[cat['id']] = cat['name']
    cat2idx = categoty_to_idx(sorted(category_id.values()))
    annotations = annotations_file['annotations']
    annotations_id = {}
    for annotation in annotations:
        if annotation['image_id'] not in annotations_id:
            annotations_id[annotation['image_id']] = set()
        annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])

    img_id = {}
    images = annotations_file['images']
    for img in images:
        if img['id'] not in annotations_id:
            continue
        if img['id'] not in img_id:
            img_id[img['id']] = {}
        img_id[img['id']]['file_name'] = img['file_name']
        img_id[img['id']]['labels'] = list(annotations_id[img['id']])

    captionsFile = json.load(open(txt_val_path))
    captions = captionsFile['annotations']
    cap_id = {}
    for cap in captions:
        if cap['image_id'] not in cap_id:
            cap_id[cap['image_id']] = set()
        cap_id[cap['image_id']].add(cap['caption'])
    imgCapIdlist = []
    for capk, capv in cap_id.items():
        for k, v in img_id.items():
            if k == capk:
                idAndCaption = v  # 字典
                idAndCaption['image_id'] = k
                idAndCaption['caption'] = list(capv)
                imgCapIdlist.append(idAndCaption)
    print(len(imgCapIdlist))
    # example
    # <class 'dict'>: {'file_name': 'COCO_val2014_000000203564.jpg',
    #                  'labels': [9, 23],
    #                  'image_id': 203564,
    #                  'caption': ['A black metal bicycle with a clock inside the front wheel.',
    #                             'A bicycle replica with a clock as the front wheel.',
    #                             'A bicycle figurine in which the front wheel is replaced with a clock/n',
    #                             'The bike has a clock as a tire.',
    #                             'A clock with the appearance of the wheel of a bicycle ']}
    phase = 'val'
    all_anno = os.path.join('output', '{}_all_anno.json'.format(phase))
    json.dump(imgCapIdlist, open(all_anno, 'w'))

def generate_train_set():
    annotations_file = json.load(open(instance_train_path))
    category = annotations_file['categories']
    category_id = {}
    for cat in category:
        category_id[cat['id']] = cat['name']
    cat2idx = categoty_to_idx(sorted(category_id.values()))
    annotations = annotations_file['annotations']
    annotations_id = {}
    for annotation in annotations:
        if annotation['image_id'] not in annotations_id:
            annotations_id[annotation['image_id']] = set()
        annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])

    img_id = {}
    images = annotations_file['images']
    for img in images:
        if img['id'] not in annotations_id:
            continue
        if img['id'] not in img_id:
            img_id[img['id']] = {}
        img_id[img['id']]['file_name'] = img['file_name']
        img_id[img['id']]['labels'] = list(annotations_id[img['id']])

    captionsFile = json.load(open(txt_train_path))
    captions = captionsFile['annotations']
    cap_id = {}
    for cap in captions:
        if cap['image_id'] not in cap_id:
            cap_id[cap['image_id']] = set()
        cap_id[cap['image_id']].add(cap['caption'])
    imgCapIdlist = []
    for capk, capv in cap_id.items():
        for k, v in img_id.items():
            if k == capk:
                idAndCaption = v
                idAndCaption['image_id'] = k
                idAndCaption['caption'] = list(capv)
                imgCapIdlist.append(idAndCaption)
    print(len(imgCapIdlist))
    phase = 'train'
    all_anno = os.path.join('output', '{}_all_anno.json'.format(phase))
    json.dump(imgCapIdlist, open(all_anno, 'w'))

def generate_mat():
    inst_val = json.load(open('output/val_all_anno.json'))
    # print(inst)
    txt = []
    label = []
    img_url = []
    for x in range(40137):
        tmp = inst_val[x]
        # print(tmp)
        captions = tmp['caption']
        final_str = ''
        for s in captions:
            final_str = final_str + s
        # print(final_str)
        txt.append(final_str)

        label_tmp = tmp['labels']
        one_hot = [0 for i in range(80)]
        for cate in label_tmp:
            one_hot[cate] = 1
        label.append(one_hot)
        url_tmp = img_val_path + '/' + tmp['file_name']
        img_url.append(url_tmp)

    inst_train = json.load(open('output/train_all_anno.json'))
    for x in range(82081):
        tmp = inst_train[x]
        # print(tmp)
        captions = tmp['caption']
        final_str = ''
        for s in captions:
            final_str = final_str + s
        # print(final_str)
        txt.append(final_str)

        # one-hot label vector
        label_tmp = tmp['labels']
        one_hot = [0 for i in range(80)]
        for cate in label_tmp:
            one_hot[cate] = 1
        label.append(one_hot)
        url_tmp = img_train_path +'/'+ tmp['file_name']
        img_url.append(url_tmp)
    #  122218*2000d
    txt_arr = BOW(txt)
    #  122218*80d
    label_arr = np.array(label)
    #  img_url
    index = np.zeros((42500, 1))
    tags = np.zeros((42500, 2000))
    imgs = np.zeros((42500, 256, 256, 3), dtype=np.int32)
    labels = np.zeros((42500, 80))
    val_ind = np.random.randint(low=0, high=40137, size=2500, dtype='l')
    current = 0
    # query set
    for x in val_ind:
        index[current] = x
        tags[current] = txt_arr[x]
        labels[current] = label_arr[x]
        tmp_url = img_url[x]
        print(tmp_url)
        inst_val = cv2.imread(tmp_url)
        assert (inst_val is not None)
        itmp = cv2.resize(inst_val, (256, 256))
        imgs[current] = itmp
        current = current+1
    assert (current == 2500)

    train_ind = np.random.randint(low=40137, high=82081+40137, size=40000, dtype='l')
    # database data
    for x in train_ind:
        index[current] = x
        tags[current] = txt_arr[x]
        labels[current] = label_arr[x]
        tmp_url = img_url[x]
        inst_train = cv2.imread(tmp_url)
        print(tmp_url)
        assert (inst_train is not None)
        itmp = cv2.resize(inst_train, (256, 256))
        imgs[current] = itmp
        current = current+1
    assert (current==42500)

    url = './output/coco.mat'
    f = h5py.File(url, 'w')
    f['Index'] = index
    f['IAll'] = imgs
    f['YAll'] = tags
    f['LAll'] = labels
    f.close()

if __name__ == '__main__':
    generate_train_set()
    generate_val_set()
    generate_mat()




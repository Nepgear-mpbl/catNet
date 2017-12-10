import os
import numpy as np
from PIL import Image
import tensorflow as tf


def get_data_url_set(src_folder='data/train/'):
    data_url = []
    label = []
    for pic in os.listdir(src_folder):
        name = pic.split('.')
        data_url.append(src_folder + pic)
        if name[0] == 'cat':
            label.append(0)
        else:
            label.append(1)
    data_url_list = np.hstack(data_url)
    label_list = np.hstack(label)
    temp = np.array([data_url_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    data_url_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return data_url_list, label_list


def get_batch_data(data_url_list, label_list, batch_size):
    temp = np.array([data_url_list, label_list])
    temp = temp.transpose()
    batch_url = np.random.choice(len(temp), batch_size)
    batch_url_list = []
    batch_label_list = []
    for i in batch_url:
        batch_url_list.append(data_url_list[i])
        batch_label_list.append(label_list[i])
    imgdata = []
    for url in batch_url_list:
        im = Image.open(url)
        x, y = im.size
        if x > y:
            crop_box = ((x - y) / 2, 0, x - ((x - y) / 2), y)
        else:
            crop_box = (0, (y - x) / 2, x, y - ((y - x) / 2))
        im = im.crop(crop_box)
        im = im.resize((208, 208))
        imdata = np.reshape(np.asarray(im, dtype='float32'), [208, 208, 3])
        imgdata.append(imdata)
    batch_data = np.asarray(imgdata, np.float32)
    batch_label = np.asarray(batch_label_list, np.int32)
    batch_label = (np.arange(2) == batch_label[:, None]).astype(np.float32)
    return batch_data, batch_label


def get_next_batch_data(data_url_list, label_list, batch_size, cur_index):
    batch_url_list = data_url_list[cur_index:cur_index + batch_size]
    batch_label_list = label_list[cur_index:cur_index + batch_size]
    cur_index+=batch_size
    cur_index%=len(data_url_list)
    imgdata = []
    for url in batch_url_list:
        im = Image.open(url)
        x, y = im.size
        if x > y:
            crop_box = ((x - y) / 2, 0, x - ((x - y) / 2), y)
        else:
            crop_box = (0, (y - x) / 2, x, y - ((y - x) / 2))
        im = im.crop(crop_box)
        im = im.resize((208, 208))
        imdata = np.reshape(np.asarray(im, dtype='float32'), [208, 208, 3])
        imgdata.append(imdata)
    batch_data = np.asarray(imgdata, np.float32)
    batch_label = np.asarray(batch_label_list, np.int32)
    batch_label = (np.arange(2) == batch_label[:, None]).astype(np.float32)
    return batch_data, batch_label,cur_index


if __name__ == '__main__':
    data_url_list, label_list = get_data_url_set()
    batch_data, batch_label,index = get_next_batch_data(data_url_list, label_list, 16,0)
    print(batch_data.shape, batch_label,index)

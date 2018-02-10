# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:23:56 2018

@author: luckycallor
"""

from sklearn import datasets as ds
import numpy as np
import pickle
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


def resize_images(image_arrays, size=[32, 32]):
    # convert float type to integer
    image_arrays = (image_arrays * 255).astype('uint8')

    resized_image_arrays = np.zeros([image_arrays.shape[0]] + size + [3])
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)
        final_image = np.zeros(size+[3])
        final_image[:,:,0] = final_image[:,:,1] = final_image[:,:,2] = resized_image
        resized_image_arrays[i] = np.asarray(final_image)

    return resized_image_arrays

def resize_cover_images(image_arrays, size=[22, 22]):
    # image_arrays = (image_arrays * 255).astype('uint8')
    ori_size = image_arrays.shape[1]
    new_size = size[0]
    s = (new_size-ori_size)//2
    resized_image_arrays = np.ones([image_arrays.shape[0]] + size) * -1
    resized_image_arrays[:, s: s+ori_size, s: s+ori_size] = image_arrays
    resized_image_arrays = (resized_image_arrays+1)/2
    # resized_image_arrays = (resized_image_arrays*255).astype('unit8')
    return resized_image_arrays

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def main():
    train_path = r'F:\digit-data\usps\usps_train'
    test_path = r'F:\digit-data\usps\usps_test'
    train_x, train_y = ds.load_svmlight_file(train_path)
    test_x, test_y = ds.load_svmlight_file(test_path)
    train_x = np.array(train_x.todense())
    test_x = np.array(test_x.todense())
    train = {'X': resize_images(resize_cover_images(train_x.reshape(-1, 16, 16))),
             'y': np.array([y-1 for y in train_y])}
    
    test = {'X': resize_images(resize_cover_images(test_x.reshape(-1, 16, 16))),
            'y': np.array([y-1 for y in test_y])}
        
    save_pickle(train, r'F:\digit-data\usps_1\train.pkl')
    save_pickle(test, r'F:\digit-data\usps_1\test.pkl')
    
    
if __name__ == "__main__":
    main()
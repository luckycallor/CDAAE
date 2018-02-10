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
        resized_image_arrays[i] = final_image

    return resized_image_arrays

def resize_cover_images(image_arrays, size=[32, 32]):
    ori_size = image_arrays.shape[1]
    new_size = size[0]
    s = (new_size-ori_size)//2
    image_arrays = (image_arrays * 255).astype('uint8')
    resized_image_arrays = np.zeros([image_arrays.shape[0]] + size + [3])
    resized_image_arrays[:, s: s+ori_size, s: s+ori_size, 0] \
    = resized_image_arrays[:, s: s+ori_size, s: s+ori_size, 1] \
    = resized_image_arrays[:, s: s+ori_size, s: s+ori_size, 2] = image_arrays
    return resized_image_arrays

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def main():
    mnist = input_data.read_data_sets('MNIST_data')
    train_x = list(mnist.train.images)
    train_x.extend(list(mnist.validation.images))
    train_x = np.array(train_x)
    train_y = list(mnist.train.labels)
    train_y.extend(list(mnist.validation.labels))
    train_y = np.array(train_y)
    test_x = mnist.test.images
    test_y = mnist.test.labels
    train = {'X': resize_images(train_x.reshape(-1, 28, 28)),
             'y': train_y}
    test = {'X': resize_images(test_x.reshape(-1, 28, 28)),
            'y': test_y}
        
    save_pickle(train, r'F:\mnist\train.pkl')
    save_pickle(test, r'F:\mnist\test.pkl')
    
    
if __name__ == "__main__":
    main()
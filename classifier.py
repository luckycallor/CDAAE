import os
import time
import pickle
import scipy.io
import scipy.misc
import numpy as np
from nets import *


def load_data(image_dir, mode='train'):
    image_file = 'train.pkl' if mode == 'train' else 'test.pkl'
    image_dir = os.path.join(image_dir, image_file)
    print('loading data: %s ...' % image_dir)
    with open(image_dir, 'rb') as f:
        data = pickle.load(f)
    images = data['X'] / 127.5 - 1
    labels = data['y']
    print('finished loading data: %s!' % image_dir)
    return images, labels


def run_acc(sess, prob, images, labels, data_test, label_test, batch_size):
    image_size = data_test.shape[1]
    idxes = list(range(len(data_test)))
    np.random.shuffle(idxes)
    data_test = data_test[idxes]
    label_test = label_test[idxes]
    total = 0
    correct = 0
    for i in range(int(len(data_test)/batch_size)):
        data_batch = data_test[i*batch_size:(i+1)*batch_size]
        label_batch = label_test[i*batch_size:(i+1)*batch_size]
        p = sess.run(prob, {images: data_batch, labels: label_batch})
        pred = np.argmax(p, axis=1)
        total += batch_size
        correct += np.sum(pred == label_batch)
    if(len(data_test) % batch_size > 0):
        num_left = len(data_test) % batch_size
        data_batch = np.zeros([batch_size, image_size, image_size, 3])
        label_batch = np.zeros([batch_size])
        data_batch[:num_left, :, :, :] = data_test[-num_left:]
        label_batch[:num_left] = label_test[-num_left:]
        p = sess.run(prob, {images: data_batch, labels: label_batch})
        pred = np.argmax(p, axis=1)
        total += num_left
        correct += np.sum(pred[:num_left] == label_batch[:num_left])
    acc = correct/total
    return acc

def train(image_size, hot_size, mode, svhn_dir, mnist_dir, model_dir, batch_size, learning_rate,
          max_step):
    images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    labels = tf.placeholder(tf.int32, [None])
    _, logits, prob = Encoder_hot(images, image_size, hot_size, is_training=True, name=mode)
    loss = slim.losses.sparse_softmax_cross_entropy(logits, labels)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = slim.learning.create_train_op(loss, optimizer, variables_to_train=tf.trainable_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if(mode == 'svhn'):
        data_train, label_train = load_data(svhn_dir, mode='train')
        data_test, label_test = load_data(svhn_dir, mode='test')
    elif(mode == 'mnist'):
        data_train, label_train = load_data(mnist_dir, mode='train')
        data_test, label_test = load_data(mnist_dir, mode='test')
    else:
        data_train = []
        label_train = []
        data_tmp, label_tmp = load_data(svhn_dir, mode='train')
        data_train.extend(list(data_tmp))
        label_train.extend(list(label_tmp))
        data_tmp, label_tmp = load_data(mnist_dir, mode='train')
        data_train.extend(list(data_tmp))
        label_train.extend(list(label_tmp))
        data_train = np.array(data_train)
        label_train = np.array(label_train)
        svhn_data_test, svhn_label_test = load_data(svhn_dir, mode='test')
        mnist_data_test, mnist_label_test = load_data(mnist_dir, mode='test')
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        print("start training...")
        step = 0
        start_time = time.time()
        while (step < max_step):
            i = step % int(data_train.shape[0] / batch_size)
            if (i == 0):
                idxes = list(range(len(data_train)))
                np.random.shuffle(idxes)
                data_train = data_train[idxes]
                label_train = label_train[idxes]
            data_batch = data_train[i * batch_size:(i + 1) * batch_size]
            label_batch = label_train[i * batch_size:(i + 1) * batch_size]
            feed_dict = {images: data_batch, labels: label_batch}
            sess.run(train_op, feed_dict)
            step += 1
            if(step % 100 == 0):
                saver.save(sess, os.path.join(model_dir, mode+'-classifier'), global_step=step)
                print('model/'+mode+'-classifier-%d saved' % step)
                if(mode == 'svhn' or mode == 'mnist'):
                    idxes = list(range(len(data_test)))
                    np.random.shuffle(idxes)
                    data_batch = data_test[idxes[:batch_size]]
                    label_batch = label_test[idxes[:batch_size]]
                    p, l = sess.run([prob, loss], {images: data_batch, labels: label_batch})
                    pred = np.argmax(p, axis=1)
                    acc = np.sum(pred == label_batch)/batch_size
                    print("[%d/%d]--[loss:%.3f]--[acc on %s:%.3f]--[time used:%.3f]"
                          %(step, max_step, l, mode, acc, (time.time()-start_time)))
                else:
                    idxes = list(range(len(svhn_data_test)))
                    np.random.shuffle(idxes)
                    data_batch = svhn_data_test[idxes[:batch_size]]
                    label_batch = svhn_label_test[idxes[:batch_size]]
                    p, l1 = sess.run([prob, loss], {images: data_batch, labels: label_batch})
                    pred = np.argmax(p, axis=1)
                    acc_svhn = np.sum(pred == label_batch)/batch_size
                    idxes = list(range(len(mnist_data_test)))
                    np.random.shuffle(idxes)
                    data_batch = mnist_data_test[idxes[:batch_size]]
                    label_batch = mnist_label_test[idxes[:batch_size]]
                    p, l2 = sess.run([prob, loss], {images: data_batch, labels: label_batch})
                    l = (l1+l2)/2.0
                    pred = np.argmax(p, axis=1)
                    acc_mnist = np.sum(pred == label_batch)/batch_size
                    print("[%d/%d]--[loss:%.3f]--[acc on svhn:%.3f]--[acc on mnist:%.3f]--[time used:%.3f]" %
                          (step, max_step, l, acc_svhn, acc_mnist, (time.time()-start_time)))
                start_time = time.time()

def eval(image_size, hot_size, mode, svhn_dir, mnist_dir, result_dir, model_path, batch_size,
         learning_rate):
    images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    labels = tf.placeholder(tf.int32, [None])
    _, logits, prob = Encoder_hot(images, image_size, hot_size, is_training=False, name=mode)
    loss = slim.losses.sparse_softmax_cross_entropy(logits, labels)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = slim.learning.create_train_op(loss, optimizer)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    f_log = open(mode + '_eval.txt', 'w')
    root_result_dir = result_dir

    with tf.Session(config=config) as sess:
        print("loading model...")
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("done!")
        for i in range(1, 101):
            step = i * 200
            result_dir = os.path.join(root_result_dir, str(step))
            if (mode == 'svhn' or mode == 'both'):
                p2s_data = np.array(pickle.load(open(os.path.join(result_dir, 'p2s_image'), 'rb')))
                p2s_label = np.array(pickle.load(open(os.path.join(result_dir, 'p2s_label'), 'rb')))
                s2s_data = np.array(pickle.load(open(os.path.join(result_dir, 's2s_image'), 'rb')))
                s2s_label = np.array(pickle.load(open(os.path.join(result_dir, 's2s_label'), 'rb')))
                t2s_data = np.array(pickle.load(open(os.path.join(result_dir, 't2s_image'), 'rb')))
                t2s_label = np.array(pickle.load(open(os.path.join(result_dir, 't2s_label'), 'rb')))
            if (mode == 'mnist' or mode == 'both'):
                p2t_data = np.array(pickle.load(open(os.path.join(result_dir, 'p2t_image'), 'rb')))
                p2t_label = np.array(pickle.load(open(os.path.join(result_dir, 'p2t_label'), 'rb')))
                t2t_data = np.array(pickle.load(open(os.path.join(result_dir, 't2t_image'), 'rb')))
                t2t_label = np.array(pickle.load(open(os.path.join(result_dir, 't2t_label'), 'rb')))
                s2t_data = np.array(pickle.load(open(os.path.join(result_dir, 's2t_image'), 'rb')))
                s2t_label = np.array(pickle.load(open(os.path.join(result_dir, 's2t_label'), 'rb')))
            svhn_data_test, svhn_label_test = load_data(svhn_dir, mode='test')
            mnist_data_test, mnist_label_test = load_data(mnist_dir, mode='test')
            if (mode == 'svhn' or mode == 'both'):
                svhn_acc = run_acc(sess, prob, images, labels, svhn_data_test, svhn_label_test,
                                   batch_size)
                p2s_acc = run_acc(sess, prob, images, labels, p2s_data, p2s_label, batch_size)
                s2s_acc = run_acc(sess, prob, images, labels, s2s_data, s2s_label, batch_size)
                t2s_acc = run_acc(sess, prob, images, labels, t2s_data, t2s_label, batch_size)
                f_log.write("classifier acc on svhn test data: %f\n" % svhn_acc)
                f_log.write("classifier acc on p2s data: %f\n" % p2s_acc)
                f_log.write("classifier acc on s2s data: %f\n" % s2s_acc)
                f_log.write("classifier acc on t2s data: %f\n" % t2s_acc)
            if (mode == 'mnist' or mode == 'both'):
                mnist_acc = run_acc(sess, prob, images, labels, mnist_data_test, mnist_label_test,
                                    batch_size)
                p2t_acc = run_acc(sess, prob, images, labels, p2t_data, p2t_label, batch_size)
                t2t_acc = run_acc(sess, prob, images, labels, t2t_data, t2t_label, batch_size)
                s2t_acc = run_acc(sess, prob, images, labels, s2t_data, s2t_label, batch_size)
                f_log.write("classifier acc on mnist test data: %f\n" % mnist_acc)
                f_log.write("classifier acc on p2t data: %f\n" % p2t_acc)
                f_log.write("classifier acc on t2t data: %f\n" % t2t_acc)
                f_log.write("classifier acc on s2t data: %f\n" % s2t_acc)
    f_log.close()

if __name__ == "__main__":
    train(image_size=32, hot_size=10, mode='mnist',
          svhn_dir='/data/hhd/svhn',
          mnist_dir='/data/hhd/mnist',
          model_dir='/data/hhd/CrossDomainAdversarialAutoencoder-PaperCode/svhn-mnist/classifier',
          batch_size=100, learning_rate=0.013, max_step=20000)
    tf.reset_default_graph()
    train(image_size=32, hot_size=10, mode='svhn',
          svhn_dir='/data/hhd/svhn',
          mnist_dir='/data/hhd/mnist',
          model_dir='/data/hhd/CrossDomainAdversarialAutoencoder-PaperCode/svhn-mnist/classifier',
          batch_size=100, learning_rate=0.013, max_step=20000)
    tf.reset_default_graph()
    train(image_size=32, hot_size=10, mode='both',
          svhn_dir='/data/hhd/svhn',
          mnist_dir='/data/hhd/mnist',
          model_dir='/data/hhd/CrossDomainAdversarialAutoencoder-PaperCode/svhn-mnist/classifier',
          batch_size=100, learning_rate=0.013, max_step=20000)
    tf.reset_default_graph()
    eval(image_size=32, hot_size=10, mode='mnist',
         svhn_dir='/data/hhd/svhn',
         mnist_dir='/data/hhd/mnist',
         result_dir='/data/hhd/CrossDomainAdversarialAutoEncoder/svhn-mnist/result/1',
         model_path='/data/hhd/CrossDomainAdversarialAutoencoder-PaperCode/svhn-mnist/classifier/mnist-classifier-20000',
         batch_size=100, learning_rate=0.013)
    tf.reset_default_graph()
    eval(image_size=32, hot_size=10, mode='svhn',
         svhn_dir='/data/hhd/svhn',
         mnist_dir='/data/hhd/mnist',
         result_dir='/data/hhd/CrossDomainAdversarialAutoEncoder/svhn-mnist/result/1',
         model_path='/data/hhd/CrossDomainAdversarialAutoencoder-PaperCode/svhn-mnist/classifier/svhn-classifier-20000',
         batch_size=100, learning_rate=0.013)
    tf.reset_default_graph()
    eval(image_size=32, hot_size=10, mode='both',
         svhn_dir='/data/hhd/svhn',
         mnist_dir='/data/hhd/mnist',
         result_dir='/data/hhd/CrossDomainAdversarialAutoEncoder/svhn-mnist/result/1',
         model_path='/data/hhd/CrossDomainAdversarialAutoencoder-PaperCode/svhn-mnist/classifier/both-classifier-20000',
         batch_size=100, learning_rate=0.013)


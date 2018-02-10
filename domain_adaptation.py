import tensorflow as tf
from model import Model
import os
import numpy as np
import scipy.io
import scipy.misc
import pickle
import time
import math


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


def merge_images(batch_size, sources, targets, k=10):
    _, h, w, _ = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([row * h, row * w * 2, 3])

    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h, :] = s
        merged[i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h, :] = t
    return merged


def train(image_size, hot_size, calm_size_src, calm_size_trg, learning_rate, batch_size,
          train_iters, src_dir, trg_dir, log_dir, model_dir, sample_dir, pre_steps=2000,
          boost=False, init_threshold=0.85, scale=10000):
    m = Model(image_size=image_size, hot_size=hot_size, calm_size_src=calm_size_src,
              calm_size_trg=calm_size_trg, learning_rate=learning_rate)
    m.build(is_training=True, alpha1=1.0, alpha2=1.0, alpha3=1.0, alpha4=1.0, beta1=0.1, beta2=0.1,
            beta3=0.1, gamma1=2.0, gamma2=0.15, lambda1=5.0, lambda2=0.5, eta1=0.3, eta2=0.3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    src_train_images_su, src_train_labels_su = load_data(src_dir, mode='train')
    trg_train_images_un, _ = load_data(trg_dir, mode='train')
    src_test_images, src_test_labels = load_data(src_dir, mode='test')
    trg_test_images, trg_test_labels = load_data(trg_dir, mode='test')

    src_samples = []
    trg_samples = []

    for i in range(10):
        cur_src_images = src_test_images[np.where(src_test_labels == i)]
        np.random.shuffle(cur_src_images)
        cur_trg_images = trg_test_images[np.where(trg_test_labels == i)]
        np.random.shuffle(cur_trg_images)
        src_samples.extend(list(cur_src_images[0:100]))
        trg_samples.extend(list(cur_trg_images[0:100]))
    src_samples = np.array(src_samples)
    trg_samples = np.array(trg_samples)
    idxes = []
    for i in range(10):
        for j in range(10):
            idxes.extend([j*100+i*10+k for k in range(10)])
    src_samples = src_samples[idxes]
    trg_samples = trg_samples[idxes]

    trg_train_images_su = []
    trg_train_labels_su = []
    trg_train_images_su_next = []
    trg_train_labels_su_next = []

    with tf.Session(config=config) as sess:
        log_file = open('log_result.txt', 'w')
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())
        saver = tf.train.Saver()
        print("start training...")
        step = 0
        while(step < train_iters):
            start_time = time.time()
            i_su = step % int(src_train_images_su.shape[0]/batch_size)
            if(i_su == 0):
                idxes = list(range(len(src_train_images_su)))
                np.random.shuffle(idxes)
                src_train_images_su = src_train_images_su[idxes]
                src_train_labels_su = src_train_labels_su[idxes]
            src_batch_images_su = src_train_images_su[i_su*batch_size : (i_su+1)*batch_size]
            src_batch_labels_su = src_train_labels_su[i_su*batch_size : (i_su+1)*batch_size]
            j_un = step % int(trg_train_images_un.shape[0]/batch_size)
            if(j_un == 0):
                idxes = list(range(len(trg_train_images_un)))
                np.random.shuffle(idxes)
                trg_train_images_un = trg_train_images_un[idxes]
                if(step > pre_steps):
                    if boost:
                        threshold = init_threshold + (1 - math.exp(-step / scale)) * (1.0 - init_threshold)
                    else:
                        threshold = init_threshold
                    trg_train_images_su_next = []
                    trg_train_labels_su_next = []
                    ph_images = np.zeros([batch_size, image_size, image_size, 3])
                    ph_labels = np.zeros([batch_size])
                    ph_hot = np.zeros([batch_size, hot_size])
                    ph_calm_src = np.zeros([batch_size, calm_size_src])
                    ph_calm_trg = np.zeros([batch_size, calm_size_trg])
                    for k in range(int(len(trg_train_images_un)/batch_size)):
                        trg_batch_images = trg_train_images_un[k*batch_size: (k+1)*batch_size]
                        feed_dict = {m.image_src: ph_images,
                                     m.image_trg: trg_batch_images,
                                     m.label_src: ph_labels,
                                     m.label_trg: ph_labels,
                                     m.prior_hot: ph_hot,
                                     m.prior_calm_src: ph_calm_src,
                                     m.prior_calm_trg: ph_calm_trg}
                        cur_prob = sess.run(m.hot_trg, feed_dict)
                        pred_prob = np.max(cur_prob, axis=1)
                        cur_pred = np.argmax(cur_prob, axis=1)
                        cur_idxes = np.where(pred_prob >= threshold)
                        trg_train_images_su_next.extend(list(trg_batch_images[cur_idxes]))
                        trg_train_labels_su_next.extend(list(cur_pred[cur_idxes]))
                    trg_train_images_su = np.array(trg_train_images_su_next)
                    trg_train_labels_su = np.array(trg_train_labels_su_next)
                    print("evaluated supervised trg: [%d], threshold: %f." % (len(trg_train_labels_su_next), threshold))
                    mnist_images_su_next = []
                    mnist_labels_su_next = []
            trg_batch_images_un = trg_train_images_un[j_un * batch_size: (j_un + 1) * batch_size]
            ph_labels = np.zeros([batch_size])
            hot_code = np.zeros([batch_size, hot_size])
            hot_code[range(batch_size), np.random.randint(0, hot_size, batch_size)] = 1
            # hot_code = np.random.randint(0, 2, [batch_size, hot_size])
            calm_code_src = np.random.randn(batch_size, calm_size_src)
            calm_code_trg = np.random.randn(batch_size, calm_size_trg)
            feed_dict = {m.image_src: src_batch_images_su,
                         m.image_trg: trg_batch_images_un,
                         m.label_src: src_batch_labels_su,
                         m.label_trg: ph_labels,
                         m.prior_hot: hot_code,
                         m.prior_calm_src: calm_code_src,
                         m.prior_calm_trg: calm_code_trg}
            _, summary_su = sess.run([m.train_op_su_src, m.summary_op_su], feed_dict)
            if (step > pre_steps):
                sess.run([m.train_op_auto, m.train_op_cc_suun], feed_dict)
                sess.run([m.train_op_D, m.train_op_Cen, m.train_op_Hen], feed_dict)
            summary = sess.run(m.summary_op, feed_dict)
            step += 1
            summary_writer.add_summary(summary_su, step)
            summary_writer.add_summary(summary, step)
            if (step == pre_steps):
                if boost:
                    threshold = init_threshold + (1 - math.exp(-step / scale)) * (1.0 - init_threshold)
                else:
                    threshold = init_threshold
                trg_train_images_su_next = []
                trg_train_labels_su_next = []
                ph_images = np.zeros([batch_size, image_size, image_size, 3])
                ph_labels = np.zeros([batch_size])
                ph_hot = np.zeros([batch_size, hot_size])
                ph_calm_src = np.zeros([batch_size, calm_size_src])
                ph_calm_trg = np.zeros([batch_size, calm_size_trg])
                for k in range(int(len(trg_train_images_un) / batch_size)):
                    trg_batch_images = trg_train_images_un[k * batch_size: (k + 1) * batch_size]
                    feed_dict = {m.image_src: ph_images,
                                 m.image_trg: trg_batch_images,
                                 m.label_src: ph_labels,
                                 m.label_trg: ph_labels,
                                 m.prior_hot: ph_hot,
                                 m.prior_calm_src: ph_calm_src,
                                 m.prior_calm_trg: ph_calm_trg}
                    cur_prob = sess.run(m.hot_trg, feed_dict)
                    pred_prob = np.max(cur_prob, axis=1)
                    cur_pred = np.argmax(cur_prob, axis=1)
                    cur_idxes = np.where(pred_prob >= threshold)
                    trg_train_images_su_next.extend(list(trg_batch_images[cur_idxes]))
                    trg_train_labels_su_next.extend(list(cur_pred[cur_idxes]))
                trg_train_images_su = np.array(trg_train_images_su_next)
                trg_train_labels_su = np.array(trg_train_labels_su_next)
                print("evaluated supervised trg: [%d], threshold: %f." % (len(trg_train_labels_su_next), threshold))
                mnist_images_su_next = []
                mnist_labels_su_next = []
            if (step > pre_steps):
                j_su = np.random.randint(0, int(len(trg_train_labels_su) / batch_size))
                trg_batch_images_su = trg_train_images_su[j_su * batch_size: (j_su + 1) * batch_size]
                trg_batch_labels_su = trg_train_labels_su[j_su * batch_size: (j_su + 1) * batch_size]
                hot_code = np.zeros([batch_size, hot_size])
                hot_code[range(batch_size), np.random.randint(0, hot_size, batch_size)] = 1
                # hot_code = np.random.randint(0, 2, [batch_size, hot_size])
                calm_code_src = np.random.randn(batch_size, calm_size_src)
                calm_code_trg = np.random.randn(batch_size, calm_size_trg)
                feed_dict = {m.image_src: src_batch_images_su,
                             m.image_trg: trg_batch_images_su,
                             m.label_src: src_batch_labels_su,
                             m.label_trg: trg_batch_labels_su,
                             m.prior_hot: hot_code,
                             m.prior_calm_src: calm_code_src,
                             m.prior_calm_trg: calm_code_trg}
                sess.run(m.train_op_su_trg, feed_dict)
                sess.run([m.train_op_D, m.train_op_Cen, m.train_op_Hen], feed_dict)
                sess.run([m.train_op_auto, m.train_op_cc_suun], feed_dict)
            if (step % 100 == 0):
                saver.save(sess, os.path.join(model_dir, 'm'), global_step=step)
                print('model/m-%d saved' % step)
                calm_code_src = np.array(list(np.random.randn(10, calm_size_src)) * 10)
                calm_code_trg = np.array(list(np.random.randn(10, calm_size_trg)) * 10)
                hot_code = np.zeros([batch_size, hot_size])
                for k in range(10):
                    hot_code[k * 10:(k + 1) * 10, k] = 1
                k = np.random.randint(0, 10)
                src_batch_samples = src_samples[k * batch_size: (k + 1) * batch_size]
                trg_batch_samples = trg_samples[k * batch_size: (k + 1) * batch_size]
                feed_dict = {m.image_src: src_batch_samples,
                             m.image_trg: trg_batch_samples,
                             m.label_src: ph_labels,
                             m.label_trg: ph_labels,
                             m.prior_hot: hot_code,
                             m.prior_calm_src: calm_code_src,
                             m.prior_calm_trg: calm_code_trg}
                fsrc_from_src, ftrg_from_trg, \
                gsrc_from_src, gtrg_from_trg, \
                gsrc_from_trg, gtrg_from_src, \
                gsrc_from_prior, gtrg_from_prior = sess.run([m.fsrc_from_src,
                                                             m.ftrg_from_trg,
                                                             m.gsrc_from_src,
                                                             m.gtrg_from_trg,
                                                             m.gsrc_from_trg,
                                                             m.gtrg_from_src,
                                                             m.gsrc_from_prior,
                                                             m.gtrg_from_prior], feed_dict)
                merged_fsrc_from_src = merge_images(batch_size, src_batch_samples, fsrc_from_src)
                merged_ftrg_from_trg = merge_images(batch_size, trg_batch_samples, ftrg_from_trg)
                merged_gsrc_from_src = merge_images(batch_size, src_batch_samples, gsrc_from_src)
                merged_gtrg_from_trg = merge_images(batch_size, trg_batch_samples, gtrg_from_trg)
                merged_gsrc_from_trg = merge_images(batch_size, trg_batch_samples, gsrc_from_trg)
                merged_gtrg_from_src = merge_images(batch_size, src_batch_samples, gtrg_from_src)
                merged_image_from_prior = merge_images(batch_size, gsrc_from_prior, gtrg_from_prior)
                scipy.misc.imsave(os.path.join(sample_dir, 'fsrc_from_src-%d.png' % step), merged_fsrc_from_src)
                scipy.misc.imsave(os.path.join(sample_dir, 'ftrg_from_trg-%d.png' % step), merged_ftrg_from_trg)
                scipy.misc.imsave(os.path.join(sample_dir, 'gsrc_from_src-%d.png' % step), merged_gsrc_from_src)
                scipy.misc.imsave(os.path.join(sample_dir, 'gtrg_from_trg-%d.png' % step), merged_gtrg_from_trg)
                scipy.misc.imsave(os.path.join(sample_dir, 'gsrc_from_trg-%d.png' % step), merged_gsrc_from_trg)
                scipy.misc.imsave(os.path.join(sample_dir, 'gtrg_from_src-%d.png' % step), merged_gtrg_from_src)
                scipy.misc.imsave(os.path.join(sample_dir, 'image_from_prior-%d.png' % step),
                                  merged_image_from_prior)
            print("step: %d, time used: %.3f" % (step, (time.time() - start_time)))
            start_time = time.time()
        calm_code_src = np.array(list(np.random.randn(10, calm_size_src)) * 10)
        calm_code_trg = np.array(list(np.random.randn(10, calm_size_trg)) * 10)
        hot_code = np.zeros([batch_size, hot_size])
        ph_images = np.zeros([batch_size, image_size, image_size, 3])
        ph_labels = np.zeros([batch_size])
        for k in range(10):
            hot_code[k * 10:(k + 1) * 10, k] = 1
        pred = []
        for k in range(int(trg_test_images.shape[0] / batch_size)):
            trg_batch_images = trg_test_images[k * batch_size:(k + 1) * batch_size]
            feed_dict = {m.image_src: ph_images,
                         m.image_trg: trg_batch_images,
                         m.label_src: ph_labels,
                         m.label_trg: ph_labels,
                         m.prior_hot: hot_code,
                         m.prior_calm_src: calm_code_src,
                         m.prior_calm_trg: calm_code_trg}
            p = sess.run(m.hot_trg, feed_dict)
            p = np.argmax(p, axis=1)
            pred.extend(list(p))
        num_left = trg_test_images.shape[0] % batch_size
        if (num_left > 0):
            trg_batch_images = np.zeros([batch_size, image_size, image_size, 3])
            trg_batch_images[:num_left, :, :, :] = trg_test_images[-num_left:]
            feed_dict = {m.image_src: ph_images,
                         m.image_trg: trg_batch_images,
                         m.label_src: ph_labels,
                         m.label_trg: ph_labels,
                         m.prior_hot: hot_code,
                         m.prior_calm_src: calm_code_src,
                         m.prior_calm_trg: calm_code_trg}
            p = sess.run(m.hot_trg, feed_dict)
            p = np.argmax(p, axis=1)
            pred.extend(list(p)[:num_left])
        pred = np.array(pred)
        acc = np.mean(pred == trg_test_labels)
        log_file.write("step:[%d/%d]--acc:[%f]\n" % (step, train_iters, acc))
        print("step:[%d/%d]--acc:[%f]\n" % (step, train_iters, acc))
        summary_writer.close()
        log_file.close()


if __name__ == "__main__":
    train(image_size=32, hot_size=10, calm_size_src=8, calm_size_trg=8,
          learning_rate=0.0013, batch_size=100, train_iters=5000,
          src_dir='/data/hhd/digit-data/svhn', trg_dir='/data/hhd/digit-data/mnist',
          log_dir='/data/hhd/CrossDomainAdversarialAutoEncoder/domain-adaptation/svhn2mnist/log',
          model_dir='/data/hhd/CrossDomainAdversarialAutoEncoder/domain-adaptation/svhn2mnist/model',
          sample_dir='/data/hhd/CrossDomainAdversarialAutoEncoder/domain-adaptation/svhn2mnist/sample',
          boost=False, pre_steps=1300, init_threshold=0.85, scale=10000)
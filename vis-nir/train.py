import tensorflow as tf
from model import Model
import os
import numpy as np
import scipy.io
import scipy.misc
import pickle
import time


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

def train(image_size, cat_num, hot_size, calm_size_src, calm_size_trg, learning_rate, batch_size,
          train_iters, data_dir, log_dir, model_dir, sample_dir):
    m = Model(cat_num=cat_num, image_size=image_size, hot_size=hot_size, calm_size_src=calm_size_src,
              calm_size_trg=calm_size_trg, learning_rate=learning_rate)
    m.build(is_training=True, alpha1=1.0, alpha2=1.0, alpha3=1.0, alpha4=1.0, beta1=0.5, beta2=0.5,
            beta3=0.5, gamma1=1.0, gamma2=1.0, lambda1=1.0, lambda2=1.0, eta1=1.0, eta2=1.0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    print("loading data...")
    vis_images_train = pickle.load(open(os.path.join(data_dir, 'image_vis_train'), 'rb'))
    vis_labels_train = pickle.load(open(os.path.join(data_dir, 'label_vis_train'), 'rb'))
    vis_images_test = pickle.load(open(os.path.join(data_dir, 'image_vis_test'), 'rb'))
    vis_labels_test = pickle.load(open(os.path.join(data_dir, 'label_vis_test'), 'rb'))
    nir_images_train = pickle.load(open(os.path.join(data_dir, 'image_nir_train'), 'rb'))
    nir_labels_train = pickle.load(open(os.path.join(data_dir, 'label_nir_train'), 'rb'))
    nir_images_test = pickle.load(open(os.path.join(data_dir, 'image_nir_test'), 'rb'))
    nir_labels_test = pickle.load(open(os.path.join(data_dir, 'label_nir_test'), 'rb'))
    print("done!")
    vis_images_train.extend(vis_images_test[100:])
    vis_images_train = np.array(vis_images_train)
    vis_images_train = vis_images_train/127.5-1
    vis_labels_train.extend(vis_labels_test[100:])
    vis_labels_train = np.array(vis_labels_train)
    nir_images_train.extend(nir_images_test[100:])
    nir_images_train = np.array(nir_images_train)
    nir_images_train = nir_images_train/127.5-1
    nir_labels_train.extend(nir_labels_test[100:])
    nir_labels_train = np.array(nir_labels_train)
    vis_images_test = np.array(vis_images_test[:100])
    vis_images_test = vis_images_test/127.5-1
    vis_labels_test = np.array(vis_labels_test[:100])
    nir_images_test = np.array(nir_images_test[:100])
    nir_images_test = nir_images_test/127.5-1
    nir_labels_test = np.array(nir_labels_test[:100])
    np.random.shuffle(vis_images_test)
    np.random.shuffle(nir_images_test)

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())
        saver = tf.train.Saver()
        print("start training...")
        step = 0
        while(step < train_iters):
            start_time = time.time()
            i = step % int(vis_images_train.shape[0]/batch_size)
            if(i == 0):
                idxes = list(range(vis_images_train.shape[0]))
                np.random.shuffle(idxes)
                vis_images_train = vis_images_train[idxes]
                vis_labels_train = vis_labels_train[idxes]
            src_images = vis_images_train[i*batch_size : (i+1)*batch_size]
            src_labels = vis_labels_train[i*batch_size : (i+1)*batch_size]
            j = step % int(nir_images_train.shape[0]/batch_size)
            if(j == 0):
                idxes = list(range(nir_images_train.shape[0]))
                np.random.shuffle(idxes)
                nir_images_train = nir_images_train[idxes]
                nir_labels_train = nir_labels_train[idxes]
            trg_images = nir_images_train[j*batch_size : (j+1)*batch_size]
            trg_labels = nir_labels_train[j*batch_size : (j+1)*batch_size]
            # i = step % int(photo_images.shape[0] / batch_size)
            # if(i == 0):
            #     np.random.shuffle(photo_images)
            # src_images = photo_images[i * batch_size:(i + 1) * batch_size]
            # j = step % int(carct_images.shape[0] / batch_size)
            # if(j == 0):
            #     np.random.shuffle(carct_images)
            # trg_images = carct_images[j * batch_size:(j + 1) * batch_size]
            # hot_code = np.random.randn(batch_size, hot_size)
            # hot_code = np.random.randint(0, 2, [batch_size, hot_size])
            # hot_code = np.zeros([batch_size, hot_size])
            # hot_code[range(batch_size), np.random.randint(0, hot_size, batch_size)] = 1
            hot_code = np.random.randn(batch_size, hot_size)
            # calm_code_src = np.random.randint(0, 2, [batch_size, calm_size_src])
            # calm_code_trg = np.random.randint(0, 2, [batch_size, calm_size_trg])
            calm_code_src = np.random.randn(batch_size, calm_size_src)
            calm_code_trg = np.random.randn(batch_size, calm_size_trg)
            feed_dict = {m.image_src: src_images,
                         m.image_trg: trg_images,
                         m.label_src: src_labels,
                         m.label_trg: trg_labels,
                         m.prior_hot: hot_code,
                         m.prior_calm_src: calm_code_src,
                         m.prior_calm_trg: calm_code_trg}
            # sess.run([m.train_op_auto, m.train_op_D, m.train_op_Hen, m.train_op_Cen, m.train_op_cc_un],
            #          feed_dict)
            sess.run([m.train_op_D, m.train_op_Cen], feed_dict)
            _, summary_su = sess.run([m.train_op_su, m.summary_op_su], feed_dict)
            sess.run([m.train_op_auto, m.train_op_cc_su], feed_dict)
            # feed_dict = {m.image_src: src_images,
            #              m.image_trg: trg_images,
            #              m.label_src: src_labels_su,
            #              m.label_trg: trg_labels_su,
            #              m.prior_hot: hot_code,
            #              m.prior_calm_src: calm_code_src,
            #              m.prior_calm_trg: calm_code_trg}
            # sess.run([m.train_op_auto, m.train_op_D, m.train_op_en], feed_dict)
            summary = sess.run(m.summary_op, feed_dict)
            step += 1
            summary_writer.add_summary(summary_su, step)
            summary_writer.add_summary(summary, step)
            print("step: %d, time used: %.3f" % (step, (time.time()-start_time)))
            start_time = time.time()
            if (step % 100 == 0):
                saver.save(sess, os.path.join(model_dir, 'm'), global_step=step)
                print('model/m-%d saved' % step)
                k = np.random.randint(0, 10)
                # hot_code = np.zeros([10, 10, hot_size])
                # hot_code[:] = np.random.randn(10, hot_size)[:, None, :]
                # hot_code = np.resize(hot_code, [batch_size, hot_size])
                # hot_code = np.zeros([batch_size, hot_size])
                # hot_code[range(batch_size), np.random.randint(0, hot_size, batch_size)] = 1
                hot_code = np.random.randn(batch_size, hot_size)
                calm_code_src = np.zeros([10, 10, calm_size_src])
                calm_code_src[:] = np.random.randn(10, calm_size_src)[:, None, :]
                calm_code_src = np.resize(calm_code_src, [batch_size, calm_size_src])
                calm_code_trg = np.zeros([10, 10, calm_size_trg])
                calm_code_trg[:] = np.random.randint(0, 2, [10, calm_size_trg])[:, None, :]
                calm_code_trg = np.resize(calm_code_trg, [batch_size, calm_size_trg])
                src_images = np.array(list(vis_images_test[k * 10:(k + 1) * 10]) * 10)
                trg_images = np.array(list(nir_images_test[k * 10:(k + 1) * 10]) * 10)
                feed_dict = {m.image_src: src_images,
                             m.image_trg: trg_images,
                             m.label_src: src_labels,
                             m.label_trg: trg_labels,
                             m.prior_hot: hot_code,
                             m.prior_calm_src: calm_code_src,
                             m.prior_calm_trg: calm_code_trg}
                fsrc_from_src, ftrg_from_trg, \
                gsrc_from_src, gtrg_from_trg, \
                gsrc_from_trg, gtrg_from_src,\
                gsrc_from_prior, gtrg_from_prior = sess.run([m.fsrc_from_src,
                                                             m.ftrg_from_trg,
                                                             m.gsrc_from_src,
                                                             m.gtrg_from_trg,
                                                             m.gsrc_from_trg,
                                                             m.gtrg_from_src,
                                                             m.gsrc_from_prior,
                                                             m.gtrg_from_prior], feed_dict)
                merged_fsrc_from_src = merge_images(batch_size, src_images, fsrc_from_src)
                merged_ftrg_from_trg = merge_images(batch_size, trg_images, ftrg_from_trg)
                merged_gsrc_from_src = merge_images(batch_size, src_images, gsrc_from_src)
                merged_gtrg_from_trg = merge_images(batch_size, trg_images, gtrg_from_trg)
                merged_gsrc_from_trg = merge_images(batch_size, trg_images, gsrc_from_trg)
                merged_gtrg_from_src = merge_images(batch_size, src_images, gtrg_from_src)
                merged_image_from_prior = merge_images(batch_size, gsrc_from_prior, gtrg_from_prior)
                scipy.misc.imsave(os.path.join(sample_dir, 'fsrc_from_src-%d.png' % step), merged_fsrc_from_src)
                scipy.misc.imsave(os.path.join(sample_dir, 'ftrg_from_trg-%d.png' % step), merged_ftrg_from_trg)
                scipy.misc.imsave(os.path.join(sample_dir, 'gsrc_from_src-%d.png' % step), merged_gsrc_from_src)
                scipy.misc.imsave(os.path.join(sample_dir, 'gtrg_from_trg-%d.png' % step), merged_gtrg_from_trg)
                scipy.misc.imsave(os.path.join(sample_dir, 'gsrc_from_trg-%d.png' % step), merged_gsrc_from_trg)
                scipy.misc.imsave(os.path.join(sample_dir, 'gtrg_from_src-%d.png' % step), merged_gtrg_from_src)
                scipy.misc.imsave(os.path.join(sample_dir, 'image_from_prior-%d.png' % step), merged_image_from_prior)
        summary_writer.close()

if __name__ == "__main__":
    train(image_size=160, cat_num=582, hot_size=512, calm_size_src=64, calm_size_trg=64, learning_rate=0.0003,
          batch_size=100, train_iters=20000, data_dir=r'/data/hhd/vis-nir',
          log_dir=r'/data/hhd/CrossDomainAdversarialAutoencoder-PaperCode/vis-nir/log/3',
          model_dir=r'/data/hhd/CrossDomainAdversarialAutoencoder-PaperCode/vis-nir/model/3',
          sample_dir=r'/data/hhd/CrossDomainAdversarialAutoencoder-PaperCode/vis-nir/sample/3')
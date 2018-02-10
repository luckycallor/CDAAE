import tensorflow as tf
from model import Model
import os
import numpy as np
import scipy.io
import scipy.misc
import pickle
import time

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
          train_iters, src_dir, trg_dir, log_dir, model_dir, sample_dir, semi_supervise=False,
          labeled_image_num=-1, pre_steps=0):
    m = Model(image_size=image_size, hot_size=hot_size, calm_size_src=calm_size_src,
              calm_size_trg=calm_size_trg, learning_rate=learning_rate)
    m.build(is_training=True, alpha1=1.0, alpha2=1.0, alpha3=1.0, alpha4=1.0, beta1=0.0, beta2=0.0,
            beta3=1.0, gamma1=2.0, gamma2=0.15, lambda1=5.0, lambda2=0.5, eta1=0.3, eta2=0.3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    src_train_images, src_train_labels = load_data(src_dir, mode='train')
    trg_train_images, trg_train_labels = load_data(trg_dir, mode='train')
    src_test_images, src_test_labels = load_data(src_dir, mode='test')
    trg_test_images, trg_test_labels = load_data(trg_dir, mode='test')

    src_train_images_su = []
    src_train_labels_su = []
    trg_train_images_su = []
    trg_train_labels_su = []
    src_train_images_un = []
    trg_train_images_un = []
    if(semi_supervise):
        for i in range(10):
            cur_src_images = src_train_images[np.where(src_train_labels == i)]
            np.random.shuffle(cur_src_images)
            src_train_images_su.extend(list(cur_src_images[0:labeled_image_num]))
            src_train_images_un.extend(list(cur_src_images[labeled_image_num: ]))
            src_train_labels_su.extend([i]*labeled_image_num)
            cur_trg_images = trg_train_images[np.where(trg_train_labels == i)]
            np.random.shuffle(cur_trg_images)
            trg_train_images_su.extend(list(cur_trg_images[0:labeled_image_num]))
            trg_train_images_un.extend(list(cur_trg_images[labeled_image_num: ]))
            trg_train_labels_su.extend([i]*labeled_image_num)
        src_train_images_su = np.array(src_train_images_su)
        src_train_labels_su = np.array(src_train_labels_su)
        trg_train_images_su = np.array(trg_train_images_su)
        trg_train_labels_su = np.array(trg_train_labels_su)
        src_train_images_un = np.array(src_train_images_un)
        trg_train_images_un = np.array(trg_train_images_un)
    else:
        src_train_images_su = src_train_images
        src_train_labels_su = src_train_labels
        trg_train_images_su = trg_train_images
        trg_train_labels_su = trg_train_labels
        svhn_images_un = None
        mnist_images_un = None

    del src_train_images
    del src_train_labels
    del trg_train_images
    del trg_train_labels

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

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=300)
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
            j_su = step % int(trg_train_images_su.shape[0]/batch_size)
            if(j_su == 0):
                idxes = list(range(len(trg_train_images_su)))
                np.random.shuffle(idxes)
                trg_train_images_su = trg_train_images_su[idxes]
                trg_train_labels_su = trg_train_labels_su[idxes]
            trg_batch_images_su = trg_train_images_su[j_su*batch_size: (j_su+1)*batch_size]
            trg_batch_labels_su = trg_train_labels_su[j_su*batch_size: (j_su+1)*batch_size]
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
            sess.run([m.train_op_D, m.train_op_Cen], feed_dict)
            _, summary_su = sess.run([m.train_op_su, m.summary_op_su], feed_dict)
            sess.run(m.train_op_auto, feed_dict)
            sess.run([m.train_op_cc_su], feed_dict)
            if(semi_supervise and step > pre_steps):
                i = step % int(src_train_images_un.shape[0] / batch_size)
                if (i == 0):
                    np.random.shuffle(src_train_images_un)
                src_batch_images_un = src_train_images_un[i * batch_size:(i + 1) * batch_size]
                j = step % int(trg_train_images_un.shape[0] / batch_size)
                if (j == 0):
                    np.random.shuffle(trg_train_images_un)
                trg_batch_images_un = trg_train_images_un[j * batch_size:(j + 1) * batch_size]
                hot_code = np.zeros([batch_size, hot_size])
                hot_code[range(batch_size), np.random.randint(0, hot_size, batch_size)] = 1
                # hot_code = np.random.randint(0, 2, [batch_size, hot_size])
                calm_code_src = np.random.randn(batch_size, calm_size_src)
                calm_code_trg = np.random.randn(batch_size, calm_size_trg)
                ph_labels = np.zeros(batch_size)
                feed_dict = {m.image_src: src_batch_images_un,
                            m.image_trg: trg_batch_images_un,
                            m.label_src: ph_labels,
                            m.label_trg: ph_labels,
                            m.prior_hot: hot_code,
                            m.prior_calm_src: calm_code_src,
                            m.prior_calm_trg: calm_code_trg}
                sess.run([m.train_op_auto, m.train_op_D, m.train_op_Hen, m.train_op_Cen, m.train_op_cc_un],
                         feed_dict)
            summary = sess.run(m.summary_op, feed_dict)
            step += 1
            summary_writer.add_summary(summary_su, step)
            summary_writer.add_summary(summary, step)
            print("step: %d, time used: %.3f" % (step, (time.time()-start_time)))
            start_time = time.time()
            if (step % 100 == 0):
                if(step % 200 == 0):
                    saver.save(sess, os.path.join(model_dir, 'm'), global_step=step)
                    print('model/m-%d saved' % step)
                calm_code_src = np.array(list(np.random.randn(10, calm_size_src))*10)
                calm_code_trg = np.array(list(np.random.randn(10, calm_size_trg))*10)
                hot_code = np.zeros([batch_size, hot_size])
                for k in range(10):
                    hot_code[k*10:(k+1)*10, k] = 1
                k = np.random.randint(0, 10)
                ph_labels = np.zeros(batch_size)
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
        summary_writer.close()


def eval(image_size, hot_size, calm_size_src, calm_size_trg, learning_rate, batch_size,
         model_path, src_dir, trg_dir, result_dir, num_per_class, num_per_image):
    m = Model(image_size=image_size, hot_size=hot_size, calm_size_src=calm_size_src,
              calm_size_trg=calm_size_trg, learning_rate=learning_rate)
    m.build(is_training=False, alpha1=1.0, alpha2=1.0, alpha3=1.0, alpha4=1.0, beta1=0.0, beta2=0.0,
            beta3=1.0, gamma1=2.0, gamma2=0.15, lambda1=5.0, lambda2=0.5, eta1=0.3, eta2=0.3)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    src_test_images, src_test_labels = load_data(src_dir, mode='test')
    trg_test_images, trg_test_labels = load_data(trg_dir, mode='test')
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    ori_path = os.path.join(result_dir, 'ori')
    if not os.path.isdir(ori_path):
        os.mkdir(ori_path)
    with tf.Session(config=config) as sess:
        print("loading model...")
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("done!")
        prior_calm_src = list(np.random.randn(num_per_image, calm_size_src))
        prior_calm_trg = list(np.random.randn(num_per_image, calm_size_trg))
        s2s_image = []
        t2t_image = []
        s2t_image = []
        t2s_image = []
        p2s_image = []
        p2t_image = []
        s2s_label = []
        t2t_label = []
        s2t_label = []
        t2s_label = []
        p2s_label = []
        p2t_label = []
        ph_label = np.zeros(batch_size)
        ph_image = np.zeros([batch_size, image_size, image_size, 3])
        batch_prior_hot = np.zeros([batch_size, hot_size])
        for i in range(int(trg_test_images.shape[0]/batch_size)):
            trg_image_batch = trg_test_images[i*batch_size: (i+1)*batch_size]
            trg_label_batch = trg_test_labels[i*batch_size: (i+1)*batch_size]
            prior_calm_src = np.random.randn(batch_size, calm_size_src)
            prior_calm_trg = np.random.randn(batch_size, calm_size_trg)
            feed_dict = {m.image_src: ph_image,
                         m.image_trg: trg_image_batch,
                         m.label_src: ph_label,
                         m.label_trg: ph_label,
                         m.prior_hot: batch_prior_hot,
                         m.prior_calm_src: prior_calm_src,
                         m.prior_calm_trg: prior_calm_trg}
            gsrc_from_trg, gtrg_from_trg = sess.run([m.gsrc_from_trg, m.gtrg_from_trg], feed_dict)
            t2s_image.extend(list(gsrc_from_trg))
            t2s_label.extend(list(trg_label_batch))
            t2t_image.extend(list(gtrg_from_trg))
            t2t_label.extend(list(trg_label_batch))
        if(trg_test_images.shape[0]%batch_size > 0):
            trg_image_batch = np.zeros([batch_size, image_size, image_size, 3])
            trg_label_batch = np.zeros([batch_size])
            num_left = trg_test_images.shape[0]%batch_size
            trg_image_batch[:num_left, :, :, :] = trg_test_images[-num_left:]
            trg_label_batch[:num_left] = trg_test_labels[-num_left:]
            prior_calm_src = np.random.randn(batch_size, calm_size_src)
            prior_calm_trg = np.random.randn(batch_size, calm_size_trg)
            feed_dict = {m.image_src: ph_image,
                         m.image_trg: trg_image_batch,
                         m.label_src: ph_label,
                         m.label_trg: ph_label,
                         m.prior_hot: batch_prior_hot,
                         m.prior_calm_src: prior_calm_src,
                         m.prior_calm_trg: prior_calm_trg}
            gsrc_from_trg, gtrg_from_trg = sess.run([m.gsrc_from_trg, m.gtrg_from_trg], feed_dict)
            t2s_image.extend(list(gsrc_from_trg)[:num_left])
            t2s_label.extend(list(trg_label_batch)[:num_left])
            t2t_image.extend(list(gtrg_from_trg)[:num_left])
            t2t_label.extend(list(trg_label_batch)[:num_left])
        pickle.dump(t2t_image, open(os.path.join(result_dir, 't2t_image'), 'wb'), True)
        pickle.dump(t2t_label, open(os.path.join(result_dir, 't2t_label'), 'wb'), True)
        pickle.dump(t2s_image, open(os.path.join(result_dir, 't2s_image'), 'wb'), True)
        pickle.dump(t2s_label, open(os.path.join(result_dir, 't2s_label'), 'wb'), True)
        for i in range(int(src_test_images.shape[0]/batch_size)):
            src_image_batch = src_test_images[i*batch_size: (i+1)*batch_size]
            src_label_batch = src_test_labels[i*batch_size: (i+1)*batch_size]
            prior_calm_src = np.random.randn(batch_size, calm_size_src)
            prior_calm_trg = np.random.randn(batch_size, calm_size_trg)
            feed_dict = {m.image_src: src_image_batch,
                         m.image_trg: ph_image,
                         m.label_src: ph_label,
                         m.label_trg: ph_label,
                         m.prior_hot: batch_prior_hot,
                         m.prior_calm_src: prior_calm_src,
                         m.prior_calm_trg: prior_calm_trg}
            gtrg_from_src, gsrc_from_src = sess.run([m.gtrg_from_src, m.gsrc_from_src], feed_dict)
            s2t_image.extend(list(gtrg_from_src))
            s2t_label.extend(list(src_label_batch))
            s2s_image.extend(list(gsrc_from_src))
            s2s_label.extend(list(src_label_batch))
        if(src_test_images.shape[0] % batch_size > 0):
            src_image_batch = np.zeros([batch_size, image_size, image_size, 3])
            src_label_batch = np.zeros([batch_size])
            num_left = src_test_images.shape[0] % batch_size
            src_image_batch[:num_left, :, :, :] = src_test_images[-num_left:]
            src_label_batch[:num_left] = src_test_labels[-num_left:]
            prior_calm_src = np.random.randn(batch_size, calm_size_src)
            prior_calm_trg = np.random.randn(batch_size, calm_size_trg)
            feed_dict = {m.image_src: src_image_batch,
                         m.image_trg: ph_image,
                         m.label_src: ph_label,
                         m.label_trg: ph_label,
                         m.prior_hot: batch_prior_hot,
                         m.prior_calm_src: prior_calm_src,
                         m.prior_calm_trg: prior_calm_trg}
            gtrg_from_src, gsrc_from_src = sess.run([m.gtrg_from_src, m.gsrc_from_src], feed_dict)
            s2t_image.extend(list(gtrg_from_src)[:num_left])
            s2t_label.extend(list(src_label_batch)[:num_left])
            s2s_image.extend(list(gsrc_from_src)[:num_left])
            s2s_label.extend(list(src_label_batch)[:num_left])
        pickle.dump(s2s_image, open(os.path.join(result_dir, 's2s_image'), 'wb'), True)
        pickle.dump(s2s_label, open(os.path.join(result_dir, 's2s_label'), 'wb'), True)
        pickle.dump(s2t_image, open(os.path.join(result_dir, 's2t_image'), 'wb'), True)
        pickle.dump(s2t_label, open(os.path.join(result_dir, 's2t_label'), 'wb'), True)
        for i in range(hot_size):
            prior_hot = np.zeros([batch_size, hot_size])
            prior_hot[:, i] = 1
            rounds = int(num_per_image*num_per_class/batch_size)
            for j in range(rounds):
                batch_calm_src = np.random.randn(batch_size, calm_size_src)
                batch_calm_trg = np.random.randn(batch_size, calm_size_trg)
                feed_dict = {m.image_src: ph_image,
                             m.image_trg: ph_image,
                             m.label_src: ph_label,
                             m.label_trg: ph_label,
                             m.prior_hot: prior_hot,
                             m.prior_calm_src: batch_calm_src,
                             m.prior_calm_trg: batch_calm_trg}
                p2s, p2t = sess.run([m.gsrc_from_prior,
                                     m.gtrg_from_prior], feed_dict=feed_dict)
                p2s_image.extend(list(p2s))
                p2s_label.extend([i]*batch_size)
                p2t_image.extend(list(p2t))
                p2t_label.extend([i]*batch_size)
        pickle.dump(p2s_image, open(os.path.join(result_dir, 'p2s_image'), 'wb'), True)
        pickle.dump(p2s_label, open(os.path.join(result_dir, 'p2s_label'), 'wb'), True)
        pickle.dump(p2t_image, open(os.path.join(result_dir, 'p2t_image'), 'wb'), True)
        pickle.dump(p2t_label, open(os.path.join(result_dir, 'p2t_label'), 'wb'), True)


if __name__ == "__main__":
    train(image_size=32, hot_size=10, calm_size_src=8, calm_size_trg=8,
          learning_rate=0.0003, batch_size=100, train_iters=20000,
          src_dir='/data/hhd/digit-data/svhn', trg_dir='/data/hhd/digit-data/mnist',
          log_dir='/data/hhd/CrossDomainAdversarialAutoEncoder/svhn-mnist/log/1',
          model_dir='/data/hhd/CrossDomainAdversarialAutoEncoder/svhn-mnist/model/1',
          sample_dir='/data/hhd/CrossDomainAdversarialAutoEncoder/svhn-mnist/sample/1',
          semi_supervise=True, labeled_image_num=100)
    for i in range(1, 101):
        step = i*200
        tf.reset_default_graph()
        eval(image_size=32, hot_size=10, calm_size_src=8, calm_size_trg=8,
             learning_rate=0.0003, batch_size=100,
             model_path='/data/hhd/CrossDomainAdversarialAutoEncoder/svhn-mnist/model/1/m-'+str(step),
             src_dir='/data/hhd/digit-data/svhn',
             trg_dir='/data/hhd/digit-data/mnist',
             result_dir='/data/hhd/CrossDomainAdversarialAutoEncoder/svhn-mnist/result/1'+'//'+str(step),
             num_per_class=20, num_per_image=50)
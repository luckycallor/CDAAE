from nets import *

class Model:

    def __init__(self, cat_num, image_size, hot_size, calm_size_src, calm_size_trg, learning_rate):
        self.cat_num = cat_num
        self.image_size = image_size
        self.hot_size = hot_size
        self.calm_size_src = calm_size_src
        self.calm_size_trg = calm_size_trg
        self.learning_rate = learning_rate

    def build(self, is_training=True, alpha1=1.0, alpha2=1.0, alpha3=1.0, alpha4=1.0, beta1=1.0,
              beta2=1.0, beta3=1.0, gamma1=1.0, gamma2=1.0, lambda1=1.0, lambda2=1.0, eta1=1.0, eta2=1.0):
        self.image_src = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3],
                                        name='image_src')
        self.image_trg = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3],
                                        name='image_trg')
        self.label_src = tf.placeholder(tf.int32, [None], name='label_src')
        self.label_trg = tf.placeholder(tf.int32, [None], name='label_trg')
        self.prior_hot = tf.placeholder(tf.float32, [None, self.hot_size], name='prior_hot')
        self.prior_calm_src = tf.placeholder(tf.float32, [None, self.calm_size_src],
                                             name='prior_calm_src')
        self.prior_calm_trg = tf.placeholder(tf.float32, [None, self.calm_size_trg],
                                             name='prior_calm_trg')

        self.code_mid_src, self.hot_src, self.logits_src, self.prob_src = \
            Encoder_hot(self.image_src, self.image_size, self.cat_num, self.hot_size, is_training=is_training,
                        name='Encoder_hot')
        self.code_mid_trg, self.hot_trg, self.logits_trg, self.prob_trg = \
            Encoder_hot(self.image_trg, self.image_size, self.cat_num, self.hot_size, reuse=True,
                        is_training=is_training, name='Encoder_hot')
        self.calm_src = Encoder_calm(self.code_mid_src, self.calm_size_src,
                                     is_training=is_training, name='Encoder_calm_src')
        self.calm_trg = Encoder_calm(self.code_mid_trg, self.calm_size_trg,
                                     is_training=is_training, name='Encoder_calm_trg')

        self.pred_src = tf.cast(tf.arg_max(self.prob_src, 1), tf.int32)
        self.pred_trg = tf.cast(tf.arg_max(self.prob_trg, 1), tf.int32)
        self.acc_src = tf.reduce_mean(tf.cast(tf.equal(self.pred_src, self.label_src), tf.float32))
        self.acc_trg = tf.reduce_mean(tf.cast(tf.equal(self.pred_trg, self.label_trg), tf.float32))

        self.code_src = tf.concat([self.hot_src, self.calm_src], axis=1)
        self.code_trg = tf.concat([self.hot_trg, self.calm_trg], axis=1)
        self.code_s2s = tf.concat([self.hot_src, self.prior_calm_src], axis=1)
        self.code_t2t = tf.concat([self.hot_trg, self.prior_calm_trg], axis=1)
        self.code_s2t = tf.concat([self.hot_src, self.prior_calm_trg], axis=1)
        self.code_t2s = tf.concat([self.hot_trg, self.prior_calm_src], axis=1)
        self.code_sprior = tf.concat([self.prior_hot, self.prior_calm_src], axis=1)
        self.code_tprior = tf.concat([self.prior_hot, self.prior_calm_trg], axis=1)

        self.fsrc_from_src = Decoder_ctnn_shallow(self.code_src, self.image_size,
                                                  self.hot_size+self.calm_size_src,
                                                  is_training=is_training, name='Decoder_src')
        self.ftrg_from_trg = Decoder_ctnn_shallow(self.code_trg, self.image_size,
                                                  self.hot_size+self.calm_size_trg,
                                                  is_training=is_training, name='Decoder_trg')
        self.gsrc_from_src = Decoder_ctnn_shallow(self.code_s2s, self.image_size,
                                                  self.hot_size+self.calm_size_src, reuse=True,
                                                  is_training=is_training, name='Decoder_src')
        self.gtrg_from_trg = Decoder_ctnn_shallow(self.code_t2t, self.image_size,
                                                  self.hot_size+self.calm_size_trg, reuse=True,
                                                  is_training=is_training, name='Decoder_trg')
        self.gsrc_from_trg = Decoder_ctnn_shallow(self.code_t2s, self.image_size,
                                                  self.hot_size+self.calm_size_src, reuse=True,
                                                  is_training=is_training, name='Decoder_src')
        self.gtrg_from_src = Decoder_ctnn_shallow(self.code_s2t, self.image_size,
                                                  self.hot_size+self.calm_size_trg, reuse=True,
                                                  is_training=is_training, name='Decoder_trg')
        self.gsrc_from_prior = Decoder_ctnn_shallow(self.code_sprior, self.image_size,
                                                    self.hot_size+self.calm_size_src, reuse=True,
                                                    is_training=is_training, name='Decoder_src')
        self.gtrg_from_prior = Decoder_ctnn_shallow(self.code_tprior, self.image_size,
                                                    self.hot_size+self.calm_size_trg, reuse=True,
                                                    is_training=is_training, name='Decoder_trg')

        self.code_mid_s2t, self.hot_s2t, self.logits_s2t, self.prob_s2t = \
            Encoder_hot(self.gtrg_from_src, self.image_size, self.cat_num, self.hot_size, is_training=is_training,
                        reuse=True, name='Encoder_hot')
        self.code_mid_t2s, self.hot_t2s, self.logits_t2s, self.prob_t2s = \
            Encoder_hot(self.gsrc_from_trg, self.image_size, self.cat_num, self.hot_size, is_training=is_training,
                        reuse=True, name='Encoder_hot')

        self.pred_s2t = tf.cast(tf.arg_max(self.prob_s2t, 1), tf.int32)
        self.pred_t2s = tf.cast(tf.arg_max(self.prob_t2s, 1), tf.int32)

        self.loss_cc_su_s2t = slim.losses.sparse_softmax_cross_entropy(self.logits_s2t, self.label_src)
        self.loss_cc_su_t2s = slim.losses.sparse_softmax_cross_entropy(self.logits_t2s, self.label_trg)
        self.loss_cc_s2t_su = slim.losses.sparse_softmax_cross_entropy(self.logits_src, self.pred_s2t)
        self.loss_cc_t2s_su = slim.losses.sparse_softmax_cross_entropy(self.logits_trg, self.pred_t2s)
        self.loss_cc_un_s2t = tf.reduce_mean(tf.square(self.hot_s2t-self.hot_src))
        self.loss_cc_un_t2s = tf.reduce_mean(tf.square(self.hot_t2s-self.hot_trg))

        # self.fsrc_mid_hsrc = Decoder_hot(self.hot_src, self.image_size, is_training=is_training,
        #                                  name='Decoder_hot')
        # self.ftrg_mid_htrg = Decoder_hot(self.hot_trg, self.image_size, is_training=is_training,
        #                                  reuse=True, name='Decoder_hot')
        # self.fsrc_mid_csrc = Decoder_calm(self.calm_src, self.image_size, is_training=is_training,
        #                                   name='Decoder_calm_src')
        # self.ftrg_mid_ctrg = Decoder_calm(self.calm_trg, self.image_size, is_training=is_training,
        #                                   name='Decoder_calm_trg')
        # self.fsrc_from_src = Decoder_image(self.fsrc_mid_hsrc, self.fsrc_mid_csrc,
        #                                    is_training=is_training, name='Decoder_image_src')
        # self.ftrg_from_trg = Decoder_image(self.ftrg_mid_htrg, self.ftrg_mid_ctrg,
        #                                    is_training=is_training, name='Decoder_image_trg')
        #
        # self.gsrc_mid_htrg = Decoder_hot(self.hot_trg, self.image_size, is_training=is_training,
        #                                  reuse=True, name='Decoder_hot')
        # self.gtrg_mid_hsrc = Decoder_hot(self.hot_src, self.image_size, is_training=is_training,
        #                                  reuse=True, name='Decoder_hot')
        # self.gsrc_mid_cprior = Decoder_calm(self.prior_calm_src, self.image_size, reuse=True,
        #                                     is_training=is_training, name='Decoder_calm_src')
        # self.gtrg_mid_cprior = Decoder_calm(self.prior_calm_trg, self.image_size, reuse=True,
        #                                     is_training=is_training, name='Decoder_calm_trg')
        # self.gsrc_from_trg = Decoder_image(self.gsrc_mid_htrg, self.gsrc_mid_cprior, reuse=True,
        #                                    is_training=is_training, name='Decoder_image_src')
        # self.gtrg_from_src = Decoder_image(self.gtrg_mid_hsrc, self.gtrg_mid_cprior, reuse=True,
        #                                    is_training=is_training, name='Decoder_image_trg')
        #
        # self.gsrc_mid_hprior = Decoder_hot(self.prior_hot, self.image_size, reuse=True,
        #                                    is_training=is_training, name='Decoder_hot')
        # self.gtrg_mid_hprior = self.gsrc_mid_hprior
        # self.gsrc_from_prior = Decoder_image(self.gsrc_mid_hprior, self.gsrc_mid_cprior, reuse=True,
        #                                      is_training=is_training, name='Decoder_image_src')
        # self.gtrg_from_prior = Decoder_image(self.gtrg_mid_hprior, self.gtrg_mid_cprior, reuse=True,
        #                                      is_training=is_training, name='Decoder_image_trg')

        self.logits_isHPrior_src = Discriminator_lc_fn(self.hot_src, self.hot_size,
                                                       name='Discriminator_isHPrior')
        self.logits_isHPrior_trg = Discriminator_lc_fn(self.hot_trg, self.hot_size, reuse=True,
                                                       name='Discriminator_isHPrior')
        self.logits_isCPrior_src = Discriminator_lc_fn(self.calm_src, self.calm_size_src,
                                                       name='Discriminator_isCPrior_src')
        self.logits_isCPrior_trg = Discriminator_lc_fn(self.calm_trg, self.calm_size_trg,
                                                       name='Discriminator_isCPrior_trg')
        self.logits_isHPrior_prior = Discriminator_lc_fn(self.prior_hot, self.hot_size, reuse=True,
                                                         name='Discriminator_isHPrior')
        self.logits_isCPrior_sprior = \
            Discriminator_lc_fn(self.prior_calm_src, self.calm_size_src, reuse=True,
                                name='Discriminator_isCPrior_src')
        self.logits_isCPrior_tprior = \
            Discriminator_lc_fn(self.prior_calm_trg, self.calm_size_trg, reuse=True,
                                name='Discriminator_isCPrior_trg')

        self.loss_hot_su_src = \
            slim.losses.sparse_softmax_cross_entropy(self.logits_src, self.label_src)
        self.loss_hot_su_trg = \
            slim.losses.sparse_softmax_cross_entropy(self.logits_trg, self.label_trg)
        self.loss_hot_su = self.loss_hot_su_src+self.loss_hot_su_trg

        self.loss_const_src = tf.reduce_mean(tf.square(self.fsrc_from_src-self.image_src))
        self.loss_const_trg = tf.reduce_mean(tf.square(self.ftrg_from_trg-self.image_trg))

        self.loss_isHPD_src = slim.losses.\
            sigmoid_cross_entropy(self.logits_isHPrior_src, tf.zeros_like(self.logits_isHPrior_src))
        self.loss_isHPD_trg = slim.losses.\
            sigmoid_cross_entropy(self.logits_isHPrior_trg, tf.zeros_like(self.logits_isHPrior_trg))
        self.loss_isHPD_prior = slim.losses.\
            sigmoid_cross_entropy(self.logits_isHPrior_prior, tf.ones_like(self.logits_isHPrior_prior))
        self.loss_isHPD = beta1*self.loss_isHPD_src+beta2*self.loss_isHPD_trg+beta3*self.loss_isHPD_prior

        self.loss_isHPen_src = slim.losses.\
            sigmoid_cross_entropy(self.logits_isHPrior_src, tf.ones_like(self.logits_isHPrior_src))
        self.loss_isHPen_trg = slim.losses.\
            sigmoid_cross_entropy(self.logits_isHPrior_trg, tf.ones_like(self.logits_isHPrior_trg))
        self.loss_isHPen = beta1*self.loss_isHPen_src+beta2*self.loss_isHPen_trg

        self.loss_isCPD_src = slim.losses.\
            sigmoid_cross_entropy(self.logits_isCPrior_src, tf.zeros_like(self.logits_isCPrior_src))
        self.loss_isCPD_trg = slim.losses.\
            sigmoid_cross_entropy(self.logits_isCPrior_trg, tf.zeros_like(self.logits_isCPrior_trg))
        self.loss_isCPD_sprior = slim.losses.\
            sigmoid_cross_entropy(self.logits_isCPrior_sprior, tf.ones_like(self.logits_isCPrior_sprior))
        self.loss_isCPD_tprior = slim.losses.\
            sigmoid_cross_entropy(self.logits_isCPrior_tprior, tf.ones_like(self.logits_isCPrior_tprior))
        self.loss_isCPD = alpha1*self.loss_isCPD_src\
                          +alpha2*self.loss_isCPD_trg\
                          +alpha3*self.loss_isCPD_sprior\
                          +alpha4*self.loss_isCPD_tprior

        self.loss_isCPen_src = slim.losses.\
            sigmoid_cross_entropy(self.logits_isCPrior_src, tf.ones_like(self.logits_isCPrior_src))
        self.loss_isCPen_trg = slim.losses.\
            sigmoid_cross_entropy(self.logits_isCPrior_trg, tf.ones_like(self.logits_isCPrior_trg))
        self.loss_isCPen = alpha1*self.loss_isCPen_src+alpha2*self.loss_isCPen_trg

        self.loss_auto = gamma1 * self.loss_const_src + gamma2 * self.loss_const_trg
        self.loss_hot_su = lambda1 * self.loss_hot_su_src + lambda2 * self.loss_hot_su_trg

        self.loss_Hen = self.loss_isHPen
        self.loss_Cen = self.loss_isCPen
        self.loss_D = self.loss_isHPD + self.loss_isCPD
        self.loss_cc_su = eta1 * self.loss_cc_su_s2t + eta2 * self.loss_cc_su_t2s
        self.loss_cc_un = eta1 * self.loss_cc_un_s2t + eta2 * self.loss_cc_un_t2s
        self.loss_cc_suun = eta1 * self.loss_cc_su_s2t + eta2 * self.loss_cc_t2s_su
        self.loss_cc_unsu = eta1 * self.loss_cc_s2t_su + eta2 * self.loss_cc_su_t2s

        if is_training:
            self.optimizer_su = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_auto = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_D = tf.train.AdamOptimizer(self.learning_rate)
            # self.optimizer_en = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_Hen = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_Cen = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_auto_c = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_cc_su = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_cc_un = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_cc_suun = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_cc_unsu = tf.train.AdamOptimizer(self.learning_rate)
            # self.optimizer_su_src = tf.train.AdamOptimizer(self.learning_rate)
            # self.optimizer_su_trg = tf.train.AdamOptimizer(self.learning_rate)
            # self.optimizer_Hen_src = tf.train.AdamOptimizer(self.learning_rate)
            # self.optimizer_Hen_trg = tf.train.AdamOptimizer(self.learning_rate)

            vars_total = tf.trainable_variables()
            vars_en = [var for var in vars_total if 'Encoder' in var.name]
            vars_de = [var for var in vars_total if 'Decoder' in var.name]
            vars_cen = [var for var in vars_total if 'Encoder_clam' in var.name]
            vars_auto = []
            vars_auto.extend(vars_en)
            vars_auto.extend(vars_de)
            vars_auto_c = []
            vars_auto_c.extend(vars_cen)
            vars_auto_c.extend(vars_de)
            vars_D = [var for var in vars_total if 'Discriminator' in var.name]

            self.train_op_su = slim.learning.create_train_op(self.loss_hot_su, self.optimizer_su,
                                                             variables_to_train=vars_en)
            # self.train_op_su_src = slim.learning.create_train_op(self.loss_hot_su_src,
            #                                                      self.optimizer_su_src,
            #                                                      variables_to_train=vars_en)
            # self.train_op_su_trg = slim.learning.create_train_op(self.loss_hot_su_trg,
            #                                                      self.optimizer_su_trg,
            #                                                      variables_to_train=vars_en)
            self.train_op_auto = slim.learning.create_train_op(self.loss_auto, self.optimizer_auto,
                                                               variables_to_train=vars_auto)
            self.train_op_auto_c = slim.learning.create_train_op(self.loss_auto, self.optimizer_auto_c,
                                                                 variables_to_train=vars_auto_c)
            self.train_op_D = slim.learning.create_train_op(self.loss_D, self.optimizer_D,
                                                            variables_to_train=vars_D)
            # self.train_op_en = slim.learning.create_train_op(self.loss_en, self.optimizer_en,
            #                                                  variables_to_train=vars_en)
            self.train_op_Hen = slim.learning.create_train_op(self.loss_Hen, self.optimizer_Hen,
                                                              variables_to_train=vars_en)
            # self.train_op_Hen_src = slim.learning.create_train_op(self.loss_isHPen_src,
            #                                                       self.optimizer_Hen_src,
            #                                                       variables_to_train=vars_en)
            # self.train_op_Hen_trg = slim.learning.create_train_op(self.loss_isHPen_trg,
            #                                                       self.optimizer_Hen_trg,
            #                                                       variables_to_train=vars_en)
            self.train_op_Cen = slim.learning.create_train_op(self.loss_Cen, self.optimizer_Cen,
                                                              variables_to_train=vars_en)
            self.train_op_cc_su = slim.learning.create_train_op(self.loss_cc_su, self.optimizer_cc_su,
                                                                variables_to_train=vars_auto)
            self.train_op_cc_un = slim.learning.create_train_op(self.loss_cc_un, self.optimizer_cc_un,
                                                                variables_to_train=vars_auto)
            self.train_op_cc_suun = slim.learning.create_train_op(self.loss_cc_suun, self.optimizer_cc_suun,
                                                                  variables_to_train=vars_auto)
            self.train_op_cc_unsu = slim.learning.create_train_op(self.loss_cc_unsu, self.optimizer_cc_unsu,
                                                                  variables_to_train=vars_auto)

        self.summary_op_su = tf.summary.merge([
            tf.summary.scalar('loss_hot_su_src', self.loss_hot_su_src),
            tf.summary.scalar('loss_hot_su_trg', self.loss_hot_su_trg),
            tf.summary.scalar('loss_hot_su', self.loss_hot_su),
            tf.summary.scalar('loss_cc_su_s2t', self.loss_cc_su_s2t),
            tf.summary.scalar('loss_cc_su_t2s', self.loss_cc_su_t2s),
            tf.summary.scalar('loss_cc_s2t_su', self.loss_cc_s2t_su),
            tf.summary.scalar('loss_cc_t2s_su', self.loss_cc_t2s_su)
        ])
        self.summary_op = tf.summary.merge([
            tf.summary.scalar('acc_src', self.acc_src),
            tf.summary.scalar('acc_trg', self.acc_trg),
            tf.summary.scalar('loss_const_src', self.loss_const_src),
            tf.summary.scalar('loss_const_trg', self.loss_const_trg),
            tf.summary.scalar('loss_auto', self.loss_auto),
            tf.summary.scalar('loss_isHPD_src', self.loss_isHPD_src),
            tf.summary.scalar('loss_isHPD_trg', self.loss_isHPD_trg),
            tf.summary.scalar('loss_isHPD_prior', self.loss_isHPD_prior),
            tf.summary.scalar('loss_isCPD_src', self.loss_isCPD_src),
            tf.summary.scalar('loss_isCPD_trg', self.loss_isCPD_trg),
            tf.summary.scalar('loss_isCPD_sprior', self.loss_isCPD_sprior),
            tf.summary.scalar('loss_isCPD_tprior', self.loss_isCPD_tprior),
            tf.summary.scalar('loss_isCPD', self.loss_isCPD),
            tf.summary.scalar('loss_isHPen_src', self.loss_isHPen_src),
            tf.summary.scalar('loss_isHPen_trg', self.loss_isHPen_trg),
            tf.summary.scalar('loss_isHPen', self.loss_isHPen),
            tf.summary.scalar('loss_isCPen_src', self.loss_isCPen_src),
            tf.summary.scalar('loss_isCPen_trg', self.loss_isCPen_trg),
            tf.summary.scalar('loss_isCPen', self.loss_isCPen),
            tf.summary.scalar('loss_cc_un_s2t', self.loss_cc_un_s2t),
            tf.summary.scalar('loss_cc_un_t2s', self.loss_cc_un_t2s)
            # tf.summary.image('image_src', self.image_src),
            # tf.summary.image('image_trg', self.image_trg),
            # tf.summary.image('fsrc_from_src', self.fsrc_from_src),
            # tf.summary.image('ftrg_from_trg', self.ftrg_from_trg),
            # tf.summary.image('gsrc_from_src', self.gsrc_from_src),
            # tf.summary.image('gtrg_from_trg', self.gtrg_from_trg),
            # tf.summary.image('gsrc_from_trg', self.gsrc_from_trg),
            # tf.summary.image('gtrg_from_src', self.gtrg_from_src),
            # tf.summary.image('gsrc_from_prior', self.gsrc_from_prior),
            # tf.summary.image('gtrg_from_prior', self.gtrg_from_prior)
        ])







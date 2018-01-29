import tensorflow as tf
import numpy as np
from ops import conv2d, deconv2d_1
from input_data import read_data
from utils import random_rotate_image, flip, check_rle
import cv2

class pixel2pixel(object):
    def __init__(self, sess, batch_size=64, sample_size=64, gf_dim=64, df_dim=64,
                 input_c_dim=3, output_c_dim=1, L1_lambda=100, lr=0.0005, epoches=400, is_train=True):
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.is_train = is_train
        self.sess = sess
        self.l1_lambda = L1_lambda
        self.lr = lr
        self.epoches=epoches

    def unet(self, image, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            self.unet_c_1 = conv2d(input=image, output_channles=self.gf_dim, is_train=self.is_train,
                                 k_h=4, k_w=4, stddev=0.02, name='unet1/conv2d')
            self.unet_c_2 = conv2d(input=self.unet_c_1, output_channles=128, is_train=self.is_train,
                                 k_h=4, k_w=4, stddev=0.02, name='unet2/conv2d')
            self.unet_c_3 = conv2d(input=self.unet_c_2, output_channles=256, is_train=self.is_train,
                                 k_h=4, k_w=4, stddev=0.02, name='unet3/conv2d')
            self.unet_c_4 = conv2d(input=self.unet_c_3, output_channles=512, is_train=self.is_train,
                                 k_h=4, k_w=4, stddev=0.02, name='unet4/conv2d')
            self.unet_c_5 = conv2d(input=self.unet_c_4, output_channles=512, is_train=self.is_train,
                                 k_h=4, k_w=4, stddev=0.02, name='unet5/conv2d')
            self.unet_c_6 = conv2d(input=self.unet_c_5, output_channles=512, is_train=self.is_train,
                                 k_h=4, k_w=4, stddev=0.02, name='unet6/conv2d')
            self.unet_c_7 = conv2d(input=self.unet_c_6, output_channles=512, is_train=self.is_train,
                                 k_h=4, k_w=4, stddev=0.02, name='unet7/conv2d')
            self.unet_c_8 = conv2d(input=self.unet_c_7, output_channles=512, is_train=self.is_train,
                                 k_h=4, k_w=4, stddev=0.02, name='unet8/conv2d')

            #mask
            list_unet_c_7 = self.unet_c_7.get_shape().as_list()[1:]
            self.unet_cd_1 = deconv2d_1(input=self.unet_c_8, width=list_unet_c_7[0], height=list_unet_c_7[1],
                                        input_channels=512, output_channels=512, is_train=self.is_train,
                                      k=4, s=2, batch_size=self.batch_size, name="unet_cd_1/deconv2d", stddev=0.02, activation_fn=None)

            list_unet_c_6 = self.unet_c_6.get_shape().as_list()[1:]
            self.unet_cd_1_1 = tf.concat([self.unet_cd_1, self.unet_c_7], axis=3)
            self.unet_cd_2 = deconv2d_1(input=self.unet_cd_1_1, width=list_unet_c_6[0], height=list_unet_c_6[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                      k=4, s=2, batch_size=self.batch_size, name="unet_cd_2/deconv2d", stddev=0.02, activation_fn=None)

            list_unet_c_5 = self.unet_c_5.get_shape().as_list()[1:]
            self.unet_cd_2_1 = tf.concat([self.unet_cd_2, self.unet_c_6], axis=3)
            self.unet_cd_3 = deconv2d_1(input=self.unet_cd_2_1, width=list_unet_c_5[0], height=list_unet_c_5[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                      k=4, s=2, batch_size=self.batch_size, name="unet_cd_3/deconv2d", stddev=0.02, activation_fn=None)

            list_unet_c_4 = self.unet_c_4.get_shape().as_list()[1:]
            self.unet_cd_3_1 = tf.concat([self.unet_cd_3, self.unet_c_5], axis=3)
            self.unet_cd_4 = deconv2d_1(input=self.unet_cd_3_1, width=list_unet_c_4[0], height=list_unet_c_4[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                      k=4, s=2, batch_size=self.batch_size, name="unet_cd_4/deconv2d", stddev=0.02, activation_fn=None)

            list_unet_c_3 = self.unet_c_3.get_shape().as_list()[1:]
            self.unet_cd_4_1 = tf.concat([self.unet_cd_4, self.unet_c_4], axis=3)
            self.unet_cd_5 = deconv2d_1(input=self.unet_cd_4_1, width=list_unet_c_3[0], height=list_unet_c_3[1],
                                        input_channels=1024, output_channels=256, is_train=self.is_train,
                                      k=4, s=2, batch_size=self.batch_size, name="unet_cd_5/deconv2d", stddev=0.02, activation_fn=None)

            list_unet_c_2 = self.unet_c_2.get_shape().as_list()[1:]
            self.unet_cd_5_1 = tf.concat([self.unet_cd_5, self.unet_c_3], axis=3)
            self.unet_cd_6 = deconv2d_1(input=self.unet_cd_5_1, width=list_unet_c_2[0], height=list_unet_c_2[1],
                                        input_channels=512, output_channels=128, is_train=self.is_train,
                                      k=4, s=2, batch_size=self.batch_size, name="unet_cd_6/deconv2d", stddev=0.02, activation_fn=None)

            list_unet_c_1 = self.unet_c_1.get_shape().as_list()[1:]
            self.unet_cd_7_1 = tf.concat([self.unet_cd_6, self.unet_c_2], axis=3)
            self.unet_cd_7 = deconv2d_1(input=self.unet_cd_7_1, width=list_unet_c_1[0], height=list_unet_c_1[1],
                                        input_channels=256, output_channels=64, is_train=self.is_train,
                                      k=4, s=2, batch_size=self.batch_size, name="unet_cd_7/deconv2d", stddev=0.02, activation_fn=None)

            list_input = image.get_shape().as_list()[1:]
            self.unet_cd_8_1 = tf.concat([self.unet_cd_7, self.unet_c_1], axis=3)
            self.unet_out = deconv2d_1(input=self.unet_cd_8_1, width=list_input[0], height=list_input[1],
                                       input_channels=128, output_channels=1, is_train=self.is_train,
                                       k=4, s=2, batch_size=self.batch_size, name="unet_out/deconv2d", stddev=0.02, activation_fn=tf.nn.tanh)

            #boundary
            list_unet_cb_7 = self.unet_c_7.get_shape().as_list()[1:]
            self.unet_cdb_1 = deconv2d_1(input=self.unet_c_8, width=list_unet_cb_7[0], height=list_unet_cb_7[1],
                                        input_channels=512, output_channels=512, is_train=self.is_train,
                                        k=4, s=2, batch_size=self.batch_size, name="unet_cdb_1/deconv2d", stddev=0.02,
                                        activation_fn=None)

            list_unet_cb_6 = self.unet_c_6.get_shape().as_list()[1:]
            self.unet_cdb_1_1 = tf.concat([self.unet_cdb_1, self.unet_c_7], axis=3)
            self.unet_cdb_2 = deconv2d_1(input=self.unet_cdb_1_1, width=list_unet_cb_6[0], height=list_unet_cb_6[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                        k=4, s=2, batch_size=self.batch_size, name="unet_cd_2/deconv2d", stddev=0.02,
                                        activation_fn=None)

            list_unet_cb_5 = self.unet_c_5.get_shape().as_list()[1:]
            self.unet_cdb_2_1 = tf.concat([self.unet_cdb_2, self.unet_c_6], axis=3)
            self.unet_cdb_3 = deconv2d_1(input=self.unet_cdb_2_1, width=list_unet_cb_5[0], height=list_unet_cb_5[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                        k=4, s=2, batch_size=self.batch_size, name="unet_cd_3/deconv2d", stddev=0.02,
                                        activation_fn=None)

            list_unet_cb_4 = self.unet_c_4.get_shape().as_list()[1:]
            self.unet_cdb_3_1 = tf.concat([self.unet_cdb_3, self.unet_c_5], axis=3)
            self.unet_cdb_4 = deconv2d_1(input=self.unet_cdb_3_1, width=list_unet_cb_4[0], height=list_unet_cb_4[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                        k=4, s=2, batch_size=self.batch_size, name="unet_cd_4/deconv2d", stddev=0.02,
                                        activation_fn=None)

            list_unet_cb_3 = self.unet_c_3.get_shape().as_list()[1:]
            self.unet_cdb_4_1 = tf.concat([self.unet_cdb_4, self.unet_c_4], axis=3)
            self.unet_cdb_5 = deconv2d_1(input=self.unet_cdb_4_1, width=list_unet_cb_3[0], height=list_unet_cb_3[1],
                                        input_channels=1024, output_channels=256, is_train=self.is_train,
                                        k=4, s=2, batch_size=self.batch_size, name="unet_cd_5/deconv2d", stddev=0.02,
                                        activation_fn=None)

            list_unet_cb_2 = self.unet_c_2.get_shape().as_list()[1:]
            self.unet_cdb_5_1 = tf.concat([self.unet_cdb_5, self.unet_c_3], axis=3)
            self.unet_cdb_6 = deconv2d_1(input=self.unet_cdb_5_1, width=list_unet_cb_2[0], height=list_unet_cb_2[1],
                                        input_channels=512, output_channels=128, is_train=self.is_train,
                                        k=4, s=2, batch_size=self.batch_size, name="unet_cd_6/deconv2d", stddev=0.02,
                                        activation_fn=None)

            list_unet_cb_1 = self.unet_c_1.get_shape().as_list()[1:]
            self.unet_cdb_7_1 = tf.concat([self.unet_cdb_6, self.unet_c_2], axis=3)
            self.unet_cdb_7 = deconv2d_1(input=self.unet_cdb_7_1, width=list_unet_cb_1[0], height=list_unet_cb_1[1],
                                        input_channels=256, output_channels=64, is_train=self.is_train,
                                        k=4, s=2, batch_size=self.batch_size, name="unet_cd_7/deconv2d", stddev=0.02,
                                        activation_fn=None)

            list_input = image.get_shape().as_list()[1:]
            self.unet_cdb_8_1 = tf.concat([self.unet_cdb_7, self.unet_c_1], axis=3)
            self.unet_b_out = deconv2d_1(input=self.unet_cdb_8_1, width=list_input[0], height=list_input[1],
                                       input_channels=128, output_channels=1, is_train=self.is_train,
                                       k=4, s=2, batch_size=self.batch_size, name="unet_out/deconv2d", stddev=0.02,
                                       activation_fn=tf.nn.tanh)

            return self.unet_out, self.unet_b_out

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            self.disc_c_1 = conv2d(input=image, output_channles=64, is_train=self.is_train,
                                   k_h=4, k_w=4, stddev=0.02, name='disc1/conv2d')
            self.disc_c_2 = conv2d(input=self.disc_c_1, output_channles=128, is_train=self.is_train,
                                   k_h=4, k_w=4, stddev=0.02, name='disc2/conv2d')
            self.disc_c_3 = conv2d(input=self.disc_c_2, output_channles=256, is_train=self.is_train,
                                   k_h=4, k_w=4, stddev=0.02, name='disc3/conv2d')
            self.disc_c_4 = conv2d(input=self.disc_c_3, output_channles=512, is_train=self.is_train,
                                   k_h=4, k_w=4, stddev=0.02, name='disc4/conv2d')
            self.disc_c_5 = conv2d(input=self.disc_c_4, output_channles=512, is_train=self.is_train,
                                   k_h=4, k_w=4, stddev=0.02, name='disc5/conv2d')
            self.disc_c_6 = conv2d(input=self.disc_c_5, output_channles=512, is_train=self.is_train,
                                   k_h=4, k_w=4, stddev=0.02, name='disc6/conv2d')
            self.disc_c_7 = conv2d(input=self.disc_c_6, output_channles=512, is_train=self.is_train,
                                   k_h=4, k_w=4, stddev=0.02, name='disc7/conv2d')

            self.avg_pool = tf.reduce_mean(self.disc_c_7, [1, 2])
            self.fc = tf.reshape(self.avg_pool, [self.batch_size, 512])
            self.disc_out = tf.layers.dense(self.fc, 1, activation=tf.nn.sigmoid)

            return self.disc_out

    def load_samples(self):
        imgs, masks_all, boundaries_all = read_data()
        return np.array(imgs), np.array(masks_all), np.array(boundaries_all)

    def mean_iou(self, y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            prec.append(score)
        return tf.reduce_mean(prec, axis=0)

    def train(self):
        self.img = tf.placeholder(tf.float32, shape=[self.batch_size, 256, 256, 3], name='image')  # B
        self.mask_boundary = tf.placeholder(tf.float32, shape=[self.batch_size, 256, 256, 1], name='mask_boundary')  # real A
        self.boundary = tf.placeholder(tf.float32, shape=[self.batch_size, 256, 256, 1], name='boundary')


        self.fake, self.fake_boundary = self.unet(self.img)  # fake A
        self.fake_sum = tf.summary.image("fake_B", self.fake)

        self.fake_AB = tf.concat([self.fake, self.img, self.fake_boundary], axis=3)
        self.real_AB = tf.concat([self.mask_boundary, self.img, self.boundary], axis=3)

        self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D_logits)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_logits)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_logits))) \
                      + self.l1_lambda * tf.reduce_mean(tf.abs(self.mask_boundary - self.fake)) +\
                      self.l1_lambda * tf.reduce_mean(tf.abs(self.boundary - self.fake_boundary))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.iou = self.mean_iou(self.mask_boundary, self.fake)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.iou_summary = tf.summary.scalar('iou', self.iou)

        self.d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        self.g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

        d_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        self.saver = tf.train.Saver()

        imgs, masks_all, boundaries_all = self.load_samples()

        for step in range(self.epoches):
            offset = step % len(imgs)
            image = imgs[offset]
            mask = masks_all[offset]
            boundary = boundaries_all[offset]
            mask_boundary = mask - boundary

            images = []
            mask_boundaries = []
            for img, ma in zip(image, mask_boundary):
                img, ma = flip(img, ma)
                img, ma = random_rotate_image(img, ma)
                images.append(img)
                mask_boundaries.append(ma)

            _, D_loss = self.sess.run([d_optim, self.d_loss],
                                             feed_dict={self.img: image, self.mask_boundary: mask_boundary})

            _, summary, G_loss, fake_mask_boundary = self.sess.run([g_optim, self.summary_op, self.g_loss, self.fake],
                                             feed_dict={self.img: image, self.mask_boundary: mask_boundary})

            check_rle(mask_boundary, fake_mask_boundary)
            self.writer.add_summary(summary, step)
            print('step is %d' % step)


            if np.mod(step, 400) == 0:
                self.saver.save(self.sess, './checkpoint', step)

            if np.mod(step, 100) == 0:
                fake_show = fake_mask_boundary[0]
                fake_show = (fake_show + 1.0) * 127.5
                fake_show = np.array(fake_show)
                fake_show = (fake_show > 127.0).astype(np.uint8)

                mask_show = mask_boundaries[0]
                mask_show = (mask_show + 1.0) * 127.5
                mask_show = np.array(mask_show)
                mask_show = (mask_show > 127.0).astype(np.uint8)

                cv2.imshow('fake', fake_show)
                cv2.imshow('mask', mask_show)
                cv2.waitKey(25)




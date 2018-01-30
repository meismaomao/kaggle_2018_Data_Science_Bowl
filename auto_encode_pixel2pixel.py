import tensorflow as tf
import numpy as np
from ops import conv2d, deconv2d_1
from input_data import load_batch
import cv2

class pixel2pixel(object):
    def __init__(self, sess, batch_size=64, sample_size=32, gf_dim=64, df_dim=64,
                 input_c_dim=3, output_c_dim=1, L1_lambda=500, rec_rate=1000, lr=0.0005, epoches=4000, is_train=True): #1.rec_rate:1000
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
        self.rec_rate = rec_rate

    def unet(self, image, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            reg_mask_list = []
            reg_boundary_list = []

            self.unet_c_1 = conv2d(input=image, output_channles=self.gf_dim, is_train=self.is_train,
                                 k_h=3, k_w=3, stddev=0.02, name='unet1/conv2d')
            self.unet_c_2 = conv2d(input=self.unet_c_1, output_channles=128, is_train=self.is_train,
                                 k_h=3, k_w=3, stddev=0.02, name='unet2/conv2d')
            self.unet_c_3 = conv2d(input=self.unet_c_2, output_channles=256, is_train=self.is_train,
                                 k_h=3, k_w=3, stddev=0.02, name='unet3/conv2d')
            self.unet_c_4 = conv2d(input=self.unet_c_3, output_channles=512, is_train=self.is_train,
                                 k_h=3, k_w=3, stddev=0.02, name='unet4/conv2d')
            self.unet_c_5 = conv2d(input=self.unet_c_4, output_channles=512, is_train=self.is_train,
                                 k_h=3, k_w=3, stddev=0.02, name='unet5/conv2d')
            self.unet_c_6 = conv2d(input=self.unet_c_5, output_channles=512, is_train=self.is_train,
                                 k_h=3, k_w=3, stddev=0.02, name='unet6/conv2d')
            self.unet_c_7 = conv2d(input=self.unet_c_6, output_channles=512, is_train=self.is_train,
                                 k_h=3, k_w=3, stddev=0.02, name='unet7/conv2d')
            self.unet_c_8 = conv2d(input=self.unet_c_7, output_channles=512, is_train=self.is_train,
                                 k_h=3, k_w=3, stddev=0.02, name='unet8/conv2d')

            #mask
            list_unet_c_7 = self.unet_c_7.get_shape().as_list()[1:]
            unet_cg_1 = deconv2d_1(input=self.unet_c_8, width=list_unet_c_7[0], height=list_unet_c_7[1],
                                        input_channels=512, output_channels=512, is_train=self.is_train,
                                      k=4, s=2, batch_size=self.batch_size, name="unet_cg_1/deconv2d", stddev=0.02,
                                        activation_fn=None)

            # self.reg_unet_cg_1 = tf.nn.tanh(unet_cg_1[:, :, :, :1])
            # reg_mask_list.append(self.reg_unet_cg_1)
            self.unet_cg_1 = tf.nn.relu(unet_cg_1)

            list_unet_c_6 = self.unet_c_6.get_shape().as_list()[1:]
            self.unet_cg_1_1 = tf.concat([self.unet_cg_1, self.unet_c_7], axis=3)
            unet_cg_2 = deconv2d_1(input=self.unet_cg_1_1, width=list_unet_c_6[0], height=list_unet_c_6[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                      k=3, s=2, batch_size=self.batch_size, name="unet_cg_2/deconv2d", stddev=0.02,
                                        activation_fn=None)

            # self.reg_unet_cg_2 = tf.nn.tanh(unet_cg_2[:, :, :, :1])
            # reg_mask_list.append(self.reg_unet_cg_2)
            self.unet_cg_2 = tf.nn.relu(unet_cg_2)

            list_unet_c_5 = self.unet_c_5.get_shape().as_list()[1:]
            self.unet_cg_2_1 = tf.concat([self.unet_cg_2, self.unet_c_6], axis=3)
            unet_cg_3 = deconv2d_1(input=self.unet_cg_2_1, width=list_unet_c_5[0], height=list_unet_c_5[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                      k=3, s=2, batch_size=self.batch_size, name="unet_cg_3/deconv2d", stddev=0.02,
                                        activation_fn=None)

            self.reg_unet_cg_3 = tf.nn.tanh(unet_cg_3[:, :, :, :1])
            reg_mask_list.append(self.reg_unet_cg_3)
            self.unet_cg_3 = tf.nn.relu(unet_cg_3)

            list_unet_c_4 = self.unet_c_4.get_shape().as_list()[1:]
            self.unet_cg_3_1 = tf.concat([self.unet_cg_3, self.unet_c_5], axis=3)
            unet_cg_4 = deconv2d_1(input=self.unet_cg_3_1, width=list_unet_c_4[0], height=list_unet_c_4[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                      k=3, s=2, batch_size=self.batch_size, name="unet_cg_4/deconv2d", stddev=0.02,
                                        activation_fn=None)

            self.reg_unet_cg_4 = tf.nn.tanh(unet_cg_4[:, :, :, :1])
            reg_mask_list.append(self.reg_unet_cg_4)
            self.unet_cg_4 = tf.nn.relu(unet_cg_4)

            list_unet_c_3 = self.unet_c_3.get_shape().as_list()[1:]
            self.unet_cg_4_1 = tf.concat([self.unet_cg_4, self.unet_c_4], axis=3)
            unet_cg_5 = deconv2d_1(input=self.unet_cg_4_1, width=list_unet_c_3[0], height=list_unet_c_3[1],
                                        input_channels=1024, output_channels=256, is_train=self.is_train,
                                      k=3, s=2, batch_size=self.batch_size, name="unet_cg_5/deconv2d", stddev=0.02,
                                        activation_fn=None)

            self.reg_unet_cg_5 = tf.nn.tanh(unet_cg_5[:, :, :, :1])
            reg_mask_list.append(self.reg_unet_cg_5)
            self.unet_cg_5 = tf.nn.relu(unet_cg_5)

            list_unet_c_2 = self.unet_c_2.get_shape().as_list()[1:]
            self.unet_cg_5_1 = tf.concat([self.unet_cg_5, self.unet_c_3], axis=3)
            unet_cg_6 = deconv2d_1(input=self.unet_cg_5_1, width=list_unet_c_2[0], height=list_unet_c_2[1],
                                        input_channels=512, output_channels=128, is_train=self.is_train,
                                      k=3, s=2, batch_size=self.batch_size, name="unet_cg_6/deconv2d", stddev=0.02,
                                        activation_fn=None)

            self.reg_unet_cg_6 = tf.nn.tanh(unet_cg_6[:, :, :, :1])
            reg_mask_list.append(self.reg_unet_cg_6)
            self.unet_cg_6 = tf.nn.relu(unet_cg_6)

            list_unet_c_1 = self.unet_c_1.get_shape().as_list()[1:]
            self.unet_cg_7_1 = tf.concat([self.unet_cg_6, self.unet_c_2], axis=3)
            unet_cg_7 = deconv2d_1(input=self.unet_cg_7_1, width=list_unet_c_1[0], height=list_unet_c_1[1],
                                        input_channels=256, output_channels=64, is_train=self.is_train,
                                      k=3, s=2, batch_size=self.batch_size, name="unet_cg_7/deconv2d", stddev=0.02,
                                        activation_fn=None)

            self.reg_unet_cg_7 = tf.nn.tanh(unet_cg_7[:, :, :, :1])
            reg_mask_list.append(self.reg_unet_cg_7)
            self.unet_cg_7 = tf.nn.relu(unet_cg_7)

            list_input = image.get_shape().as_list()[1:]
            self.unet_cd_8_1 = tf.concat([self.unet_cg_7, self.unet_c_1], axis=3)
            self.unet_out = deconv2d_1(input=self.unet_cd_8_1, width=list_input[0], height=list_input[1],
                                       input_channels=128, output_channels=1, is_train=self.is_train,
                                       k=3, s=2, batch_size=self.batch_size, name="unet_out/deconv2d", stddev=0.02,
                                       activation_fn=tf.nn.tanh)


            #boundary
            list_unet_cb_7 = self.unet_c_7.get_shape().as_list()[1:]
            unet_cdb_1 = deconv2d_1(input=self.unet_c_8, width=list_unet_cb_7[0], height=list_unet_cb_7[1],
                                        input_channels=512, output_channels=512, is_train=self.is_train,
                                        k=3, s=2, batch_size=self.batch_size, name="unet_cdb_1/deconv2d", stddev=0.02,
                                        activation_fn=None)

            # self.reg_unet_cbd_1 = tf.nn.tanh(unet_cdb_1[:, :, :, :1])
            # reg_boundary_list.append(self.reg_unet_cbd_1)
            self.unet_cdb_1 = tf.nn.relu(unet_cdb_1)

            list_unet_cb_6 = self.unet_c_6.get_shape().as_list()[1:]
            self.unet_cdb_1_1 = tf.concat([self.unet_cdb_1, self.unet_c_7], axis=3)
            unet_cdb_2 = deconv2d_1(input=self.unet_cdb_1_1, width=list_unet_cb_6[0], height=list_unet_cb_6[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                        k=3, s=2, batch_size=self.batch_size, name="unet_cdb_2/deconv2d", stddev=0.02,
                                        activation_fn=None)

            # self.reg_unet_cbd_2 = tf.nn.tanh(unet_cdb_2[:, :, :, :1])
            # reg_boundary_list.append(self.reg_unet_cbd_2)
            self.unet_cdb_2 = tf.nn.relu(unet_cdb_2)

            list_unet_cb_5 = self.unet_c_5.get_shape().as_list()[1:]
            self.unet_cdb_2_1 = tf.concat([self.unet_cdb_2, self.unet_c_6], axis=3)
            unet_cdb_3 = deconv2d_1(input=self.unet_cdb_2_1, width=list_unet_cb_5[0], height=list_unet_cb_5[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                        k=3, s=2, batch_size=self.batch_size, name="unet_cdb_3/deconv2d", stddev=0.02,
                                        activation_fn=None)

            self.reg_unet_cbd_3 = tf.nn.tanh(unet_cdb_3[:, :, :, :1])
            reg_boundary_list.append(self.reg_unet_cbd_3)
            self.unet_cdb_3 = tf.nn.relu(unet_cdb_3)

            list_unet_cb_4 = self.unet_c_4.get_shape().as_list()[1:]
            self.unet_cdb_3_1 = tf.concat([self.unet_cdb_3, self.unet_c_5], axis=3)
            unet_cdb_4 = deconv2d_1(input=self.unet_cdb_3_1, width=list_unet_cb_4[0], height=list_unet_cb_4[1],
                                        input_channels=1024, output_channels=512, is_train=self.is_train,
                                        k=3, s=2, batch_size=self.batch_size, name="unet_cdb_4/deconv2d", stddev=0.02,
                                        activation_fn=None)

            self.reg_unet_cbd_4 = tf.nn.tanh(unet_cdb_4[:, :, :, :1])
            reg_boundary_list.append(self.reg_unet_cbd_4)
            self.unet_cdb_4 = tf.nn.relu(unet_cdb_4)

            list_unet_cb_3 = self.unet_c_3.get_shape().as_list()[1:]
            self.unet_cdb_4_1 = tf.concat([self.unet_cdb_4, self.unet_c_4], axis=3)
            unet_cdb_5 = deconv2d_1(input=self.unet_cdb_4_1, width=list_unet_cb_3[0], height=list_unet_cb_3[1],
                                        input_channels=1024, output_channels=256, is_train=self.is_train,
                                        k=3, s=2, batch_size=self.batch_size, name="unet_cdb_5/deconv2d", stddev=0.02,
                                        activation_fn=None)

            self.reg_unet_cbd_5 = tf.nn.tanh(unet_cdb_5[:, :, :, :1])
            reg_boundary_list.append(self.reg_unet_cbd_5)
            self.unet_cdb_5 = tf.nn.relu(unet_cdb_5)

            list_unet_cb_2 = self.unet_c_2.get_shape().as_list()[1:]
            self.unet_cdb_5_1 = tf.concat([self.unet_cdb_5, self.unet_c_3], axis=3)
            unet_cdb_6 = deconv2d_1(input=self.unet_cdb_5_1, width=list_unet_cb_2[0], height=list_unet_cb_2[1],
                                        input_channels=512, output_channels=128, is_train=self.is_train,
                                        k=3, s=2, batch_size=self.batch_size, name="unet_cdb_6/deconv2d", stddev=0.02,
                                        activation_fn=None)

            self.reg_unet_cbd_6 = tf.nn.tanh(unet_cdb_6[:, :, :, :1])
            reg_boundary_list.append(self.reg_unet_cbd_6)
            self.unet_cdb_6 = tf.nn.relu(unet_cdb_6)

            list_unet_cb_1 = self.unet_c_1.get_shape().as_list()[1:]
            self.unet_cdb_7_1 = tf.concat([self.unet_cdb_6, self.unet_c_2], axis=3)
            unet_cdb_7 = deconv2d_1(input=self.unet_cdb_7_1, width=list_unet_cb_1[0], height=list_unet_cb_1[1],
                                        input_channels=256, output_channels=64, is_train=self.is_train,
                                        k=3, s=2, batch_size=self.batch_size, name="unet_cdb_7/deconv2d", stddev=0.02,
                                        activation_fn=None)

            self.reg_unet_cbd_7 = tf.nn.tanh(unet_cdb_7[:, :, :, :1])
            reg_boundary_list.append(self.reg_unet_cbd_7)
            self.unet_cdb_7 = tf.nn.relu(unet_cdb_7)

            list_input = image.get_shape().as_list()[1:]
            self.unet_cdb_8_1 = tf.concat([self.unet_cdb_7, self.unet_c_1], axis=3)
            self.unet_b_out = deconv2d_1(input=self.unet_cdb_8_1, width=list_input[0], height=list_input[1],
                                       input_channels=128, output_channels=1, is_train=self.is_train,
                                       k=3, s=2, batch_size=self.batch_size, name="unet_out/deconv2d", stddev=0.02,
                                       activation_fn=tf.nn.tanh)

            return self.unet_out, self.unet_b_out, reg_mask_list, reg_boundary_list


    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            self.disc_c_1 = conv2d(input=image, output_channles=64, is_train=self.is_train,
                                   k_h=3, k_w=3, stddev=0.02, name='disc1/conv2d')
            self.disc_c_2 = conv2d(input=self.disc_c_1, output_channles=128, is_train=self.is_train,
                                   k_h=3, k_w=3, stddev=0.02, name='disc2/conv2d')
            self.disc_c_3 = conv2d(input=self.disc_c_2, output_channles=256, is_train=self.is_train,
                                   k_h=3, k_w=3, stddev=0.02, name='disc3/conv2d')
            self.disc_c_4 = conv2d(input=self.disc_c_3, output_channles=512, is_train=self.is_train,
                                   k_h=3, k_w=3, stddev=0.02, name='disc4/conv2d')
            self.disc_c_5 = conv2d(input=self.disc_c_4, output_channles=512, is_train=self.is_train,
                                   k_h=3, k_w=3, stddev=0.02, name='disc5/conv2d')
            self.disc_c_6 = conv2d(input=self.disc_c_5, output_channles=512, is_train=self.is_train,
                                   k_h=3, k_w=3, stddev=0.02, name='disc6/conv2d')
            self.disc_c_7 = conv2d(input=self.disc_c_6, output_channles=512, is_train=self.is_train,
                                   k_h=3, k_w=3, stddev=0.02, name='disc7/conv2d')

            self.avg_pool = tf.reduce_mean(self.disc_c_7, [1, 2])
            self.fc = tf.reshape(self.avg_pool, [self.batch_size, 512])
            self.disc_out = tf.layers.dense(self.fc, 1, activation=tf.nn.sigmoid)

            return self.disc_out

    def train(self):
        self.img, self.mask_boundary, self.boundary = load_batch(self.batch_size, 256, 256, shuffle=True)

        self.fake, self.fake_boundary, self.reg_mask_list, self.reg_boundary_list = self.unet(self.img)  # fake A
        self.fake_sum = tf.summary.image("fake_B", self.fake)
        self.fake_boundary_sum = tf.summary.image('fake_boundary', self.fake_boundary)
        self.real_img_sum = tf.summary.image('real_image', self.mask_boundary)
        self.real_boundary_sum = tf.summary.image('real_boundary', self.boundary)

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
                      + self.l1_lambda * (tf.reduce_mean(tf.square(self.mask_boundary - self.fake)) +
                                          tf.reduce_mean(tf.square(self.reg_mask_list[0] - tf.image.resize_images(self.fake, (8, 8)))) +
                                          tf.reduce_mean(tf.square(self.reg_mask_list[1] - tf.image.resize_images(self.fake, (16, 16)))) +
                                          tf.reduce_mean(tf.square(self.reg_mask_list[2] - tf.image.resize_images(self.fake, (32, 32)))) +
                                          tf.reduce_mean(tf.square(self.reg_mask_list[3] - tf.image.resize_images(self.fake, (64, 64)))) +
                                          tf.reduce_mean(tf.square(self.reg_mask_list[4] - tf.image.resize_images(self.fake, (128, 128)))))\
                      + self.rec_rate * (tf.reduce_mean(tf.square(self.boundary - self.fake_boundary))
                                         + tf.reduce_mean(tf.square(self.reg_boundary_list[0] - tf.image.resize_images(self.boundary, (8, 8)))) +
                                                           tf.reduce_mean(tf.square(self.reg_boundary_list[1] - tf.image.resize_images(self.boundary, (16, 16)))) +
                                                           tf.reduce_mean(tf.square(self.reg_boundary_list[2] - tf.image.resize_images(self.boundary, (32, 32)))) +
                                                           tf.reduce_mean(tf.square(self.reg_boundary_list[3] - tf.image.resize_images(self.boundary, (64, 64)))) +
                                                           tf.reduce_mean(tf.square(self.reg_boundary_list[4] - tf.image.resize_images(self.boundary, (128, 128)))))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        self.g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

        d_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        self.saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()
        init_op1 = tf.local_variables_initializer()
        self.sess.run([init_op, init_op1])

        # model_file = tf.train.latest_checkpoint('./checkpoint/')
        # self.saver.restore(self.sess, model_file)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            for step in range(self.epoches * self.batch_size):

                _, D_loss, mask_boundary = self.sess.run([d_optim, self.d_loss, self.mask_boundary])

                _, summary, G_loss, fake_mask_boundary, fake_boundary = self.sess.run([g_optim, self.summary_op,
                                                                        self.g_loss, self.fake, self.fake_boundary])

                self.writer.add_summary(summary, step)
                print('step is %d' % step)


                if np.mod(step, 400) == 0:
                    self.saver.save(self.sess, './checkpoint/checkpoint.ckpt', step)

        except tf.errors.OutOfRangeError:
            print('Done train %d steps!' % step)

        finally:
            coord.request_stop()
            coord.join(threads)






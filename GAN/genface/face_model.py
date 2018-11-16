from scipy.misc import imresize
import tensorflow as tf
import numpy as np
import datetime
from GAN.layer import convolution_2d, deconvolution_2d, leaky_relu, weight_xavier_init, bias_variable, save_images


# Serve data by batches
def _next_batch(train_images, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], index_in_epoch


class WGAN_GPModel(object):
    def __init__(self, image_height, image_width, channels=3, z_dim=128, batch_size=64):
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.X = tf.placeholder(tf.float32, shape=[batch_size, image_height, image_width, channels])
        self.Z = tf.placeholder("float", shape=[batch_size, z_dim])
        self.D_real = self._GAN_discriminator(self.X, reuse=False)
        self.Gen = self._GAN_generator(self.Z, z_dim)
        self.D_fake = self._GAN_discriminator(self.Gen, reuse=True)
        self.d_loss, self.g_loss = self._Loss_Function(scale=10)

    def _GAN_generator(self, z, z_dim, reuse=False):
        # Model Parameters
        with tf.variable_scope("generator", reuse=reuse):
            # Layer1
            g_w1 = weight_xavier_init(shape=[z_dim, 1024], n_inputs=z_dim, n_outputs=1024, variable_name='g_w1')
            g_b1 = bias_variable(shape=[1024], variable_name='g_b1')
            g1 = tf.matmul(z, g_w1) + g_b1
            g1 = tf.nn.relu(g1)

            # Layer2
            g_w2 = weight_xavier_init(shape=[1024, 512 * 6 * 6], n_inputs=1024, n_outputs=512 * 6 * 6,
                                      variable_name='g_w2')
            g_b2 = bias_variable(shape=[512 * 6 * 6], variable_name='g_b2')
            g2 = tf.matmul(g1, g_w2) + g_b2
            g2 = tf.nn.relu(g2)
            g2 = tf.reshape(g2, shape=[-1, 6, 6, 512])
            # Layer3
            g_w3 = weight_xavier_init(shape=[5, 5, 256, 512], n_inputs=512, n_outputs=256,
                                      variable_name='g_w3')
            g_b3 = bias_variable(shape=[256], variable_name='g_b3')
            g3 = deconvolution_2d(g2, g_w3, 2)
            g3 = g3 + g_b3
            g3 = tf.nn.relu(g3)
            # Layer4
            g_w4 = weight_xavier_init(shape=[5, 5, 128, 256], n_inputs=256, n_outputs=128,
                                      variable_name='g_w4')
            g_b4 = bias_variable(shape=[128], variable_name='g_b4')
            g4 = deconvolution_2d(g3, g_w4, 2)
            g4 = g4 + g_b4
            g4 = tf.nn.relu(g4)
            # Layer5
            g_w5 = weight_xavier_init(shape=[5, 5, 64, 128], n_inputs=128, n_outputs=64,
                                      variable_name='g_w5')
            g_b5 = bias_variable(shape=[64], variable_name='g_b5')
            g5 = deconvolution_2d(g4, g_w5, 2)
            g5 = g5 + g_b5
            g5 = tf.nn.relu(g5)
            # Layer6
            g_w6 = weight_xavier_init(shape=[5, 5, 32, 64], n_inputs=64, n_outputs=32,
                                      variable_name='g_w6')
            g_b6 = bias_variable(shape=[32], variable_name='g_b6')
            g6 = deconvolution_2d(g5, g_w6, 2)
            g6 = g6 + g_b6
            g6 = tf.nn.relu(g6)

            # Final convolution with one output channel
            g_w7 = weight_xavier_init(shape=[1, 1, 32, self.channels], n_inputs=32, n_outputs=self.channels,
                                      variable_name='g_w7')
            g_b7 = bias_variable(shape=[self.channels], variable_name='g_b7')
            g7 = convolution_2d(g6, g_w7)
            g7 = g7 + g_b7
            out = tf.nn.sigmoid(g7)
        return out

    def _GAN_discriminator(self, X, reuse=False):
        # Model Parameters
        # CNN model
        with tf.variable_scope("discriminator", reuse=reuse):
            X1 = tf.reshape(X, shape=[-1, self.image_width, self.image_height, self.channels])
            # First convolutional and pool layers
            # This finds 32 different 5 x 5 pixel features
            d_w1 = weight_xavier_init(shape=[5, 5, self.channels, 64], n_inputs=self.channels, n_outputs=64,
                                      variable_name='d_w1')
            d_b1 = bias_variable(shape=[64], variable_name='d_b1')
            d1 = convolution_2d(X1, d_w1, stride=2)
            d1 = d1 + d_b1
            d1 = leaky_relu(d1)

            # Second convolutional
            # This finds 64 different 5 x 5 pixel features
            d_w2 = weight_xavier_init(shape=[5, 5, 64, 128], n_inputs=64, n_outputs=128,
                                      variable_name='d_w2')
            d_b2 = bias_variable(shape=[128], variable_name='d_b2')
            d2 = convolution_2d(d1, d_w2, stride=2)
            d2 = d2 + d_b2
            d2 = leaky_relu(d2)
            # Second convolutional one
            d_w2_2 = weight_xavier_init(shape=[5, 5, 128, 256], n_inputs=128, n_outputs=256,
                                        variable_name='d_w2_2')
            d_b2_2 = bias_variable(shape=[256], variable_name='d_b2_2')
            d2_2 = convolution_2d(d2, d_w2_2, stride=2)
            d2_2 = d2_2 + d_b2_2
            d2_2 = leaky_relu(d2_2)
            # Second convolutional one
            d_w2_3 = weight_xavier_init(shape=[5, 5, 256, 512], n_inputs=256, n_outputs=512,
                                        variable_name='d_w2_3')
            d_b2_3 = bias_variable(shape=[512], variable_name='d_b2_3')
            d2_3 = convolution_2d(d2_2, d_w2_3, stride=2)
            d2_3 = d2_3 + d_b2_3
            d2_3 = leaky_relu(d2_3)
            # First fully connected layer
            d_w3 = weight_xavier_init(shape=[6 * 6 * 512, 1024], n_inputs=6 * 6 * 512, n_outputs=1024,
                                      variable_name='d_w3')
            d_b3 = bias_variable(shape=[1024], variable_name='d_b3')
            d3 = tf.reshape(d2_3, [-1, 6 * 6 * 512])
            d3 = tf.matmul(d3, d_w3)
            d3 = d3 + d_b3
            d3 = leaky_relu(d3)

            # Second fully connected layer
            d_w4 = weight_xavier_init(shape=[1024, self.channels], n_inputs=1024, n_outputs=self.channels,
                                      variable_name='d_w4')
            d_b4 = bias_variable(shape=[self.channels], variable_name='d_b4')
            out_logit = tf.matmul(d3, d_w4) + d_b4
            return out_logit

    def _Loss_Function(self, scale=10.0):
        d_loss = tf.reduce_mean(self.D_real) - tf.reduce_mean(self.D_fake)
        g_loss = tf.reduce_mean(self.D_fake)
        # losses
        """ Gradient Penalty """
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        alpha = tf.random_uniform(shape=self.X.get_shape(), minval=0., maxval=1.)
        differences = self.Gen - self.X  # This is different from MAGAN
        interpolates = self.X + (alpha * differences)
        D_inter = self._GAN_discriminator(interpolates, reuse=True)
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        d_loss += scale * gradient_penalty
        return d_loss, g_loss

    def train(self, train_images, model_path, logs_path, learning_rate=1e-4, train_epochs=1000):
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        D_vars = [var for var in t_vars if 'd_' in var.name]
        G_vars = [var for var in t_vars if 'g_' in var.name]

        trainD_op = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=D_vars)
        trainG_op = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=G_vars)

        tf.get_variable_scope().reuse_variables()

        """ Summary """
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        g_sum_op = tf.summary.merge([g_loss_sum])
        d_sum_op = tf.summary.merge([d_loss_sum])
        '''
        TensorFlow Session
        '''
        # start TensorFlow session
        init = tf.initialize_all_variables()
        saver = tf.train.Saver(tf.all_variables())
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        logdir = logs_path + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
        sess.run(init)

        DISPLAY_STEP = 10
        index_in_epoch = 0

        # Train generator and discriminator together
        for i in range(train_epochs):
            d_iters = 5
            # get new batch
            batch_xs, index_in_epoch = _next_batch(train_images, self.batch_size, index_in_epoch)
            for _ in range(d_iters):
                # train on batch
                # Train discriminator on both real and fake images
                z_batch = np.random.normal(0, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                _, summaryD, dLoss = sess.run([trainD_op, d_sum_op, self.d_loss],
                                              feed_dict={self.X: batch_xs, self.Z: z_batch})
                summary_writer.add_summary(summaryD, i)
            # Train generator
            z_batch = np.random.normal(0, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
            _, summaryG, gLoss = sess.run([trainG_op, g_sum_op, self.g_loss],
                                          feed_dict={self.Z: z_batch})
            summary_writer.add_summary(summaryG, i)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                print("=========== updating G&D ==========")
                print("iteration:", i)
                print("gen loss:", gLoss)
                print("dis loss:", dLoss)
                z_batch = np.random.normal(0, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                outimage = self.Gen.eval(feed_dict={self.Z: z_batch}, session=sess)
                save_images(outimage, [8, 8], 'img/' + 'sampleface_%d_epoch.png' % (i))

            if i % (DISPLAY_STEP * 10) == 0 and i:
                DISPLAY_STEP *= 10

        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, model_path, test_image, scale_factor=2):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver.restore(sess, model_path)
        init_width, init_height = test_image.shape[0], test_image.shape[1]
        intermediate_img = imresize(test_image, (init_width * scale_factor, init_height * scale_factor))
        # Normalize from [0:255] => [0.0:1.0]
        img_conv = intermediate_img / 255.
        img_conv = np.reshape(img_conv, (1, init_width * scale_factor * init_height * scale_factor))
        srimage = self.Gen.eval(feed_dict={self.X: img_conv}, session=sess)

        result = srimage.astype(np.float32) * 255.
        result = np.clip(result, 0, 255).astype('uint8')
        result = np.reshape(result, (init_width * scale_factor, init_height * scale_factor))
        return result, intermediate_img

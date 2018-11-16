import tensorflow as tf
import numpy as np
import datetime
import time
from GAN.layer import convolution_2d, average_pool_2x2, deconvolution_2d, weight_xavier_init, bias_variable, leaky_relu, \
    save_images
import cv2


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


class GANModel(object):
    def __init__(self, image_height, image_width, channels=1, z_dim=100):
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.z_dim = z_dim
        self.X = tf.placeholder("float", shape=[None, image_height * image_width * channels])
        self.Z = tf.placeholder("float", shape=[None, z_dim])
        self.phase = tf.placeholder(tf.bool)

        self.D_real, self.D_real_logit = self._GAN_discriminator(self.X, self.image_width, self.image_height,
                                                                 reuse=False)
        self.Gen = self._GAN_generator(self.Z, z_dim)
        self.D_fake, self.D_fake_logit = self._GAN_discriminator(self.Gen, self.image_width, self.image_height,
                                                                 reuse=True)
        # self.d_loss_real, self.d_loss_fake = self._Loss_discriminator()
        # self.d_loss = self.d_loss_real + self.d_loss_fake
        self.d_loss = self._Loss_discriminator()
        self.g_loss = self._Loss_generator()

    def _GAN_generator(self, z, z_dim):
        # Model Parameters
        g_w1 = tf.get_variable('g_w1', [z_dim, 1024], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b1 = tf.get_variable('g_b1', [1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g1 = tf.matmul(z, g_w1) + g_b1
        # g1 = tf.contrib.layers.batch_norm(g1, center=True, scale=True, is_training=self.phase, scope='bn1')
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, is_training=self.phase, scope='bn1')
        g1 = tf.nn.relu(g1)

        # Generate 50 features
        g_w2 = tf.get_variable('g_w2', [1024, 64 * 7 * 7], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b2 = tf.get_variable('g_b2', [64 * 7 * 7], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g2 = tf.matmul(g1, g_w2) + g_b2
        # g2 = tf.contrib.layers.batch_norm(g2, center=True, scale=True, is_training=self.phase, scope='bn2')
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, is_training=self.phase, scope='bn2')
        g2 = tf.nn.relu(g2)
        g2 = tf.reshape(g2, shape=[-1, 7, 7, 64])
        # Generate 25 features
        g_w3 = tf.get_variable('g_w3', [5, 5, 32, 64], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b3 = tf.get_variable('g_b3', [32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g3 = deconvolution_2d(g2, g_w3)
        g3 = g3 + g_b3
        # g3 = tf.contrib.layers.batch_norm(g3, center=True, scale=True, is_training=self.phase, scope='bn3')
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, is_training=self.phase, scope='bn3')
        g3 = tf.nn.relu(g3)

        # Final convolution with one output channel
        g_w4 = tf.get_variable('g_w4', [5, 5, 16, 32], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b4 = tf.get_variable('g_b4', [16], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g4 = deconvolution_2d(g3, g_w4)
        g4 = g4 + g_b4
        g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5, is_training=self.phase, scope='bn4')
        g4 = tf.nn.relu(g4)

        g_w5 = tf.get_variable('g_w5', [1, 1, 16, 1], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b5 = tf.get_variable('g_b5', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g5 = convolution_2d(g4, g_w5)
        g5 = g5 + g_b5
        g5 = tf.nn.tanh(g5)

        return g5

    def _GAN_discriminator(self, X, image_width, image_height, image_channel=1, reuse=False):
        # Model Parameters
        # CNN model
        with tf.variable_scope("discriminator", reuse=reuse):
            X1 = tf.reshape(X, shape=[-1, image_width, image_height, image_channel])
            # First convolutional and pool layers
            # This finds 32 different 5 x 5 pixel features
            d_w1 = tf.get_variable('d_w1', shape=[5, 5, 1, 32],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
            d1 = convolution_2d(X1, d_w1)
            d1 = d1 + d_b1
            d1 = tf.nn.relu(d1)
            d1 = average_pool_2x2(d1)

            # Second convolutional and pool layers
            # This finds 64 different 5 x 5 pixel features
            d_w2 = tf.get_variable('d_w2', shape=[5, 5, 32, 64],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
            d2 = convolution_2d(d1, d_w2)
            d2 = d2 + d_b2
            d2 = tf.nn.relu(d2)
            d2 = average_pool_2x2(d2)

            # First fully connected layer
            d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
            d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
            d3 = tf.matmul(d3, d_w3)
            d3 = d3 + d_b3
            d3 = tf.nn.relu(d3)

            # Second fully connected layer
            d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
            out_logit = tf.matmul(d3, d_w4) + d_b4
            out = tf.nn.sigmoid(out_logit)
            return out, out_logit

    def _Loss_discriminator(self):
        # d_loss_real = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logit, labels=tf.ones_like(self.D_real)))
        # d_loss_fake = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logit, labels=tf.zeros_like(self.D_fake)))
        # return d_loss_real, d_loss_fake
        return -tf.reduce_mean(tf.log(self.D_real) + tf.log(1 - self.D_fake))

    def _Loss_generator(self):
        # g_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logit, labels=tf.ones_like(self.D_fake)))
        # return g_loss
        return -tf.reduce_mean(tf.log(self.D_fake))

    def train(self, train_images, model_path, logs_path, learning_rate=1e-4, beta1=0.9, train_epochs=100,
              batch_size=128):
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        D_vars = [var for var in t_vars if 'd_' in var.name]
        G_vars = [var for var in t_vars if 'g_' in var.name]

        trainD_op = tf.train.AdamOptimizer(learning_rate, beta1).minimize(self.d_loss, var_list=D_vars)
        trainG_op = tf.train.AdamOptimizer(learning_rate, beta1).minimize(self.g_loss, var_list=G_vars)

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
        sess = tf.InteractiveSession()
        logdir = logs_path + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
        sess.run(init)

        DISPLAY_STEP = 10
        index_in_epoch = 0

        # Pre-train discriminator
        for i in range(30):
            z_batch = np.random.normal(0, 1, size=[batch_size, self.z_dim]).astype(np.float32)
            batch_xs, index_in_epoch = _next_batch(train_images, batch_size, index_in_epoch)
            sess.run([trainD_op], feed_dict={self.X: batch_xs, self.Z: z_batch, self.phase: 1})
        # Train generator and discriminator together
        for i in range(train_epochs):
            # get new batch
            z_batch = np.random.normal(0, 1, size=[batch_size, self.z_dim]).astype(np.float32)
            batch_xs, index_in_epoch = _next_batch(train_images, batch_size, index_in_epoch)
            # train on batch
            # Train discriminator on both real and fake images
            _, summaryD = sess.run([trainD_op, d_sum_op],
                                   feed_dict={self.X: batch_xs, self.Z: z_batch, self.phase: 1})
            summary_writer.add_summary(summaryD, i)
            # Train generator
            _, summaryG = sess.run([trainG_op, g_sum_op], feed_dict={self.X: batch_xs, self.Z: z_batch, self.phase: 1})
            summary_writer.add_summary(summaryG, i)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                dLoss, gLoss = sess.run([self.d_loss, self.g_loss],
                                        feed_dict={self.X: batch_xs, self.Z: z_batch, self.phase: 1})
                print("=========== updating G&D ==========")
                print("iteration:", i)
                print("gen loss:", gLoss)
                print("dis loss:", dLoss)

                outimage = self.Gen.eval(feed_dict={self.Z: z_batch, self.phase: 1}, session=sess)

                for index in range(3):
                    result = (outimage[index].astype(np.float32)) * 255.
                    result = np.clip(result, 0, 255).astype('uint8')
                    result = np.reshape(result, (28, 28))
                    cv2.imwrite("out" + str(index + 1) + ".bmp", result)

                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, model_path, batch_size=1):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver.restore(sess, model_path)
        # Normalize from [0:255] => [0.0:1.0]
        z_batch = np.random.normal(0, 1, size=[batch_size, self.z_dim]).astype(np.float32)
        outimage = self.Gen.eval(feed_dict={self.Z: z_batch, self.phase: 1}, session=sess)

        result = outimage.astype(np.float32) * 255.
        result = np.clip(result, 0, 255).astype('uint8')
        result = np.reshape(result, (-1, 28, 28))
        return result


class WGANModel(object):
    def __init__(self, image_height, image_width, channels=1, z_dim=100, batch_size=64):
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.X = tf.placeholder("float", shape=[batch_size, image_height, image_width, channels])
        self.Z = tf.placeholder("float", shape=[batch_size, z_dim])
        self.phase = tf.placeholder(tf.bool)

        self.D_real = self._GAN_discriminator(self.X, reuse=False)
        self.Gen = self._GAN_generator(self.Z, z_dim)
        self.D_fake = self._GAN_discriminator(self.Gen, reuse=True)
        self.d_loss, self.g_loss = self._Loss_Function(scale=10)

    def _GAN_generator(self, z, z_dim, reuse=False):
        # Model Parameters
        with tf.variable_scope("generator", reuse=reuse):
            g_w1 = weight_xavier_init(shape=[z_dim, 1024], n_inputs=z_dim, n_outputs=1024, variable_name='g_w1')
            g_b1 = bias_variable(shape=[1024], variable_name='g_b1')
            g1 = tf.matmul(z, g_w1) + g_b1
            g1 = tf.nn.relu(g1)

            # Generate 50 features
            g_w2 = weight_xavier_init(shape=[1024, 128 * 7 * 7], n_inputs=1024, n_outputs=128 * 7 * 7,
                                      variable_name='g_w2')
            g_b2 = bias_variable(shape=[128 * 7 * 7], variable_name='g_b2')
            g2 = tf.matmul(g1, g_w2) + g_b2
            g2 = tf.nn.relu(g2)
            g2 = tf.reshape(g2, shape=[-1, 7, 7, 128])
            # Generate 25 features
            g_w3 = weight_xavier_init(shape=[5, 5, 64, 128], n_inputs=128, n_outputs=64, variable_name='g_w3')
            g_b3 = bias_variable(shape=[64], variable_name='g_b3')
            g3 = deconvolution_2d(g2, g_w3)
            g3 = g3 + g_b3
            g3 = tf.nn.relu(g3)

            # Final convolution with one output channel
            g_w4 = weight_xavier_init(shape=[5, 5, 32, 64], n_inputs=64, n_outputs=32, variable_name='g_w4')
            g_b4 = bias_variable(shape=[32], variable_name='g_b4')
            g4 = deconvolution_2d(g3, g_w4)
            g4 = g4 + g_b4
            g4 = tf.nn.relu(g4)

            g_w5 = weight_xavier_init(shape=[1, 1, 32, 1], n_inputs=32, n_outputs=1, variable_name='g_w5')
            g_b5 = bias_variable(shape=[1], variable_name='g_b5')
            g5 = convolution_2d(g4, g_w5)
            g5 = g5 + g_b5
            out = tf.nn.sigmoid(g5)
        return out

    def _GAN_discriminator(self, X, reuse=True):
        # Model Parameters
        # CNN model
        with tf.variable_scope("discriminator", reuse=reuse):
            X1 = tf.reshape(X, shape=[self.batch_size, self.image_width, self.image_height, self.channels])
            # First convolutional and pool layers
            # This finds 32 different 5 x 5 pixel features
            d_w1 = weight_xavier_init(shape=[5, 5, 1, 64], n_inputs=1, n_outputs=64, variable_name='d_w1')
            d_b1 = bias_variable(shape=[64], variable_name='d_b1')
            d1 = convolution_2d(X1, d_w1, 2)
            d1 = d1 + d_b1
            d1 = leaky_relu(d1)

            # Second convolutional and pool layers
            # This finds 64 different 5 x 5 pixel features
            d_w2 = weight_xavier_init(shape=[5, 5, 64, 128], n_inputs=64, n_outputs=128, variable_name='d_w2')
            d_b2 = bias_variable(shape=[128], variable_name='d_b2')
            d2 = convolution_2d(d1, d_w2, 2)
            d2 = d2 + d_b2
            d2 = leaky_relu(d2)

            # First fully connected layer
            d_w3 = weight_xavier_init(shape=[7 * 7 * 128, 1024], n_inputs=7 * 7 * 128, n_outputs=1024,
                                      variable_name='d_w3')
            d_b3 = bias_variable(shape=[1024], variable_name='d_b3')
            d3 = tf.reshape(d2, [-1, 7 * 7 * 128])
            d3 = tf.matmul(d3, d_w3)
            d3 = d3 + d_b3
            d3 = leaky_relu(d3)

            # Second fully connected layer
            d_w4 = weight_xavier_init(shape=[1024, 1], n_inputs=1024, n_outputs=1,
                                      variable_name='d_w4')
            d_b4 = bias_variable(shape=[1], variable_name='d_b4')
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

    def train(self, train_images, model_path, logs_path, learning_rate=1e-4, train_epochs=100):
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
        start_time = time.time()

        # Train generator and discriminator together
        for i in range(train_epochs):
            d_iters = 5
            # get new batch
            batch_xs, index_in_epoch = _next_batch(train_images, self.batch_size, index_in_epoch)
            for _ in range(0, d_iters):
                z_batch = np.random.normal(0, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                # train on batch
                # Train discriminator on both real and fake images
                _, summaryD, dLoss = sess.run([trainD_op, d_sum_op, self.d_loss],
                                              feed_dict={self.X: batch_xs, self.Z: z_batch, self.phase: 1})
                summary_writer.add_summary(summaryD, i)
            # Train generator
            z_batch = np.random.normal(0, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
            _, summaryG, gLoss = sess.run([trainG_op, g_sum_op, self.g_loss],
                                          feed_dict={self.Z: z_batch, self.phase: 1})
            summary_writer.add_summary(summaryG, i)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                print("=========== updating G&D ==========")
                print('Time: %.2f' % (time.time() - start_time))
                print("iteration:", i)
                print("gen loss:", gLoss)
                print("dis loss:", dLoss)
                z_batch = np.random.normal(0, 1, size=[self.batch_size, self.z_dim]).astype(np.float32)
                outimage = self.Gen.eval(feed_dict={self.Z: z_batch, self.phase: 1}, session=sess)
                save_images(outimage, [8, 8], 'img/' + 'sample_%d_epoch.png' % (i))

                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, model_path, batch_size=1):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver.restore(sess, model_path)
        # Normalize from [0:255] => [0.0:1.0]
        z_batch = np.random.normal(0, 1, size=[batch_size, self.z_dim]).astype(np.float32)
        outimage = self.Gen.eval(feed_dict={self.Z: z_batch, self.phase: 1}, session=sess)
        result = outimage.astype(np.float32) * 255.
        result = np.clip(result, 0, 255).astype('uint8')
        result = np.reshape(result, (-1, 28, 28))
        return result

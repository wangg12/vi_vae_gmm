from __future__ import absolute_import, print_function, division
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(THIS_DIR, '../zhusuan/'))

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import six.moves.cPickle as pickle
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # TkAgg to show
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shutil
import zhusuan as zs
import random
from tqdm import tqdm
from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity

# from examples import conf
from examples.utils import dataset, save_image_collections

def main():
    # manual seed
    #seed = random.randint(0, 10000) # fix seed
    seed = 1234 # N=100, K=3
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


    # load MNIST data ---------------------------------------------------------
    data_path = os.path.join('../data/', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
            dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid]).astype('float32')

    # model parameters --------------------------------------------------------
    K = 10
    D = 40
    dim_z = K
    dim_h = D
    dim_x = x_train.shape[1] # 784
    N = x_train.shape[0]

    # Define training/evaluation parameters ---------------------------------------------
    resume = False
    epoches = 50 # 2000
    save_freq = 5
    batch_size = 100
    train_iters = int(np.ceil(N / batch_size))

    learning_rate = 0.001
    anneal_lr_freq = 10
    anneal_lr_rate = 0.9
    n_particles = 20

    n_gen = 100

    result_path = "./results/3_gmvae"

    @zs.reuse(scope='decoder')
    def vae(observed, n, n_particles, is_training, dim_h=40, dim_z=10, dim_x=784):
        '''decoder: z-->h-->x
        n: batch_size
        dim_z: K = 10
        dim_x: 784
        dim_h: D = 40
        '''
        with zs.BayesianNet(observed=observed) as model:
            normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
            pai = tf.get_variable('pai', shape=[dim_z],
                                dtype=tf.float32,
                                trainable=True,
                                initializer=tf.constant_initializer(1.0)
                                )
            n_pai = tf.tile(tf.expand_dims(pai, 0), [n, 1])
            z = zs.OnehotCategorical('z', logits=n_pai,
                            dtype=tf.float32,
                            n_samples=n_particles
                            )
            mu = tf.get_variable('mu', shape=[dim_z, dim_h],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-1, 1))
            log_sigma = tf.get_variable('log_sigma', shape=[dim_z, dim_h],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-3, -2)
                        )
            h_mean = tf.reshape(tf.matmul(tf.reshape(z, [-1, dim_z]), mu), [n_particles, -1, dim_h]) # [n_particles, None, dim_x]
            h_logstd = tf.reshape(tf.matmul(tf.reshape(z, [-1, dim_z]), log_sigma), [n_particles, -1, dim_h])

            h = zs.Normal('h', mean=h_mean, logstd=h_logstd,
                            #n_samples=n_particles,
                            group_event_ndims=1
                            )
            lx_h = layers.fully_connected(
                            h, 512,
                            # normalizer_fn=layers.batch_norm,
                            # normalizer_params=normalizer_params
                            )
            lx_h = layers.fully_connected(
                            lx_h, 512,
                            # normalizer_fn=layers.batch_norm,
                            # normalizer_params=normalizer_params
                            )
            x_logits = layers.fully_connected(lx_h, dim_x, activation_fn=None) # the log odds of being 1
            x = zs.Bernoulli('x', x_logits,
                            #n_samples=n_particles,
                            group_event_ndims=1)
        return model, x_logits, h, z.tensor


    @zs.reuse(scope='encoder')
    def q_net(x, dim_h, n_particles, is_training):
        '''encoder: x-->h'''
        with zs.BayesianNet() as variational:
            normalizer_params = {'is_training': is_training,
                             # 'updates_collections': None
                             }
            lh_x = layers.fully_connected(tf.to_float(x), 512,
                            # normalizer_fn=layers.batch_norm,
                            # normalizer_params=normalizer_params,
                            weights_initializer=tf.contrib.layers.xavier_initializer())
            lh_x = tf.contrib.layers.dropout(lh_x, keep_prob=0.9, is_training=is_training)
            lh_x = layers.fully_connected(lh_x, 512,
                            # normalizer_fn=layers.batch_norm,
                            # normalizer_params=normalizer_params,
                            weights_initializer=tf.contrib.layers.xavier_initializer())
            lh_x = tf.contrib.layers.dropout(lh_x, keep_prob=0.9, is_training=is_training)
            h_mean = layers.fully_connected(lh_x, dim_h, activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer())
            h_logstd = layers.fully_connected(lh_x, dim_h, activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer())
            h = zs.Normal('h', mean=h_mean, logstd=h_logstd,
                            n_samples=n_particles,
                            group_event_ndims=1
                            )
        return variational


    x_ph = tf.placeholder(tf.int32, shape=[None, dim_x], name='x_ph')
    x_orig_ph = tf.placeholder(tf.float32, shape=[None, dim_x], name='x_orig_ph')
    x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig_ph), 0, 1), x_orig_ph), tf.int32)
    is_training_ph = tf.placeholder(tf.bool, shape=[], name='is_training_ph')

    n = tf.shape(x_ph)[0]


    def log_joint(observed):
        z_obs = tf.eye(dim_z, batch_shape=[n_particles, n])
        z_obs = tf.transpose(z_obs, [2, 0, 1, 3]) # [K, n_p, bs, K]
        log_pz_list = []
        log_ph_z_list = []
        log_px_h = None
        for i in range(dim_z):
            observed['z'] = z_obs[i,:] # the i-th dimension is 1
            model, _, _, _ = vae(observed, n, n_particles, is_training_ph, dim_h=dim_h, dim_z=dim_z, dim_x=dim_x)
            log_pz_i, log_ph_z_i, log_px_h = model.local_log_prob(['z', 'h', 'x'])
            log_pz_list.append(log_pz_i)
            log_ph_z_list.append(log_ph_z_i)
        log_pz = tf.stack(log_pz_list, axis=0)
        log_ph_z = tf.stack(log_ph_z_list, axis=0)
        # p(X, H) = p(X|H) sum_Z(p(Z) * p(H|Z))
        # log p(X, H) = log p(X|H) + log sum_Z exp(log p(Z) + log p(H|Z))
        log_p_xh = log_px_h + tf.reduce_logsumexp(log_pz + log_ph_z, axis=0) # log p(X, H)
        return log_p_xh

    variational = q_net(x_ph, dim_h, n_particles, is_training_ph)
    qh_samples, log_qh = variational.query('h', outputs=True,
                                           local_log_prob=True)

    x_obs = tf.tile(tf.expand_dims(x_ph, 0), [n_particles, 1, 1])

    lower_bound = zs.sgvb(log_joint,
                            observed={'x': x_obs},
                            latent={'h': [qh_samples, log_qh]},
                            axis=0)

    mean_lower_bound = tf.reduce_mean(lower_bound)
    with tf.name_scope('neg_lower_bound'):
        neg_lower_bound = tf.reduce_mean(- mean_lower_bound)

    train_vars = tf.trainable_variables()
    with tf.variable_scope('decoder', reuse=True):
        pai = tf.get_variable('pai')
        mu = tf.get_variable('mu')
        log_sigma = tf.get_variable('log_sigma')

    clip_pai = pai.assign(tf.clip_by_value(pai, 0.7, 1.3))

    # _, pai_var = tf.nn.moments(pai, axes=[-1])
    # _, mu_var = tf.nn.moments(mu, axes=[0, 1], keep_dims=False)
    # regularizer = tf.add_n([tf.nn.l2_loss(v) for v in train_vars
    #                     if not 'pai' in v.name and not 'mu' in v.name])
    # loss = neg_lower_bound + pai_var - mu_var # + 1e-4 * regularizer # loss -------------
    loss = neg_lower_bound #+ 0.001 * tf.nn.l2_loss(mu-1)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')

    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars]
    infer = optimizer.apply_gradients(clipped_gvs)

    # Generate images -----------------------------------------------------
    z_manual_feed = tf.eye(dim_z, batch_shape=[10]) # [10, K, K]
    z_manual_feed = tf.transpose(z_manual_feed, [1, 0, 2]) # [K, 10, K]
    _, x_logits, _, z_onehot = vae({'z': z_manual_feed}, 10, n_particles=1, is_training=False,
                                    dim_h=dim_h, dim_z=dim_z, dim_x=dim_x) # n and n_particles do not matter, since we have manually feeded z
    print('x_logits:', x_logits.shape.as_list()) # [1, 100, 784]
    x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1])
    z_gen = tf.argmax(tf.reshape(z_onehot, [-1, dim_z]), axis=1)


    # tensorboard summary ---------------------------------------------------
    image_for_summ = []
    for i in range(n_gen//10):
        tmp = [x_gen[j+i*10,:] for j in range(10)]
        tmp = tf.concat(tmp, 1)
        image_for_summ.append(tmp)
    image_for_summ = tf.expand_dims(tf.concat(image_for_summ, 0), 0)
    print('image_for_summ:', image_for_summ.shape.as_list())
    gen_image_summ = tf.summary.image('gen_images', image_for_summ, max_outputs=100)
    lb_summ = tf.summary.scalar("lower_bound", mean_lower_bound)
    lr_summ = tf.summary.scalar("learning_rate", learning_rate_ph)
    loss_summ = tf.summary.scalar('loss', loss)

    for var in train_vars:
        tf.summary.histogram(var.name, var)
    for grad, _ in grads_and_vars:
        tf.summary.histogram(grad.name, grad)

    for i in train_vars:
        print(i.name, i.get_shape())
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=10)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None and resume: # resume ---------------------------------------
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        x_train_normed = x_train # no normalization
        x_train_normed_no_shuffle = x_train_normed


        log_dir = './log/3_gmvae/'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        global mu_res, log_sigma_res, pai_res
        global gen_images, z_gen_res, epoch
        print('training...') # ----------------------------------------------------------------
        pai_res_0, mu_res_0, log_sigma_res_0 = sess.run([pai, mu, log_sigma])
        global_step = 0
        for epoch in tqdm(range(begin_epoch, epoches + 1)):
            time_epoch = -time.time()
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train_normed) # shuffle training data
            lbs = []

            for t in tqdm(range(train_iters)):
                global_step += 1
                x_batch = x_train_normed[t * batch_size : (t + 1) * batch_size] # get batched data
                x_batch_bin = sess.run(x_bin, feed_dict={x_orig_ph: x_batch})
                # sess.run(clip_pai)
                _, lb, merge_all = sess.run([infer, mean_lower_bound, merged_summary_op],
                                feed_dict={x_ph: x_batch_bin,
                                           learning_rate_ph: learning_rate,
                                           is_training_ph: True})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                    epoch, time_epoch, np.mean(lbs)))
                # print(grad_var_res[-3:])


            summary_writer.add_summary(merge_all, global_step=epoch)

            if epoch % save_freq == 0: # save ---------------------------------------------------
                print('Saving model...')
                save_path = os.path.join(result_path, "gmvae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)

                gen_images, z_gen_res = sess.run([x_gen, z_gen]) #, feed_dict={is_training_ph: False})

                # dump data
                pai_res, mu_res, log_sigma_res = sess.run([pai, mu, log_sigma])
                data_dump = {'epoch':epoch,
                        'images': gen_images, 'clusters': z_gen_res,
                        'pai_0': pai_res_0, 'mu_0': mu_res_0, 'log_sigma_0': log_sigma_res_0,
                        'pai_res': pai_res, 'mu_res': mu_res, 'log_sigma_res': log_sigma_res
                        }
                pickle.dump(data_dump, open(os.path.join(result_path, 'gmvae_results_epoch_{}.pkl'.format(epoch)), 'w'), protocol=2)
                save_image_with_clusters(gen_images, z_gen_res, filename="results/3_gmvae/gmvae_epoch_{}.png".format(epoch))
                print('Done')


        pai_res, mu_res, log_sigma_res = sess.run([pai, mu, log_sigma])
        print("Random Seed: ", seed)
        data_dump = {'epoch':epoch,
                    'images': gen_images, 'clusters': z_gen_res,
                    'pai_0': pai_res_0, 'mu_0': mu_res_0, 'log_sigma_0': log_sigma_res_0,
                    'pai_res': pai_res, 'mu_res': mu_res, 'log_sigma_res': log_sigma_res
                    }
        pickle.dump(data_dump, open(os.path.join(result_path, 'gmvae_results_epoch_{}.pkl'.format(epoch)), 'w'), protocol=2)
        plot_images_and_clusters(gen_images, z_gen_res, epoch, save_path=result_path, ncol=10)


def save_images_and_clusters(images, clusters, epoch, shape=(10,10)):
    for i in range(10):
        name_i = "results/3_gmvae/epoch_{}/cluster_{}.png".format(epoch, i)
        images_i = images[clusters==i, :]
        if images_i.shape[0] == 0:
            continue
        save_image_collections(images_i, name_i, shape=shape)


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def save_image_with_clusters(x, clusters, filename, shape=(10, 10), scale_each=False,
                           transpose=False):
    '''single image, each row is a cluster'''
    makedirs(filename)
    n = x.shape[0]

    images = np.zeros_like(x)
    curr_len = 0
    for i in range(10):
        images_i = x[clusters==i, :]
        n_i = images_i.shape[0]
        images[curr_len : curr_len+n_i, :] = images_i
        curr_len += n_i

    x = images

    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))

    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


def plot_images_and_clusters(images, clusters, epoch, save_path, ncol=10):
    '''use multiple images'''
    fig = plt.figure()#facecolor='black')
    images = np.squeeze(images, -1)

    nrow = int(np.ceil(images.shape[0] / float(ncol)))
    gs = gridspec.GridSpec(nrow, ncol,
                        width_ratios=[1]*ncol, height_ratios=[1]*nrow,
        #                         wspace=0.01, hspace=0.001,
        #                         top=0.95, bottom=0.05,
        #                         left=0.05, right=0.95
                        )
    gs.update(wspace=0, hspace=0)
    n = 0
    for i in range(10):
        images_i = images[clusters==i, :, :]
        if images_i.shape[0] == 0:
            continue

        for j in range(images_i.shape[0]):
            ax = plt.subplot(gs[n])
            n += 1
            plt.imshow(images_i[j,:], cmap='gray')
            plt.axis('off')
            ax.set_aspect('auto')
    plt.savefig(os.path.join(save_path, 'plot_gmvae_epoch_{}.png'.format(epoch)), dpi=fig.dpi)

if __name__ == "__main__":
    main()






from __future__ import absolute_import, print_function, division
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(THIS_DIR, '../zhusuan/'))

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # TkAgg to show
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import hickle
import io
import shutil
import zhusuan as zs
import random

# from examples import conf
from examples.utils import dataset, save_image_collections


def main(N=100, K=3):
    # manual seed
    seed = random.randint(0, 10000) # fix seed
    # seed = 3899 # N=100, K=3
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # tf.set_random_seed(2333)
    # np.random.seed(4567)

    D = 2
    n_gen = N # the number of generated samples x
    dim_z = K
    dim_x = D

    # Define training parameters ---------------------------------------------
    epoches = 2000
    batch_size = min(100, N)
    iters_per_batch = N // batch_size
    save_freq = 10
    plot_freq = 200
    learning_rate = 0.001
    anneal_lr_freq = 200
    anneal_lr_rate = 0.9
    n_particles = 100

    @zs.reuse(scope='decoder')
    def vae(observed, n, dim_x, dim_z, n_particles):
        '''decoder: z-->x'''
        with zs.BayesianNet(observed=observed) as model:
            pai = tf.get_variable('pai', shape=[dim_z],
                                dtype=tf.float32,
                                trainable=True,
                                initializer=tf.constant_initializer(1.0), #tf.random_uniform_initializer(),  #tf.ones([dim_z]),
                                )
            n_pai = tf.tile(tf.expand_dims(pai, 0), [n, 1])
            z = zs.OnehotCategorical('z', logits=n_pai,
                            dtype=tf.float32,
                            n_samples=n_particles
                            #group_event_ndims=1
                            )  # zhusuan.model.stochastic.OnehotCategorical
            print('-'*10, 'z:', z.tensor.get_shape().as_list()) # [n_particles, None, dim_z]
            mu = tf.get_variable('mu', shape=[dim_z, dim_x],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(0, 1))
            log_sigma = tf.get_variable('log_sigma', shape=[dim_z, dim_x],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-3, -2)
                        ) # tf.random_normal_initializer(-3, 0.5)) #tf.contrib.layers.xavier_initializer())
            x_mean = tf.reshape(tf.matmul(tf.reshape(z, [-1, dim_z]), mu), [n_particles, n, dim_x]) # [n_particles, None, dim_x]
            x_logstd = tf.reshape(tf.matmul(tf.reshape(z, [-1, dim_z]), log_sigma), [n_particles, n, dim_x])

            # print('x_mean:', x_mean.get_shape().as_list())
            # print('x_logstd:', x_logstd.get_shape().as_list())
            x = zs.Normal('x', mean=x_mean, logstd=x_logstd, group_event_ndims=1)
            # print('x:', x.tensor.get_shape().as_list())
        return model, x.tensor, z.tensor


    @zs.reuse(scope='encoder')
    def q_net(x, dim_z, n_particles):
        '''encoder: x-->z'''
        with zs.BayesianNet() as variational:
            lz_x = layers.fully_connected(tf.to_float(x), 256,
                        weights_initializer=tf.contrib.layers.xavier_initializer())
            # lz_x = layers.fully_connected(lz_x, 256,
            #             weights_initializer=tf.contrib.layers.xavier_initializer())
            z_logits = layers.fully_connected(lz_x, dim_z, activation_fn=None,
                        weights_initializer=tf.contrib.layers.xavier_initializer())
            z = zs.OnehotCategorical('z', logits=z_logits, dtype=tf.float32,
                        n_samples=n_particles,
                        #group_event_ndims=1
                        )
        return variational, z_logits

    # def baseline_net(x):
    #     with tf.variable_scope('baseline_net'):
    #         lc_x = layers.fully_connected(tf.to_float(x), 100,
    #                 weights_initializer=tf.contrib.layers.xavier_initializer())
    #         lc_x = layers.fully_connected(lc_x, 1, activation_fn=None,
    #                 weights_initializer=tf.contrib.layers.xavier_initializer())
    #         lc_x = tf.squeeze(lc_x, -1)
    #         return lc_x

    x_ph = tf.placeholder(tf.float32, shape=[None, dim_x], name='x')
    # is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    n = tf.shape(x_ph)[0]


    def log_joint(observed):
        model, _, _ = vae(observed, n, dim_x, dim_z, n_particles)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    variational, q_cluster = q_net(x_ph, dim_z, n_particles)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)

    # cx = tf.expand_dims(baseline_net(x_ph), 0)
    x_obs = tf.tile(tf.expand_dims(x_ph, 0), [n_particles, 1, 1])
    surrogate_cost, lower_bound = zs.nvil(log_joint,
                            observed={'x': x_obs},
                            latent={'z': [qz_samples, log_qz]},
                            #baseline=cx,
                            axis=0)
    # print('-'*10)
    mean_lower_bound = tf.reduce_mean(lower_bound)
    with tf.name_scope('model_loss'):
        loss = tf.reduce_mean(surrogate_cost)

    train_vars = tf.trainable_variables()
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, epsilon=1e-4)
    grads_and_vars = optimizer.compute_gradients(loss)
    infer = optimizer.apply_gradients(grads_and_vars)

    # Generate x samples
    _, x_gen, z_gen = vae({}, 1, dim_x, dim_z, n_particles=n_gen)
    x_gen = tf.squeeze(x_gen, 1)
    z_gen = tf.squeeze(z_gen, 1)

    # tensorboard summary ---------------------------------------------------
    lb_summ = tf.summary.scalar("lower_bound", mean_lower_bound)
    loss_summ = tf.summary.scalar("loss", loss)
    lr_ = tf.reduce_mean(learning_rate_ph, name='lr_')
    lr_summ = tf.summary.scalar("learning_rate", lr_)

    for var in train_vars:
        tf.summary.histogram(var.name, var)

    for i in train_vars:
        print(i.name, i.get_shape())
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    plot_buf_ph = tf.placeholder(tf.string)
    image = tf.image.decode_png(plot_buf_ph, channels=4)
    image = tf.expand_dims(image, 0)  # make it batched
    plot_image_summary = tf.summary.image('clusters', image, max_outputs=10)

    # initialization for train data generation----------------------------
    global pai_init, mu_init, log_sigma_init
    pai_init = np.random.uniform(0, 2, (dim_z)).astype(np.float32) #np.ones(K)
    # mu_init = np.array([[3, 5],[-3, -4], [-5, 5]], dtype=np.float32)
    # sigma_init = np.array([[0, 0],[0, 0], [0, 0]], dtype=np.float32)
    mu_init = np.random.uniform(0, 1, (dim_z, dim_x)).astype(np.float32)
    log_sigma_init = np.random.normal(-3, 0.5, (dim_z, dim_x)).astype(np.float32)
    print('pai init for generating train data: ', pai_init)
    print('mu init for generating train data: \n', mu_init)
    print('sigma init for generating train data: \n', np.exp(log_sigma_init))
    with tf.variable_scope('decoder', reuse=True):
        pai = tf.get_variable('pai')
        mu = tf.get_variable('mu')
        log_sigma = tf.get_variable('log_sigma')
    pai_assign = pai.assign(pai_init)
    mu_assign = mu.assign(mu_init)
    log_sigma_assign = log_sigma.assign(log_sigma_init)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # generate train data -------------------------------------------------
        train_filename = '../data/N_{}_K_{}_2d_gaussian_gzip.hkl'.format(N, K)
        global x_train, z_train
        # if not os.path.exists(train_filename):
        sess.run([pai_assign, mu_assign, log_sigma_assign])
        x_train, z_train = sess.run([x_gen, z_gen])
        print('x_train shape:', x_train.shape)
        hickle.dump(x_train, train_filename, mode='w', compression='gzip')
        # x_train_mean = np.mean(x_train, 0)
        # x_train_std = np.std(x_train, 0)
        # x_train_normed = (x_train - x_train_mean)/x_train_std
        x_train_normed = x_train # no normalization
        x_train_normed_no_shuffle = x_train_normed
        # print(x_train_mean)
        # print(x_train_std)
        # x_train_min = x_train.min()
        # x_train_max = x_train.max()
        # x_train_normed = (x_train - x_train_min)/(x_train_max - x_train_min)
        print(x_train_normed.max(), x_train_normed.min())
        # else: # load existing file
            # print('load existing file: {}'.format(train_filename))
            # x_train = hickle.load(train_filename)
            # print(x_train.shape)
        # plt.plot(x_train[:,0], x_train[:,1], '+')
        # plt.show()
        log_dir = './log/N_{}_K_{}_2d_gaussian/'.format(N, K)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir,
                                graph=tf.get_default_graph())

        global x_gen_list, clusters_list # , q_res_list
        global mu_res, log_sigma_res, pai_res
        x_gen_list = []
        # q_res_list = []
        clusters_list = []

        print('training...') # ---------------------------------------------------
        sess.run(tf.global_variables_initializer())
        pai_res_0, mu_res_0, log_sigma_res_0 = sess.run([pai, mu, log_sigma])
        print('random initializing...')
        print('pai_res_0: ', pai_res_0)
        print('mu_res_0: \n', mu_res_0)
        print('sigma_res_0: \n', np.exp(log_sigma_res_0))
        global_step = 0
        for epoch in range(1, epoches + 1):
            if epoch % anneal_lr_freq == 0:
                learning_rate *= anneal_lr_rate
            np.random.shuffle(x_train_normed) # shuffle training data
            lbs = []
            for t in range(iters_per_batch):
                global_step += 1
                x_batch = x_train_normed[t * batch_size : (t + 1) * batch_size] # get batched data
                # print('x_batch shape:',  x_batch.shape)
                _, lb, q_cluster_res = sess.run([infer, mean_lower_bound,
                                q_cluster],
                                feed_dict={x_ph: x_batch,
                                           learning_rate_ph: learning_rate})
                lbs.append(lb)
                # print(grad_var_res[-3:])

            if epoch % save_freq == 0: # results -------------------------------------------------
                # x_train_gen = sess.run(x_gen)
                # print(x_train == x_train)
                # x_gen_list.append(x_train_gen)
                print('Epoch {}: average Lower bound = {}'.format(
                    epoch, np.mean(lbs)))
                q_cluster_res_save, lb_summ_res, merge_all = sess.run([q_cluster, lb_summ, merged_summary_op],
                                feed_dict={x_ph: x_train_normed_no_shuffle, learning_rate_ph: learning_rate})
                # summary_writer.add_summary(lb_summ_res, global_step=epoch)
                summary_writer.add_summary(merge_all, global_step=epoch)
                clusters = np.argmax(q_cluster_res_save, axis=1)
                # print(clusters.shape)
                # print(qz_samples_res_save.shape)
                clusters_list.append(clusters)

                if epoch % plot_freq == 0: # plot scatter ------------------------------------
                    # plot_buf = get_plot_buf(x_train_normed_no_shuffle, clusters)
                    pai_res, mu_res, log_sigma_res = sess.run([pai, mu, log_sigma])
                    plot_buf = get_plot_buf(x_train, clusters, mu_res, log_sigma_res, mu_init, log_sigma_init)
                    plot_image_summary_ = sess.run(
                            plot_image_summary,
                            feed_dict={plot_buf_ph: plot_buf.getvalue()})
                    summary_writer.add_summary(plot_image_summary_, global_step=epoch)
                # q_res_list.append(qz_samples_res)
                # print(qz_samples_res)
                # name = "results/vae/vae.epoch.{}.png".format(epoch)

        pai_res, mu_res, log_sigma_res = sess.run([pai, mu, log_sigma])
        print("Random Seed: ", seed)
        print('pai init for generating train data: ', pai_init)
        print('mu init for generating train data: \n', mu_init)
        print('sigma init for generating train data: \n', np.exp(log_sigma_init))
        print('*'*10)
        print('pai_res: ', pai_res)
        print('mu_res: \n', mu_res)
        print('sigma_res: \n', np.exp(log_sigma_res))



def get_plot_buf(x, clusters, mu, logstd, true_mu, true_logstd):
    N = x.shape[0]
    K = mu.shape[0]
    fig = plt.figure()
    # print(clusters.shape)
    # print(x.shape)
    ax = fig.add_subplot(111, aspect='auto')
    plt.scatter(x[:, 0], x[:, 1], c=clusters, s=50)
    # print(mu, logstd)
    ells = [Ellipse(xy=mean_, width=6*np.exp(logstd_[0]), height=6*np.exp(logstd_[1]),
                angle=0, facecolor='none', zorder=10, edgecolor='g', label='predict' if i==0 else None)
            for i, (mean_, logstd_) in enumerate(zip(mu, logstd))]
    true_ells = [Ellipse(xy=mean_, width=6*np.exp(logstd_[0]), height=6*np.exp(logstd_[1]),
                angle=0, facecolor='none', zorder=10, edgecolor='r', label='true' if i==0 else None)
            for i,(mean_, logstd_) in enumerate(zip(true_mu, true_logstd))]
    # print(ells[0])
    [ax.add_patch(ell) for ell in ells]
    [ax.add_patch(true_ell) for true_ell in true_ells]
    ax.legend(loc='best')
    ax.set_title('N={},K={}'.format(N, K))
    plt.autoscale(True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

if __name__ == "__main__":
    N = 100
    K = 3
    main(N=N, K=K)
    clusters = clusters_list[-1]
    # print(clusters)
    # for clusters in clusters_list:
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='auto')
    plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, s=50)

    ells = [Ellipse(xy=mean_, width=6*np.exp(logstd_[0]), height=6*np.exp(logstd_[1]),
                angle=0, facecolor='none', zorder=10, edgecolor='g', label='predict' if i==0 else None)
            for i,(mean_, logstd_) in enumerate(zip(mu_res, log_sigma_res))]
    true_ells = [Ellipse(xy=mean_, width=6*np.exp(logstd_[0]), height=6*np.exp(logstd_[1]),
                angle=0, facecolor='none', zorder=10, edgecolor='r', label='true' if i==0 else None)
            for i,(mean_, logstd_) in enumerate(zip(mu_init, log_sigma_init))]

    [ax.add_patch(ell) for ell in ells]
    [ax.add_patch(true_ell) for true_ell in true_ells]
    ax.legend(loc='best')
    ax.set_title('N={},K={}'.format(N, K))
    plt.autoscale(True)
    fig.savefig('./results/result_N_{}_K_{}.png'.format(N, K), dpi=fig.dpi)

    # plt.show()




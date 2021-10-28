from cvm_net import cvm_net_I, cvm_net_II
from input_data import InputData

import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
import cv2
from operator import itemgetter

import time


# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
network_type = 'CVM-NET-I'

batch_size = 1
is_training = True
loss_weight = 10.0
number_of_epoch = 5

learning_rate_val = 1e-5
keep_prob_val = 0.8
# -------------------------------------------------------- #



def validate(grd_descriptor, sat_descriptor):
    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        print(prediction)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy


def compute_loss(sat_global, grd_global, batch_hard_count=0):
    '''
    Compute the weighted soft-margin triplet loss
    :param sat_global: the satellite image global descriptor
    :param grd_global: the ground image global descriptor
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    :return: the loss
    '''
    with tf.name_scope('weighted_soft_margin_triplet_loss'):
        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)
        pos_dist = tf.diag_part(dist_array)
        if batch_hard_count == 0:
            pair_n = batch_size * (batch_size - 1.0)

            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))
            top_k_g2s, _ = tf.nn.top_k(tf.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = tf.reduce_mean(top_k_g2s)

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            triplet_dist_s2g = tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))
            top_k_s2g, _ = tf.nn.top_k(triplet_dist_s2g, batch_hard_count)
            loss_s2g = tf.reduce_mean(top_k_s2g)

            loss = (loss_g2s + loss_s2g) / 2.0

    return loss


def train(start_epoch=1):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 1.
    '''

    # import data
    input_data = InputData()


    # define placeholders
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
    grd_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='grd_x')
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)


    # build model
    if network_type == 'CVM-NET-I':
        sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, is_training)
    elif network_type == 'CVM-NET-II':
        sat_global, grd_global = cvm_net_II(sat_x, grd_x, keep_prob, is_training)
    else:
        print ('CONFIG ERROR: wrong network type, only CVM-NET-I and CVM-NET-II are valid')


    # define loss
    loss = compute_loss(sat_global, grd_global, 0)


    # set training
    global_step = tf.Variable(0, trainable=False)
    with tf.device('/gpu:0'):
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)


    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    #config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join("../Model/CVM-NET-I/0/", "model.ckpt"))

        print('load model...')
        load_model_path = '../Model/' + network_type + '/' + str(start_epoch - 1) + '/model.ckpt'
        saver.restore(sess, load_model_path)
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        # Train
        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            while True:
                # train
                batch_sat, batch_grd = input_data.next_pair_batch(batch_size)
                if batch_sat is None:
                    break

                global_step_val = tf.train.global_step(sess, global_step)

                feed_dict = {sat_x: batch_sat, grd_x: batch_grd,
                             learning_rate: learning_rate_val, keep_prob: keep_prob_val}
                if iter % 20 == 0:
                    _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
                    print('global %d, epoch %d, iter %d: loss : %.4f' %
                          (global_step_val, epoch, iter, loss_val))
                else:
                    sess.run(train_step, feed_dict=feed_dict)

                iter += 1
            
            # ---------------------- validation ----------------------
            print('validate...')
            print('   compute global descriptors')
            input_data.reset_scan()
            sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
            grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
            val_i = 0
            while True:
                print('      progress %d' % val_i)
                batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
                if batch_sat is None:
                    break
                feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
                sat_global_val, grd_global_val = \
                    sess.run([sat_global, grd_global], feed_dict=feed_dict)

                sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
                grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
                val_i += sat_global_val.shape[0]

            print('   compute accuracy')
            val_accuracy = validate(grd_global_descriptor, sat_global_descriptor)
            with open('../Result/' + str(network_type) + '_accuracy.txt', 'a') as file:
                file.write(str(epoch) + ' ' + str(iter) + ' : ' + str(val_accuracy) + '\n')
            print('   %d: accuracy = %.1f%%' % (epoch, val_accuracy*100.0))

            model_dir = '../Model/' + network_type + '/' + str(epoch) + '/'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_path = saver.save(sess, model_dir + 'model.ckpt')
            print("Model saved in file: %s" % save_path)
            # ---------------------------------------------------------

def test():
    input_data = InputData()
    start_time = time.perf_counter()
    print("test_model")
    tf.compat.v1.disable_eager_execution()
    sat_x = tf.compat.v1.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
    grd_x = tf.compat.v1.placeholder(tf.float32, [None, 512, 512, 3], name='grd_x')
    keep_prob = tf.compat.v1.placeholder(tf.float32)  # dropout
    learning_rate = tf.compat.v1.placeholder(tf.float32)

    # just BUILDING MODEL, satellite and ground image will be given later
    # tf.compat.v1.disable_eager_execution()
    #global sat_global, grd_global
    if network_type == 'CVM-NET-I':
        sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, is_training)
    if network_type == 'CVM-NET-II':
        sat_global, grd_global = cvm_net_II(sat_x, grd_x, keep_prob, is_training)

    # run model
    print('run model...')
    config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        print('load model...')
        # os.chdir('./Model/')
        # cwd = os.getcwd()
        load_model_path = 'E:/postgraduate/crossview_localisation/src/Model' + '/' + network_type + '/' + network_type + '_model'
        print(load_model_path)
        saver = tf.compat.v1.train.import_meta_graph(load_model_path + "/model.ckpt.meta")
        load_model_path += '/model.ckpt'
        saver.restore(sess, load_model_path)
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')
        model_vars = tf.compat.v1.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        print('testing...')
        print('   compute global descriptors')
        input_data.reset_scan()
        sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
        grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
        val_i = 0
        while True:
            print('progress %d' % val_i)
            batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break  # break when all the batches are evaluated
            feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
            # works fine until here
            # forward pass
            # tf.compat.v1.reset_default_graph()
            # print(batch_sat)
            # print("------------------------------------------")
            # print(batch_grd)
            # print(feed_dict)
            # graph = tf.compat.v1.get_default_graph()
            # with sess.as_default():
            #     with graph.as_default():
            # sat_global, grd_global = cvm_net_II(batch_sat, batch_grd, 0.8, True)
            sat_global_val, grd_global_val = \
                sess.run([sat_global, grd_global], feed_dict=feed_dict)
            #print('sat_global_val ', sat_global_val)
            #print(sat_global_val)
            sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
            grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
            val_i += sat_global_val.shape[0]

        print('compute accuracy')
        val_accuracy = validate(grd_global_descriptor, sat_global_descriptor)
        print(' accuracy = %.1f%%' % (val_accuracy * 100.0))
        end_time = time.perf_counter()
        print("final is in : %s Seconds " % (end_time-start_time))


def test_query(query_image):
    input_data = InputData()
    print("test_model")

    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
    grd_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='grd_x')
    keep_prob = tf.placeholder(tf.float32)  # dropout
    learning_rate = tf.placeholder(tf.float32)

    # just BUILDING MODEL, satellite and ground image will be given later
    if network_type == 'CVM-NET-I':
        sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, is_training)
    if network_type == 'CVM-NET-II':
        sat_global, grd_global = cvm_net_II(sat_x, grd_x, keep_prob, is_training)

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        print('load model...')
        os.chdir('../Model/')
        cwd = os.getcwd()
        os.chdir('../CVM-Net')

        load_model_path = cwd + '/' + network_type + '/' + network_type + '_model'
        saver = tf.train.import_meta_graph(load_model_path + "/model.ckpt.meta")
        load_model_path += '/model.ckpt'
        saver.restore(sess, load_model_path)
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')
        # print model
        # import tensorflow.contrib.slim as slim
        # model_vars = tf.trainable_variables()
        # slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        print('testing...')

        print('   compute global descriptors')
        input_data.reset_scan()
        sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
        grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
        val_i = 0
        # this is just pre processing step
        while True:
            print('progress %d' % val_i)
            batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break  # break when all the batches are evaluated
            feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
            # works fine until here
            # forward pass
            sat_global_val, grd_global_val = \
                sess.run([sat_global, grd_global], feed_dict=feed_dict)  # feed in the batch input here

            sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
            grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
            val_i += sat_global_val.shape[0]

        print('compute accuracy')
        # This is where you compare the each individual ground descriptor in the
        # dataset
        query_id = int((query_image.split('/')[-1]).split('.')[0])
        # dist array is 123 x 1
        dist_array = 2 - 2 * np.matmul(sat_global_descriptor, np.transpose(grd_global_descriptor[query_id, :]))
        print(dist_array)
        k = 10
        # it is better you calculate top 10 per cent images for a smaller dataset like ours
        topk_percent = int(dist_array.shape[0] * (k / 100.0)) + 1

        # sorted(range(len(my_list)),key=my_list.__getitem__)
        indices, list_sorted = zip(*sorted(enumerate(dist_array), key=itemgetter(1)))
        list_sorted = list(list_sorted)
        indices = list(indices)
        k_indices = indices[:k]
        k_vals = list_sorted[:k]
        # these are the top k image indices
        img_root = os.getcwd()
        os.chdir('../Data/CVUSA/')
        img_root = os.getcwd()
        os.chdir('../../CVM-Net/')

        for i in range(len(k_indices)):
            img_index = k_indices[i]
            print(img_index)
            # img_path = img_root + '/satellite/' + str(img_index) + '.jpeg'
            img_path = img_root + '/streetview/' + '1.jpg'
            print('img_path ',img_path)
            c_img = cv2.imread(img_path)
            save_path = os.getcwd() + '/top_k/' + str(query_id) + '/' + str(img_index) + '.jpg'
            # print('save_path ', save_path)
            # cv2.imwrite(save_path, c_img)
            # cv2.imshow('top k', c_img)
            # cv2.waitKey(0)
        print('\n')
        # these are the cosine similarity values
        for i in range(len(k_indices)):
            print(k_vals[i])
        # See what is the ranking of the actual ground truth in top k predictions.
        try:
            gt_pos = k_indices.index(query_id)
        except:
            gt_pos = -1
        print('gt_pos ', gt_pos)
if __name__ == '__main__':
    # # cwd=os.getcwd()
    # # print(cwd)
    # os.chdir('../Data/CVUSA')
    # img_root = os.getcwd()
    # print(img_root)
    # # going back to where this file is located
    # os.chdir('../../CVM-Net')
    # img_index = 0
    # query_image = img_root + '/streetview/' + str(img_index) + '.jpg'
    # test_query(query_image)
    test()
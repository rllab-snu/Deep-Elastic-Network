import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import utils_resnext_ours as model
import TINY_input as tiny
import CIFAR_input as cifar
import STL_input as stl

# PARSER
parser = argparse.ArgumentParser(description='MULTI_INPUT_v2')
# environment
parser.add_argument('--GPUS', default=[0, 1], type=int, metavar='N', nargs='+', help='GPU id to use')
# model structure for STEMNet
parser.add_argument('--nlevels', default=4, type=int, metavar='N', help='Number of compression levels in each block')
parser.add_argument('--nsamples', default=1, type=int, metavar='N', help='Number of samples to explore')
parser.add_argument('--rho', default=10, type=float, metavar='N', help='Rho')
parser.add_argument('--is_train', default=False, type=bool, metavar='N', help='Need to train a base-network')
parser.add_argument('--model_path', default='', type=str, help='path to saved model')
# model structure for ResNext
parser.add_argument('--num_blocks', default=3, type=int, metavar='N', help='Number of blocks in each stage (default: 3)')
parser.add_argument('--cardinality', default=4, type=int, metavar='N', help='Number of groups (default: 4)')
parser.add_argument('--unit_depth', default=16, type=int, metavar='N', help='Depth of each group (default: 64)')
parser.add_argument('--num_batches', default=782 * 2, type=int, metavar='N', help='The number of batches in an epoch (default: 391)')
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='Batch size (default: 128)''')
# training options
parser.add_argument('--max_step', default=156400, type=int, metavar='N', help='Total training steps''')
parser.add_argument('--decay_step0', default=78200, type=int, metavar='N', help='Step to decay the learning rate')
parser.add_argument('--decay_step1', default=117300, type=int, metavar='N', help='Step to decay the learning rate')
parser.add_argument('--init_lr', default=0.1, type=float, metavar='N', help='Initial learning rate (default: 0.1)')
args = parser.parse_args()

# Settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '%d, %d' % (args.GPUS[0], args.GPUS[1])

FILENAME = 'saveText/%s_multi_input.txt' % (datetime.date.today())

sel_blocks = 4 * args.num_blocks
total_blocks = 4 * sel_blocks


class Train(object):
    def __init__(self):
        self.placeholders()

    def placeholders(self):
        self.img32 = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 32, 32, 3])
        self.img64 = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 64, 64, 3])
        self.img96 = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 96, 96, 3])
        self.lab = tf.placeholder(dtype=tf.int32, shape=[args.batch_size])
        self.compress = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, sel_blocks, 1, 1, 1])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])
        self.epsilon = tf.placeholder(dtype=tf.float32, shape=[])
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[])

    def build_inference32(self, img_list, lab_list, comp_list, reuse):
        ls_list = []
        ac_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0 or reuse:
                        reuse = True
                    else:
                        reuse = False
                    logits = model.inference32('cifar', img_list[i], args.num_blocks, reuse, reuse, comp_list[i],
                                               args.unit_depth, args.cardinality, self.is_training)
                    ls, ac = self.performance(logits, lab_list[i])
                    ac_list.append(ac)
                    ls_list.append(ls)

        return ac_list, ls_list

    def build_inference64(self, img_list, lab_list, comp_list, reuse):
        ls_list = []
        ac_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0 or reuse:
                        reuse = True
                    else:
                        reuse = False
                    logits = model.inference64('tiny', img_list[i], args.num_blocks, True, reuse, comp_list[i],
                                               args.unit_depth, args.cardinality, self.is_training)
                    ls, ac = self.performance(logits, lab_list[i])
                    ac_list.append(ac)
                    ls_list.append(ls)

        return ac_list, ls_list

    def build_inference96(self, img_list, lab_list, comp_list, reuse):
        ls_list = []
        ac_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0 or reuse:
                        reuse = True
                    else:
                        reuse = False
                    logits = model.inference96('stl', img_list[i], args.num_blocks, True, reuse, comp_list[i],
                                               args.unit_depth, args.cardinality, self.is_training)
                    ls, ac = self.performance(logits, lab_list[i])
                    ac_list.append(ac)
                    ls_list.append(ls)

        return ac_list, ls_list

    def build_policy32(self, img_list, lab_list, reuse):
        pl_list = []
        ls_list = []
        ac_list = []
        cp_list = []
        batch_size = int(args.batch_size / 2)
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0 or reuse:
                        reuse = True
                    else:
                        reuse = False
                    probs = model.policy_inference32('cifar', img_list[i], args.num_blocks, reuse, reuse,
                                                     args.unit_depth, args.cardinality, self.is_training)
                    probs_ = tf.reshape(probs, [batch_size, sel_blocks, args.nlevels])
                    policy = tf.nn.softmax(probs_)

                    pred_policy = tf.cast(tf.reshape(tf.argmax(policy, 2), [batch_size, sel_blocks, 1, 1, 1]),
                                          tf.float32)
                    logits_policy = model.inference32('cifar', img_list[i], args.num_blocks, True, True, pred_policy,
                                                      args.unit_depth, args.cardinality, self.is_training)
                    ploss_list = []
                    for sample in range(args.nsamples):
                        esl = tf.random_uniform([batch_size, 1, 1], minval=0, maxval=1, dtype=tf.float32)
                        esl = tf.where(esl > self.epsilon,
                                       tf.ones([batch_size, 1, 1], tf.float32),
                                       tf.zeros([batch_size, 1, 1], tf.float32))
                        rand_policy = tf.random_uniform([batch_size, sel_blocks, args.nlevels], minval=0, maxval=1,
                                                        dtype=tf.float32)
                        rand_policy = tf.nn.softmax(rand_policy)
                        rpolicy = esl * policy + (1. - esl) * rand_policy
                        pred_rand = self.sample_pred(rpolicy)
                        pred_rand_ = tf.cast(tf.reshape(pred_rand, [batch_size, sel_blocks]), tf.int32)
                        logits_rand = model.inference32('cifar', img_list[i], args.num_blocks, True, True, pred_rand,
                                                        args.unit_depth, args.cardinality, self.is_training)
                        reward = self.get_reward(logits_rand, lab_list[i], pred_rand, args.rho)
                        prob = tf.reduce_sum(tf.cast(tf.one_hot(pred_rand_, 4), tf.float32) * policy, 2)
                        policy_loss = - tf.log(tf.reduce_prod(prob, 1) + 1e-15) * reward
                        ploss_list.append(tf.reduce_mean(policy_loss))
                    ploss = tf.reduce_mean(ploss_list)
                    ls, ac = self.performance(logits_policy, lab_list[i])
                    pl_list.append(ploss)
                    ls_list.append(ls)
                    ac_list.append(ac)
                    cp_list.append(self.density(pred_policy))

        return pl_list, ac_list, ls_list, cp_list

    def build_policy64(self, img_list, lab_list, reuse):
        pl_list = []
        ls_list = []
        ac_list = []
        cp_list = []
        batch_size = int(args.batch_size / 2)
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0 or reuse:
                        reuse = True
                    else:
                        reuse = False
                    probs = model.policy_inference64('tiny', img_list[i], args.num_blocks, True, reuse,
                                                     args.unit_depth, args.cardinality, self.is_training)
                    probs_ = tf.reshape(probs, [batch_size, sel_blocks, args.nlevels])
                    policy = tf.nn.softmax(probs_)

                    pred_policy = tf.cast(tf.reshape(tf.argmax(policy, 2), [batch_size, sel_blocks, 1, 1, 1]),
                                          tf.float32)
                    logits_policy = model.inference64('tiny', img_list[i], args.num_blocks, True, True, pred_policy,
                                                      args.unit_depth, args.cardinality, self.is_training)
                    ploss_list = []
                    for sample in range(args.nsamples):
                        esl = tf.random_uniform([batch_size, 1, 1], minval=0, maxval=1, dtype=tf.float32)
                        esl = tf.where(esl > self.epsilon,
                                       tf.ones([batch_size, 1, 1], tf.float32),
                                       tf.zeros([batch_size, 1, 1], tf.float32))
                        rand_policy = tf.random_uniform([batch_size, sel_blocks, args.nlevels], minval=0, maxval=1,
                                                        dtype=tf.float32)
                        rand_policy = tf.nn.softmax(rand_policy)
                        rpolicy = esl * policy + (1. - esl) * rand_policy
                        pred_rand = self.sample_pred(rpolicy)
                        pred_rand_ = tf.cast(tf.reshape(pred_rand, [batch_size, sel_blocks]), tf.int32)
                        logits_rand = model.inference64('tiny', img_list[i], args.num_blocks, True, True, pred_rand,
                                                        args.unit_depth, args.cardinality, self.is_training)
                        reward = self.get_reward(logits_rand, lab_list[i], pred_rand, args.rho)
                        prob = tf.reduce_sum(tf.cast(tf.one_hot(pred_rand_, 4), tf.float32) * policy, 2)
                        policy_loss = - tf.log(tf.reduce_prod(prob, 1) + 1e-15) * reward
                        ploss_list.append(tf.reduce_mean(policy_loss))
                    ploss = tf.reduce_mean(ploss_list)
                    ls, ac = self.performance(logits_policy, lab_list[i])
                    pl_list.append(ploss)
                    ls_list.append(ls)
                    ac_list.append(ac)
                    cp_list.append(self.density(pred_policy))

        return pl_list, ac_list, ls_list, cp_list

    def build_policy96(self, img_list, lab_list, reuse):
        pl_list = []
        ls_list = []
        ac_list = []
        cp_list = []
        batch_size = int(args.batch_size / 2)
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0 or reuse:
                        reuse = True
                    else:
                        reuse = False
                    probs = model.policy_inference96('stl', img_list[i], args.num_blocks, True, reuse,
                                                     args.unit_depth, args.cardinality, self.is_training)
                    probs_ = tf.reshape(probs, [batch_size, sel_blocks, args.nlevels])
                    policy = tf.nn.softmax(probs_)

                    pred_policy = tf.cast(tf.reshape(tf.argmax(policy, 2), [batch_size, sel_blocks, 1, 1, 1]),
                                          tf.float32)
                    logits_policy = model.inference96('stl', img_list[i], args.num_blocks, True, True, pred_policy,
                                                      args.unit_depth, args.cardinality, self.is_training)
                    ploss_list = []
                    for sample in range(args.nsamples):
                        esl = tf.random_uniform([batch_size, 1, 1], minval=0, maxval=1, dtype=tf.float32)
                        esl = tf.where(esl > self.epsilon,
                                       tf.ones([batch_size, 1, 1], tf.float32),
                                       tf.zeros([batch_size, 1, 1], tf.float32))
                        rand_policy = tf.random_uniform([batch_size, sel_blocks, args.nlevels], minval=0, maxval=1,
                                                        dtype=tf.float32)
                        rand_policy = tf.nn.softmax(rand_policy)
                        rpolicy = esl * policy + (1. - esl) * rand_policy
                        pred_rand = self.sample_pred(rpolicy)
                        pred_rand_ = tf.cast(tf.reshape(pred_rand, [batch_size, sel_blocks]), tf.int32)
                        logits_rand = model.inference96('stl', img_list[i], args.num_blocks, True, True, pred_rand,
                                                        args.unit_depth, args.cardinality, self.is_training)
                        reward = self.get_reward(logits_rand, lab_list[i], pred_rand, args.rho)
                        prob = tf.reduce_sum(tf.cast(tf.one_hot(pred_rand_, 4), tf.float32) * policy, 2)
                        policy_loss = - tf.log(tf.reduce_prod(prob, 1) + 1e-15) * reward
                        ploss_list.append(tf.reduce_mean(policy_loss))
                    ploss = tf.reduce_mean(ploss_list)
                    ls, ac = self.performance(logits_policy, lab_list[i])
                    pl_list.append(ploss)
                    ls_list.append(ls)
                    ac_list.append(ac)
                    cp_list.append(self.density(pred_policy))

        return pl_list, ac_list, ls_list, cp_list

    def build_graph(self):
        global_step = tf.Variable(0, trainable=False)
        img_list_32 = []
        img_list_64 = []
        img_list_96 = []
        lab_list = []
        comp_list = []

        batch_size = int(args.batch_size / 2)
        img_list_32.append(self.img32[:batch_size, ...])
        img_list_32.append(self.img32[batch_size:, ...])
        img_list_64.append(self.img64[:batch_size, ...])
        img_list_64.append(self.img64[batch_size:, ...])
        img_list_96.append(self.img96[:batch_size, ...])
        img_list_96.append(self.img96[batch_size:, ...])
        lab_list.append(self.lab[:batch_size])
        lab_list.append(self.lab[batch_size:])
        comp_list.append(self.compress[:batch_size, ...])
        comp_list.append(self.compress[batch_size:, ...])

        opt_e = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=True)
        opt_s = tf.train.AdamOptimizer(self.lr)

        ac32, ls32 = self.build_inference32(img_list_32, lab_list, comp_list, False)

        t_vars = tf.trainable_variables()
        t_vars = [var for var in t_vars if 'resnext' in var.name]
        cifar_vars = [var for var in t_vars if 'stl' not in var.name]
        cifar_vars = [var for var in cifar_vars if 'tiny' not in var.name]
        grad_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    grads = opt_e.compute_gradients(tf.add_n([ls32[i]] + regu_losses), cifar_vars)
                    grad_list.append(grads)
        grads = self.average_gradients(grad_list)
        apply_grad_op = opt_e.apply_gradients(grads, global_step=global_step)
        self.op_est32 = tf.group(*([apply_grad_op]))
        self.est_ac32 = tf.reduce_mean(ac32)
        self.est_ls32 = tf.reduce_mean(ls32)

        ac64, ls64 = self.build_inference64(img_list_64, lab_list, comp_list, False)

        t_vars = tf.trainable_variables()
        t_vars = [var for var in t_vars if 'resnext' in var.name]
        tiny_vars = [var for var in t_vars if 'stl' not in var.name]
        tiny_vars = [var for var in tiny_vars if 'cifar' not in var.name]
        grad_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    grads = opt_e.compute_gradients(tf.add_n([ls64[i]] + regu_losses), tiny_vars)
                    grad_list.append(grads)
        grads = self.average_gradients(grad_list)
        apply_grad_op = opt_e.apply_gradients(grads, global_step=global_step)
        self.op_est64 = tf.group(*([apply_grad_op]))
        self.est_ac64 = tf.reduce_mean(ac64)
        self.est_ls64 = tf.reduce_mean(ls64)

        ac96, ls96 = self.build_inference96(img_list_96, lab_list, comp_list, False)

        t_vars = tf.trainable_variables()
        t_vars = [var for var in t_vars if 'resnext' in var.name]
        stl_vars = [var for var in t_vars if 'tiny' not in var.name]
        stl_vars = [var for var in stl_vars if 'cifar' not in var.name]
        grad_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    grads = opt_e.compute_gradients(tf.add_n([ls96[i]] + regu_losses), stl_vars)
                    grad_list.append(grads)
        grads = self.average_gradients(grad_list)
        apply_grad_op = opt_e.apply_gradients(grads, global_step=global_step)
        self.op_est96 = tf.group(*([apply_grad_op]))
        self.est_ac96 = tf.reduce_mean(ac96)
        self.est_ls96 = tf.reduce_mean(ls96)

        pl32, ac32, ls32, cp32 = self.build_policy32(img_list_32, lab_list, False)

        p_vars = tf.trainable_variables()
        p_vars = [var for var in p_vars if 'policy' in var.name]
        cifar_pcy = [var for var in p_vars if 'stl' not in var.name]
        cifar_pcy = [var for var in cifar_pcy if 'tiny' not in var.name]
        grad_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    grads = opt_s.compute_gradients(tf.add_n([pl32[i]]), cifar_pcy)
                    grad_list.append(grads)
        grads = self.average_gradients(grad_list)
        apply_grad_op = opt_s.apply_gradients(grads, global_step=global_step)
        self.op_sel32 = tf.group(*([apply_grad_op]))

        grad_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    grads = opt_e.compute_gradients(tf.add_n([ls32[i]] + regu_losses), cifar_vars)
                    grad_list.append(grads)
        grads = self.average_gradients(grad_list)
        apply_grad_op = opt_e.apply_gradients(grads, global_step=global_step)
        self.op_com32 = tf.group(*([apply_grad_op]))

        self.com_ac32 = tf.reduce_mean(ac32)
        self.com_cp32 = tf.reduce_mean(cp32)

        pl64, ac64, ls64, cp64 = self.build_policy64(img_list_64, lab_list, False)

        p_vars = tf.trainable_variables()
        p_vars = [var for var in p_vars if 'policy' in var.name]
        tiny_pcy = [var for var in p_vars if 'stl' not in var.name]
        tiny_pcy = [var for var in tiny_pcy if 'cifar' not in var.name]
        grad_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    grads = opt_s.compute_gradients(tf.add_n([pl64[i]]), tiny_pcy)
                    grad_list.append(grads)
        grads = self.average_gradients(grad_list)
        apply_grad_op = opt_s.apply_gradients(grads, global_step=global_step)
        self.op_sel64 = tf.group(*([apply_grad_op]))

        grad_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    grads = opt_e.compute_gradients(tf.add_n([ls64[i]] + regu_losses), tiny_vars)
                    grad_list.append(grads)
        grads = self.average_gradients(grad_list)
        apply_grad_op = opt_e.apply_gradients(grads, global_step=global_step)
        self.op_com64 = tf.group(*([apply_grad_op]))

        self.com_ac64 = tf.reduce_mean(ac64)
        self.com_cp64 = tf.reduce_mean(cp64)

        pl96, ac96, ls96, cp96 = self.build_policy96(img_list_96, lab_list, False)

        p_vars = tf.trainable_variables()
        p_vars = [var for var in p_vars if 'policy' in var.name]
        stl_pcy = [var for var in p_vars if 'cifar' not in var.name]
        stl_pcy = [var for var in stl_pcy if 'tiny' not in var.name]
        grad_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    grads = opt_s.compute_gradients(tf.add_n([pl96[i]]), stl_pcy)
                    grad_list.append(grads)
        grads = self.average_gradients(grad_list)
        apply_grad_op = opt_s.apply_gradients(grads, global_step=global_step)
        self.op_sel96 = tf.group(*([apply_grad_op]))

        grad_list = []
        for i in range(2):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    grads = opt_e.compute_gradients(tf.add_n([ls96[i]] + regu_losses), stl_vars)
                    grad_list.append(grads)
        grads = self.average_gradients(grad_list)
        apply_grad_op = opt_e.apply_gradients(grads, global_step=global_step)
        self.op_com96 = tf.group(*([apply_grad_op]))

        self.com_ac96 = tf.reduce_mean(ac96)
        self.com_cp96 = tf.reduce_mean(cp96)

    def train(self):
        print('dataset ready ...')
        cifar_t_img, cifar_t_lab = cifar.read_train_data()
        cifar_v_img, cifar_v_lab = cifar.read_vali_data()
        tiny_t_img, tiny_t_lab = tiny.read_train_data()
        tiny_v_img, tiny_v_lab = tiny.read_vali_data()
        stl_t_img, stl_t_lab = stl.read_train_data()
        stl_v_img, stl_v_lab = stl.read_vali_data()

        print('build model ...')
        self.build_graph()

        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        f = open(FILENAME, 'w')

        mp = 0
        mp1_1, mp1_2, mp1_3 = 0, 0, 0
        mp2_1, mp2_2, mp2_3 = 0, 0, 0
        mp3_1, mp3_2, mp3_3 = 0, 0, 0

        if args.is_train:
            lr_v = args.init_lr
            print('train jointly ...')
            for step in range(args.max_step):
                if step is args.decay_step0 or step is args.decay_step1:
                    lr_v *= 0.1
                    print('lr decay to %.4f ...' % lr_v)
                cifar_batch_img, cifar_batch_lab = self.generate_augment_train_batch(32, cifar_t_img, cifar_t_lab)
                tiny_batch_img, tiny_batch_lab = self.generate_augment_train_batch(64, tiny_t_img, tiny_t_lab)
                stl_batch_img, stl_batch_lab = self.generate_augment_train_batch(96, stl_t_img, stl_t_lab)
                if step % args.num_batches == 0:
                    _, cifar_acc = self.full_validation(self.est_ls32, self.est_ac32, sess, 32, cifar_v_img,
                                                        tiny_v_img, stl_v_img, cifar_v_lab,
                                                        (args.nlevels - 1) * np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32))
                    _, tiny_acc = self.full_validation(self.est_ls64, self.est_ac64, sess, 64, cifar_v_img,
                                                       tiny_v_img, stl_v_img, tiny_v_lab,
                                                       (args.nlevels - 1) * np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32))
                    _, stl_acc = self.full_validation(self.est_ls96, self.est_ac96, sess, 96, cifar_v_img,
                                                      tiny_v_img, stl_v_img, stl_v_lab,
                                                      (args.nlevels - 1) * np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32))
                    print('Epoch {:03d}, cifar: {:.4f}, tiny: {:.4f}, stl: {:.4f}'
                          .format(step // args.num_batches, cifar_acc, tiny_acc, stl_acc))
                    print >> f, ('Epoch {:03d}, cifar: {:.4f}, tiny: {:.4f}, stl: {:.4f}'
                                 .format(step // args.num_batches, cifar_acc, tiny_acc, stl_acc))
                    if mp1_1 < cifar_acc:
                        mp1_1 = cifar_acc
                        mp1_2 = tiny_acc
                        mp1_3 = stl_acc
                    if mp2_2 < tiny_acc:
                        mp2_1 = cifar_acc
                        mp2_2 = tiny_acc
                        mp2_3 = stl_acc
                    if mp3_3 < stl_acc:
                        mp3_1 = cifar_acc
                        mp3_2 = tiny_acc
                        mp3_3 = stl_acc
                    if mp < (cifar_acc + tiny_acc + stl_acc) / 3:
                        mp = (cifar_acc + tiny_acc + stl_acc) / 3
                _ = sess.run(self.op_est32,
                             {self.img32: cifar_batch_img, self.img64: tiny_batch_img, self.img96: stl_batch_img,
                              self.compress: np.concatenate(((args.nlevels - 1) * np.ones([int(0.75 * args.batch_size), sel_blocks, 1, 1, 1], np.float32),
                                             np.random.randint(0, args.nlevels, [int(0.25 * args.batch_size), sel_blocks, 1, 1, 1])), 0),
                              self.lab: cifar_batch_lab, self.lr: lr_v, self.is_training: True, self.epsilon: 0})
                _ = sess.run(self.op_est64,
                             {self.img32: cifar_batch_img, self.img64: tiny_batch_img, self.img96: stl_batch_img,
                              self.compress: np.concatenate(((args.nlevels - 1) * np.ones([int(0.75 * args.batch_size), sel_blocks, 1, 1, 1], np.float32),
                                             np.random.randint(0, args.nlevels, [int(0.25 * args.batch_size), sel_blocks, 1, 1, 1])), 0),
                              self.lab: tiny_batch_lab, self.lr: lr_v, self.is_training: True, self.epsilon: 0})
                _ = sess.run(self.op_est96,
                             {self.img32: cifar_batch_img, self.img64: tiny_batch_img, self.img96: stl_batch_img,
                              self.compress: np.concatenate(((args.nlevels - 1) * np.ones([int(0.75 * args.batch_size), sel_blocks, 1, 1, 1], np.float32),
                                             np.random.randint(0, args.nlevels, [int(0.25 * args.batch_size), sel_blocks, 1, 1, 1])), 0),
                              self.lab: stl_batch_lab, self.lr: lr_v, self.is_training: True, self.epsilon: 0})
        else:
            saver.restore(sess, args.model_path)

        print('max_performance: {:.4f} / {:.4f} / {:.4f} \n'
              '                 {:.4f} / {:.4f} / {:.4f} \n'
              '                 {:.4f} / {:.4f} / {:.4f}'
              .format(mp1_1, mp1_2, mp1_3, mp2_1, mp2_2, mp2_3, mp3_1, mp3_2, mp3_3))
        print >> f, ('max_performance: {:.4f} / {:.4f} / {:.4f} \n'
                     '                 {:.4f} / {:.4f} / {:.4f} \n'
                     '                 {:.4f} / {:.4f} / {:.4f}'
                     .format(mp1_1, mp1_2, mp1_3, mp2_1, mp2_2, mp2_3, mp3_1, mp3_2, mp3_3))

        print('train policy network ...')

        lr_v = 1e-3
        lr_w = 1e-1
        mp1_1, mp1_2, mp1_3 = 0, 0, 0
        mp2_1, mp2_2, mp2_3 = 0, 0, 0
        mp3_1, mp3_2, mp3_3 = 0, 0, 0
        mc1_1, mc1_2, mc1_3 = 0, 0, 0
        mc2_1, mc2_2, mc2_3 = 0, 0, 0
        mc3_1, mc3_2, mc3_3 = 0, 0, 0

        for stage in range(10):
            print('stage %d ...' % (stage + 1))
            if stage > 4:
                lr_v = 1e-4
            for step in range(4693 * 2):
                cifar_batch_img, cifar_batch_lab = self.generate_augment_train_batch(32, cifar_t_img, cifar_t_lab)
                tiny_batch_img, tiny_batch_lab = self.generate_augment_train_batch(64, tiny_t_img, tiny_t_lab)
                stl_batch_img, stl_batch_lab = self.generate_augment_train_batch(96, stl_t_img, stl_t_lab)
                if step % args.num_batches == 0:
                    cifar_comp, cifar_acc = self.full_validation(self.com_cp32, self.com_ac32, sess, 32, cifar_v_img,
                                                                 tiny_v_img, stl_v_img, cifar_v_lab,
                                                                 (args.nlevels - 1) * np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32))
                    tiny_comp, tiny_acc = self.full_validation(self.com_cp64, self.com_ac64, sess, 64, cifar_v_img,
                                                               tiny_v_img, stl_v_img, tiny_v_lab,
                                                               (args.nlevels - 1) * np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32))
                    stl_comp, stl_acc = self.full_validation(self.com_cp96, self.com_ac96, sess, 96, cifar_v_img,
                                                             tiny_v_img, stl_v_img, stl_v_lab,
                                                             (args.nlevels - 1) * np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32))
                    print('Epoch {:03d}, cifar: {:.4f} ({:.4f}), tiny: {:.4f} ({:.4f}), stl: {:.4f} ({:.4f})'
                          .format(step // args.num_batches, cifar_acc, cifar_comp, tiny_acc, tiny_comp, stl_acc, stl_comp))
                    print >> f, ('Epoch {:03d}, cifar: {:.4f} ({:.4f}), tiny: {:.4f} ({:.4f}), stl: {:.4f} ({:.4f})'
                                 .format(step // args.num_batches, cifar_acc, cifar_comp, tiny_acc, tiny_comp, stl_acc, stl_comp))
                    if mp1_1 < cifar_acc:
                        mp1_1 = cifar_acc
                        mp1_2 = tiny_acc
                        mp1_3 = stl_acc
                        mc1_1 = cifar_comp
                        mc1_2 = tiny_comp
                        mc1_3 = stl_comp
                    if mp2_2 < tiny_acc:
                        mp2_1 = cifar_acc
                        mp2_2 = tiny_acc
                        mp2_3 = stl_acc
                        mc2_1 = cifar_comp
                        mc2_2 = tiny_comp
                        mc2_3 = stl_comp
                    if mp3_3 < stl_acc:
                        mp3_1 = cifar_acc
                        mp3_2 = tiny_acc
                        mp3_3 = stl_acc
                        mc3_1 = cifar_comp
                        mc3_2 = tiny_comp
                        mc3_3 = stl_comp
                step_epsilon = np.power(0.99, step)
                _ = sess.run(self.op_sel32,
                             {self.img32: cifar_batch_img, self.img64: tiny_batch_img, self.img96: stl_batch_img,
                              self.compress: np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32),
                              self.lab: cifar_batch_lab, self.lr: lr_v, self.is_training: True,
                              self.epsilon: step_epsilon})
                _ = sess.run(self.op_sel64,
                             {self.img32: cifar_batch_img, self.img64: tiny_batch_img, self.img96: stl_batch_img,
                              self.compress: np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32),
                              self.lab: tiny_batch_lab, self.lr: lr_v, self.is_training: True,
                              self.epsilon: step_epsilon})
                _ = sess.run(self.op_sel96,
                             {self.img32: cifar_batch_img, self.img64: tiny_batch_img, self.img96: stl_batch_img,
                              self.compress: np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32),
                              self.lab: stl_batch_lab, self.lr: lr_v, self.is_training: True,
                              self.epsilon: step_epsilon})
            if stage < 9:
                if stage is 3 or stage is 6:
                    lr_w *= 0.1
                for step in range(15642 * 2):
                    cifar_batch_img, cifar_batch_lab = self.generate_augment_train_batch(32, cifar_t_img, cifar_t_lab)
                    tiny_batch_img, tiny_batch_lab = self.generate_augment_train_batch(64, tiny_t_img, tiny_t_lab)
                    stl_batch_img, stl_batch_lab = self.generate_augment_train_batch(96, stl_t_img, stl_t_lab)
                    _ = sess.run(self.op_com32,
                                 {self.img32: cifar_batch_img, self.img64: tiny_batch_img, self.img96: stl_batch_img,
                                  self.compress: np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32),
                                  self.lab: cifar_batch_lab, self.lr: lr_w, self.is_training: True,
                                  self.epsilon: 0})
                    _ = sess.run(self.op_com64,
                                 {self.img32: cifar_batch_img, self.img64: tiny_batch_img, self.img96: stl_batch_img,
                                  self.compress: np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32),
                                  self.lab: tiny_batch_lab, self.lr: lr_w, self.is_training: True,
                                  self.epsilon: 0})
                    _ = sess.run(self.op_com96,
                                 {self.img32: cifar_batch_img, self.img64: tiny_batch_img, self.img96: stl_batch_img,
                                  self.compress: np.ones([args.batch_size, sel_blocks, 1, 1, 1], np.float32),
                                  self.lab: stl_batch_lab, self.lr: lr_w, self.is_training: True,
                                  self.epsilon: 0})

        print('max_performance: {:.4f} ({:.4f}) / {:.4f} ({:.4f}) / {:.4f} ({:.4f}) \n'
              '                 {:.4f} ({:.4f}) / {:.4f} ({:.4f}) / {:.4f} ({:.4f}) \n'
              '                 {:.4f} ({:.4f}) / {:.4f} ({:.4f}) / {:.4f} ({:.4f})'
              .format(mp1_1, mc1_1, mp1_2, mc1_2, mp1_3, mc1_3,
                      mp2_1, mc2_1, mp2_2, mc2_2, mp2_3, mc2_3,
                      mp3_1, mc3_1, mp3_2, mc3_2, mp3_3, mc3_3))
        print >> f, ('max_performance: {:.4f} ({:.4f}) / {:.4f} ({:.4f}) / {:.4f} ({:.4f}) \n'
                     '                 {:.4f} ({:.4f}) / {:.4f} ({:.4f}) / {:.4f} ({:.4f}) \n'
                     '                 {:.4f} ({:.4f}) / {:.4f} ({:.4f}) / {:.4f} ({:.4f})'
                     .format(mp1_1, mc1_1, mp1_2, mc1_2, mp1_3, mc1_3,
                             mp2_1, mc2_1, mp2_2, mc2_2, mp2_3, mc2_3,
                             mp3_1, mc3_1, mp3_2, mc3_2, mp3_3, mc3_3))

    # Helper functions
    def full_validation(self, loss, acc, sess, id, vali_data32, vali_data64, vali_data96, vali_labels, comp):
        if id is 32:
            num_batches = 10000 // args.batch_size
            order = np.random.choice(10000, num_batches * args.batch_size)
            vali_data_subset_32 = vali_data32[order, ...]
            vali_data_subset_64 = vali_data64[:args.batch_size, ...]
            vali_data_subset_96 = vali_data96[:args.batch_size, ...]
            vali_labels_subset = vali_labels[order]
            ls_list = []
            ac_list = []
            for step in range(num_batches):
                offset = step * args.batch_size
                feed_dict = {self.img32: vali_data_subset_32[offset:offset + args.batch_size, ...],
                             self.img64: vali_data_subset_64,
                             self.img96: vali_data_subset_96,
                             self.lab: vali_labels_subset[offset:offset + args.batch_size],
                             self.compress: comp,
                             self.lr: 0,
                             self.is_training: False,
                             self.epsilon: 0}
                ls, ac = sess.run([loss, acc], feed_dict=feed_dict)
                ls_list.append(ls)
                ac_list.append(ac)
        elif id is 64:
            num_batches = 10000 // args.batch_size
            order = np.random.choice(10000, num_batches * args.batch_size)
            vali_data_subset_32 = vali_data32[:args.batch_size, ...]
            vali_data_subset_64 = vali_data64[order, ...]
            vali_data_subset_96 = vali_data96[:args.batch_size, ...]
            vali_labels_subset = vali_labels[order]
            ls_list = []
            ac_list = []
            for step in range(num_batches):
                offset = step * args.batch_size
                feed_dict = {self.img32: vali_data_subset_32,
                             self.img64: vali_data_subset_64[offset:offset + args.batch_size, ...],
                             self.img96: vali_data_subset_96,
                             self.lab: vali_labels_subset[offset:offset + args.batch_size],
                             self.compress: comp,
                             self.lr: 0,
                             self.is_training: False,
                             self.epsilon: 0}
                ls, ac = sess.run([loss, acc], feed_dict=feed_dict)
                ls_list.append(ls)
                ac_list.append(ac)
        elif id is 96:
            num_batches = 8000 // args.batch_size
            order = np.random.choice(8000, num_batches * args.batch_size)
            vali_data_subset_32 = vali_data32[:args.batch_size, ...]
            vali_data_subset_64 = vali_data64[:args.batch_size, ...]
            vali_data_subset_96 = vali_data96[order, ...]
            vali_labels_subset = vali_labels[order]
            ls_list = []
            ac_list = []
            for step in range(num_batches):
                offset = step * args.batch_size
                feed_dict = {self.img32: vali_data_subset_32,
                             self.img64: vali_data_subset_64,
                             self.img96: vali_data_subset_96[offset:offset + args.batch_size, ...],
                             self.lab: vali_labels_subset[offset:offset + args.batch_size],
                             self.compress: comp,
                             self.lr: 0,
                             self.is_training: False,
                             self.epsilon: 0}
                ls, ac = sess.run([loss, acc], feed_dict=feed_dict)
                ls_list.append(ls)
                ac_list.append(ac)

        return np.mean(ls_list), np.mean(ac_list)

    def generate_augment_train_batch(self, id, train_data, train_labels):
        if id is 32:
            offset = np.random.choice(50000, args.batch_size)
            batch_data = train_data[offset, ...]
            batch_data = cifar.random_crop_and_flip(batch_data, padding_size=4)
            batch_label = train_labels[offset]
        elif id is 64:
            offset = np.random.choice(100000, args.batch_size)
            batch_data = train_data[offset, ...]
            batch_data = tiny.random_crop_and_flip(batch_data, padding_size=4)
            batch_label = train_labels[offset]
        elif id is 96:
            offset = np.random.choice(5000, args.batch_size)
            batch_data = train_data[offset, ...]
            batch_data = stl.random_crop_and_flip(batch_data, padding_size=4)
            batch_label = train_labels[offset]

        return batch_data, batch_label

    def density(self, policy):
        policy = policy + tf.where(policy == 3.,
                                   tf.ones(policy.get_shape().as_list()),
                                   tf.zeros(policy.get_shape().as_list()))
        block_use = tf.reduce_mean(tf.reduce_sum(policy, [1, 2, 3, 4])) / total_blocks

        return block_use

    def sample_pred(self, policy):
        shape = policy.get_shape().as_list()
        rand = tf.random_uniform([shape[0], shape[1]], minval=0, maxval=1, dtype=tf.float32)
        rlo = tf.zeros([shape[0], shape[1]], dtype=tf.int32)
        for i in range(args.nlevels - 1):
            if i is 0:
                temp = policy[:, :, 0]
            else:
                temp = tf.reduce_sum(policy[:, :, :i + 1], 2)
            rlo += tf.where(tf.greater(temp, rand),
                            tf.zeros([shape[0], shape[1]], dtype=tf.int32),
                            tf.ones([shape[0], shape[1]], dtype=tf.int32))
        pred = tf.cast(tf.reshape(rlo, [shape[0], shape[1], 1, 1, 1]), tf.float32)

        return pred

    def get_reward(self, preds, targets, policy, rho):
        block_use = self.density(policy)
        sparse_reward = 1.0 - tf.pow(block_use, 2)
        match = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=targets)

        return sparse_reward / rho - match + 1.0

    def performance(self, logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        ones = tf.constant(np.ones([int(args.batch_size / 2)]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([int(args.batch_size / 2)]), dtype=tf.float32)
        acc = tf.reduce_mean(tf.where(tf.equal(tf.to_int32(tf.argmax(logits, 1)), labels), ones, zeros))

        return tf.reduce_mean(cross_entropy), acc

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            if grad_and_vars[0][0] is None:
                continue

            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)

                grads.append(expanded_g)

            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads


# data_input.maybe_download_and_extract()
# Initialize the Train object
train = Train()
# Start the training session
train.train()
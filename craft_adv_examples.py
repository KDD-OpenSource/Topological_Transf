#TODO
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import warnings
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
import keras.backend as K
from keras.models import load_model
import tensorflow_hub as hub

from util import get_data, get_model, cross_entropy
from attacks import fast_gradient_sign_method, basic_iterative_method, saliency_map_method
from cw_attacks import CarliniL2, CarliniLID
from dataset import split_trainings_data_into_train_and_val
from deepfool import deepfool
import sys
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
import zipfile
from tensorflow.python.platform import gfile

# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 0.40, 'eps_iter': 0.010, 'image_size': 28, 'num_channels': 1, 'num_labels': 10},
    'mnist_aug': {'eps': 0.40, 'eps_iter': 0.010, 'image_size': 28, 'num_channels': 1, 'num_labels': 10},
    'fashion_mnist': {'eps': 0.40, 'eps_iter': 0.010, 'image_size': 28, 'num_channels': 1, 'num_labels': 10},
    'fashion_mnist_aug': {'eps': 0.40, 'eps_iter': 0.010, 'image_size': 28, 'num_channels': 1, 'num_labels': 10},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005, 'image_size': 32, 'num_channels': 3, 'num_labels': 10},
    'cifar_aug': {'eps': 0.050, 'eps_iter': 0.005, 'image_size': 32, 'num_channels': 3, 'num_labels': 10},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010, 'image_size': 32, 'num_channels': 3, 'num_labels': 10},
    'svhn_aug': {'eps': 0.130, 'eps_iter': 0.010, 'image_size': 32, 'num_channels': 3, 'num_labels': 10},
    'Animals_10': {'eps': 0.130, 'eps_iter': 0.010, 'image_size': 224, 'num_channels': 3, 'num_labels': 2}, # TODO 10 makes no sense change model to two
    'PetImages': {'eps': 0.130, 'eps_iter': 0.010, 'image_size': 224, 'num_channels': 3, 'num_labels': 2}
}

# CLIP_MIN = 0.0
# CLIP_MAX = 1.0
CLIP_MIN = -0.5
CLIP_MAX = 0.5
PATH_DATA = "data/"

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def universal_perturbation(dataset, f, grads, delta=0.2, max_iter_uni = np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """

    v = 0
    fooling_rate = 0.0
    num_images =  np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION

    itr = 0
    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
        np.random.shuffle(dataset)

        print ('Starting pass number ', itr)

        # Go through the data set and compute the perturbation increments sequentially
        for k in range(0, num_images):
            cur_img = dataset[k:(k+1), :, :, :]

            if int(np.argmax(np.array(f(cur_img)).flatten())) == int(np.argmax(np.array(f(cur_img+v)).flatten())):
                print('>> k = ', k, ', pass #', itr)

                # Compute adversarial perturbation
                dr,iter,_,_ = deepfool(cur_img + v, f, grads, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr

                    # Project on l_p ball
                    v = proj_lp(v, xi, p)

        itr = itr + 1

        # Perturb the dataset with computed perturbation
        dataset_perturbed = dataset + v

        est_labels_orig = np.zeros((num_images))
        est_labels_pert = np.zeros((num_images))

        batch_size = 100
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

        # Compute the estimated labels in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
            est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
        print('FOOLING RATE = ', fooling_rate)

    return v

def craft_one_type(sess, model, X, Y, dataset, attack, batch_size):
    """
    TODO
    :param sess:
    :param model:
    :param X:
    :param Y:
    :param dataset:
    :param attack:
    :param batch_size:
    :return:
    """
    if attack == 'fgsm':
        # FGSM attack
        print('Crafting fgsm adversarial samples...')
        X_adv = fast_gradient_sign_method(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], clip_min=CLIP_MIN,
            clip_max=CLIP_MAX, batch_size=batch_size
        )
    elif attack in ['bim-a', 'bim-b']:
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        its, results = basic_iterative_method(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'],
            eps_iter=ATTACK_PARAMS[dataset]['eps_iter'], clip_min=CLIP_MIN,
            clip_max=CLIP_MAX, batch_size=batch_size
        )
        if attack == 'bim-a':
            # BIM-A
            # For each sample, select the time step where that sample first
            # became misclassified
            X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])
        else:
            # BIM-B
            # For each sample, select the very last time step
            X_adv = results[-1]
    elif attack == 'jsma':
        # JSMA attack
        print('Crafting jsma adversarial samples. This may take > 5 hours')
        X_adv = saliency_map_method(
            sess, model, X, Y, theta=1, gamma=0.1, clip_min=CLIP_MIN, clip_max=CLIP_MAX
        )
    elif attack == 'cw-l2':
        # C&W attack
        print('Crafting %s examples. This takes > 5 hours due to internal grid search' % attack)
        image_size = ATTACK_PARAMS[dataset]['image_size']
        num_channels = ATTACK_PARAMS[dataset]['num_channels']
        num_labels = ATTACK_PARAMS[dataset]['num_labels']
        cw_attack = CarliniL2(sess, model, image_size, num_channels, num_labels, batch_size=batch_size)
        X_adv = cw_attack.attack(X, Y)
    elif attack == 'cw-lid':
        # C&W attack to break LID detector
        print('Crafting %s examples. This takes > 5 hours due to internal grid search' % attack)
        image_size = ATTACK_PARAMS[dataset]['image_size']
        num_channels = ATTACK_PARAMS[dataset]['num_channels']
        num_labels = ATTACK_PARAMS[dataset]['num_labels']
        cw_attack = CarliniLID(sess, model, image_size, num_channels, num_labels, batch_size=batch_size)
        X_adv = cw_attack.attack(X, Y)
    elif attack == 'UAP':
        image_size = ATTACK_PARAMS[dataset]['image_size']
        num_channels = ATTACK_PARAMS[dataset]['num_channels']
        num_labels = ATTACK_PARAMS[dataset]['num_labels']
        with tf.device('/gpu:0'):
            persisted_sess = tf.Session()
            inception_model_path = os.path.join('data', 'tensorflow_inception_graph.pb')

            if os.path.isfile(inception_model_path) == 0:
                print('Downloading Inception model...')
                urlretrieve("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip",
                            os.path.join('data', 'inception5h.zip'))
                # Unzipping the file
                zip_ref = zipfile.ZipFile(os.path.join('data', 'inception5h.zip'), 'r')
                zip_ref.extract('tensorflow_inception_graph.pb', 'data')
                zip_ref.close()

            model = os.path.join(inception_model_path)

            # Load the Inception model
            with gfile.FastGFile(model, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                persisted_sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

            persisted_sess.graph.get_operations()
        persisted_sess.graph.get_operations()

        def jacobian(y_flat, x, inds):
            n = num_labels  # Not really necessary, just a quick fix.
            loop_vars = [
                tf.constant(0, tf.int32),
                tf.TensorArray(tf.float32, size=n),
            ]
            _, jacobian = tf.while_loop(
                lambda j, _: j < n,
                lambda j, result: (j + 1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
                loop_vars)
            return jacobian.stack()

        persisted_input = persisted_sess.graph.get_tensor_by_name("input:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")
        print('>> Compiling the gradient tensorflow functions. This might take some time...')
        y_flat = tf.reshape(persisted_output, (-1,))
        inds = tf.placeholder(tf.int32, shape=(num_labels,))
        dydx = jacobian(y_flat, persisted_input, inds)

        print('>> Computing feedforward function...')
        def f(image_inp):
            return persisted_sess.run(persisted_output,
                                      feed_dict={persisted_input: np.reshape(image_inp, (image_size, image_size,num_channels))})

        print('>> Computing gradient function...')
        def grad_fs(image_inp, indices):
            return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)

        X_adv = universal_perturbation(X, f, grad_fs, delta=0.2,num_classes=num_labels)

    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size, verbose=0)
    print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    np.save(os.path.join(PATH_DATA, 'Adv_%s_%s.npy' % (dataset, attack)), X_adv)
    np.save(os.path.join(PATH_DATA, 'Original_Image_%s_%s.npy' % (dataset, attack)), X)
    l2_diff = np.linalg.norm(
        X_adv.reshape((len(X), -1)) -
        X.reshape((len(X), -1)),
        axis=1
    ).mean()
    np.save(os.path.join(PATH_DATA, 'Adv_labels_%s_%s.npy' % (dataset, attack)), Y)
    print("Average L-2 perturbation size of the %s attack: %0.2f" %
          (attack, l2_diff))

def main(args):
    assert args.dataset in ['mnist', 'mnist_aug', 'mnist_swap', 'mnist_swap2','cifar', 'svhn', 'Animals_10', 'PetImages'], \
        "Dataset parameter must be either 'mnist', 'cifar', 'Animals_10', 'PetImages' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'UAP', 'cw-lid', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma', 'cw-l2', 'all' or 'cw-lid' for attacking LID detector"
    model_file = os.path.join(PATH_DATA, "model_%s.h5" % args.dataset)
    assert os.path.isfile(model_file), \
        'model file not found... must first train model using train_model.py.'
    if args.dataset == 'svhn' and args.attack == 'cw-l2':
        assert args.batch_size == 16, \
        "svhn has 26032 test images, the batch_size for cw-l2 attack should be 16, " \
        "otherwise, there will be error at the last batch-- needs to be fixed."


    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    # Create TF session, set it as Keras backend
    sess = tf.Session()
    K.set_session(sess)
    if (args.dataset == "PetImages" or args.dataset == "Animals_10") and (args.attack == 'cw-l2' or args.attack == 'cw-lid'):
        model = load_model((model_file), custom_objects={'KerasLayer': hub.KerasLayer}) # TODO why this not used?
        # TODO maybe  remove last softmax layer somehow - look up why
    if args.attack == 'cw-l2' or args.attack == 'cw-lid' and not (args.dataset == "PetImages" or args.dataset == "Animals_10"):
        warnings.warn("Important: remove the softmax layer for cw attacks!")
        # use softmax=False to load without softmax layer
        model = get_model(args.dataset, softmax=False)
        model.compile(
            loss=cross_entropy,
            optimizer='adadelta',
            metrics=['accuracy']
        )
        model.load_weights(model_file)
    else:
        model = load_model(model_file, custom_objects={'KerasLayer': hub.KerasLayer})

    _, _, X_test, Y_test = get_data(args.dataset)
    print(X_test.shape)
    #label = np.argmax(Y_test, axis=1) # TODO fix for Animals_10 - why needed?
    _, acc = model.evaluate(X_test, Y_test, batch_size=args.batch_size, verbose=0)
    print("Accuracy on the test set: %0.2f%%" % (100*acc))

    if args.attack == 'cw-lid': # white box attacking LID detector - an example
        X_test = X_test[:1000]
        Y_test = Y_test[:1000]

    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2']:
            craft_one_type(sess, model, X_test, Y_test, args.dataset, attack,
                           args.batch_size)
    else:
        # Craft one specific attack type
        # TODO ? labels= np.argmax(X_test, axis=1) # ensure that for Animals10 labels not one hot encoded
        craft_one_type(sess, model,X_test, Y_test, args.dataset, args.attack,
                       args.batch_size)# Y_test
    print('Adversarial samples crafted and saved to %s ' % PATH_DATA)
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar','Animals_10', 'PetImages',  or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma', or 'cw-l2' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=100)
    args = parser.parse_args()
    main(args)

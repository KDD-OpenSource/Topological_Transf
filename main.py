# with umap, svhn, jsma

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10
from torchvision import transforms
import kmapper
import torchvision
from torch.utils.data import DataLoader
import umapF
from sklearn.cluster._dbscan import DBSCAN
from sklearn.manifold import Isomap
import networkx as nx
import netlsd
import tensorflow.compat.v1 as tf
import numpy as np
import kmapper as km
from dataset import CatDogDataset, split_trainings_data_into_train_and_val
from keras.models import load_model
from util import get_model, cross_entropy, get_data
from keras.preprocessing.image import ImageDataGenerator
import os
from subprocess import call
import scipy.io as sio
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from craft_adv_examples import craft_one_type
import keras.backend as K
import sklearn
import argparse
#############################

###############################


CLIP_MIN = -0.5
CLIP_MAX = 0.5
PATH_DATA = "data/"

#######################
AMOUNT_PIX_SWAP= 18
EPOCHS = 1500
DATASET= "mnist"
attack_used = 'fgsm'
#################

#def craft_adver(args_attack):
#    if args_attack== '':

def train(X_train, Y_train, X_test, Y_test, dataset='mnist', batch_size=128, epochs=50):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param args:
    :return:
    """
    print('Data set: %s' % dataset)

    #X_train, Y_train, X_test, Y_test = get_data(dataset)

    # tuner = RandomSearch(
    #     build_model(dataset),
    #     objective= 'val_accuracy',
    #     max_trials= 1,
    #     executions_per_trial= 1,
    #     directory = LOG_DIR
    # )
    # tuner.search(
    #     x= X_train,
    #     y=Y_train,
    #     epochs= 1,
    #     batch_size= batch_size,
    #     validation_data = (X_test, Y_test)
    # )

#------------------------------------------
    model = get_model(dataset)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=6e-3),
        metrics=['accuracy']
    )
#  --------------------------------------
#     # training without data augmentation
#     model.fit(
#         X_train, Y_train,
#         epochs=epochs,
#         batch_size=batch_size,
#         shuffle=True,
#         verbose=1,
#         validation_data=(X_test, Y_test)
#     )

    # training with data augmentation
    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    print(X_train.shape)
    print(Y_train.shape)
    model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test))

    model.save('data/model_%s.h5' % dataset)


def data_aug_for_topology_change(image):
    image_array = np.random.randint(0, 28, (10000,2)).astype(np.uint8)
    value = np.random.randint(0,255)
    for i in range(len(image_array)): #set a random pixel to random value for each image instance
        image[i][image_array[i][0]][image_array[i][1]] =value
        value = np.random.randint(0, 255)

    return image

##################################
#Transf script
#load dataset
#load labels
for i in range(30):
    AMOUNT_PIX_SWAP= i
    if DATASET== "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = (X_train / 255.0) - (1.0 - CLIP_MAX)
        X_test = (X_test / 255.0) - (1.0 - CLIP_MAX)
    elif DATASET == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = (X_train / 255.0) - (1.0 - CLIP_MAX)
        X_test = (X_test / 255.0) - (1.0 - CLIP_MAX)
    elif DATASET == "cifar":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = (X_train / 255.0) - (1.0 - CLIP_MAX)
        X_test = (X_test / 255.0) - (1.0 - CLIP_MAX)
    elif DATASET == "svhn":
        training = sio.loadmat(os.path.join(PATH_DATA, 'svhn_train.mat'))
        testing  = sio.loadmat(os.path.join(PATH_DATA, 'svhn_test.mat'))
        X_train = np.transpose(training['X'], axes=[3, 0, 1, 2])
        X_test = np.transpose(testing['X'], axes=[3, 0, 1, 2])
        # reshape (n_samples, 1) to (n_samples,) and change 1-index
        # to 0-index
        y_train = np.reshape(training['y'], (-1,)) - 1
        y_test = np.reshape(testing['y'], (-1,)) - 1

        # cast pixels to floats, normalize to [0, 1] range
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = (X_train / 255.0) - (1.0 - CLIP_MAX)
        X_test = (X_test / 255.0) - (1.0 - CLIP_MAX)

    # one-hot-encode the labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    #data = torchvision.datasets.MNIST(download=True, train=True, root=".").data.float() #Goal: augment mnist such that topology changes measurably
    # print("data shape")
    # print(data.shape)

    for i in range(AMOUNT_PIX_SWAP): #set i indexes to random values, augment data set
        X_train = data_aug_for_topology_change(X_train) # TODO leave out if mnist with mnist compare
        X_test = data_aug_for_topology_change(X_test)


    #limit to 10000 data points in order to limit computational runtime

    X_train= X_train[:10000]
    Y_train= Y_train[:10000]

    if DATASET == "cifar" or DATASET == "svhn":
        data_aug = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
    else:
        data_aug = X_train.reshape(X_train.shape[0], X_train.shape[1]* X_train.shape[2])
    print(X_train.shape)
    # #######################################
    #TDA mapper
    mapper = km.KeplerMapper(verbose=2)
    projected_data = mapper.fit_transform(data_aug, projection= [Isomap(n_components=100, n_jobs=-1),umap.UMAP()]) #,projection=sklearn.decomposition.PCA())
    #cluster data
    graph = mapper.map(
        projected_data,
        clusterer=DBSCAN(),
        cover=km.Cover(35, 0.4),
    )
    nx_graph = kmapper.adapter.to_nx(graph)

    fileID = "projection_augmented"

    #visualize the graph
    mapper.visualize(graph,
                     path_html= "mapper_ex_" + fileID + ".html",
                     title= fileID)

    descriptor1= netlsd.heat(nx_graph)
    #print(descriptor1)
    ############################
    if DATASET == "mnist":
        data2 = torchvision.datasets.MNIST(download=True, train=True, root=".").data.float()  # compare with MNIST
        data2 = data2[:10000]
        data2 = data2.reshape(data2.shape[0], data2.shape[1] * data2.shape[2])

    elif DATASET == "fashion_mnist":
        data2 = torchvision.datasets.FashionMNIST(download=True, train=True, root=".").data.float() # compare with FashionMNIST
        data2 = data2[:10000]
        data2 = data2.reshape(data2.shape[0], data2.shape[1] * data2.shape[2])

    elif DATASET == "cifar": #CIFAR10
        data2= torchvision.datasets.CIFAR10(download=True, train= True, root=".").data #float? TODO
        data2 = data2[:10000]
        data2 = data2.reshape(data2.shape[0], data2.shape[1] * data2.shape[2] * data2.shape[3])
    elif DATASET == "svhn":
        data2 = torchvision.datasets.SVHN(download=True, root=".").data
        data2 = data2[:10000]
        data2 = data2.reshape(data2.shape[0], data2.shape[1] * data2.shape[2] * data2.shape[3])


    mapper2 = km.KeplerMapper(verbose=2)
    projected_data2 = mapper2.fit_transform( data2, projection=[Isomap(n_components=100, n_jobs=-1), umap.UMAP()]) #data2, projection=sklearn.decomposition.PCA())
    #cluster data
    graph2 = mapper2.map(
        projected_data2,
        clusterer=DBSCAN(),
        cover=km.Cover(35, 0.4),
    )

    fileID = "projection"

    #visualize the graph
    mapper.visualize(graph2,
                     path_html= "mapper_ex_" + fileID + ".html",
                     title= fileID)

    nx_graph2 = kmapper.adapter.to_nx(graph2)

    #descriptor2 = netlsd.heat(projected_data2)
    descriptor2 = netlsd.heat(nx_graph2) # compute the signature
    #print(descriptor2)

    distance = netlsd.compare(descriptor1, descriptor2) # compare the signatures using l2 distance
    print("distance:")
    print(distance)
    #### Test transferability of modified data set and the actual topological distance proxied by netlsd

    path = "../lid_adversarial_subspace_detection-master/data/"
    if DATASET == "mnist":
        arr = np.load(path + "Adv_mnist_" + attack_used+ ".npy")  ###load mnist adversarials
        label = np.load(path + "Adv_labels_mnist_" + attack_used+ ".npy")  # load original labels of mnist adversarials
        #print(arr.shape)
        original_im = np.load(path + "Original_Image_mnist_" + attack_used + ".npy")  # load and show original image if wanted
        original_im.reshape(arr.shape[0],1,28,28) #reshape
        model_file = path + "model_mnist.h5" #load mnist model
        model = load_model(model_file)
        arr.reshape(arr.shape[0], 1, 28, 28)
    elif DATASET == 'fashion_mnist':
        arr= np.load(path + "Adv_fashion_mnist_" + attack_used+ ".npy") ###load mnist adversarials
        label =np.load( path + "Adv_labels_fashion_mnist_" + attack_used+ ".npy") #load original labels of mnist adversarials
        #print(arr.shape)
        original_im= np.load(path + "Original_Image_fashion_mnist_" + attack_used+ ".npy") # load and show original image if wanted # TODO remove ses
        original_im.reshape(arr.shape[0],1,28,28) #reshape
        model_file = path + "model_fashion_mnist.h5" #load fashion mnist model
        model = load_model(model_file)
        arr.reshape(arr.shape[0], 1, 28, 28)
    elif DATASET =="cifar":
        arr = np.load(path + "Adv_cifar_" + attack_used+ ".npy")  ###load cifarS adversarials
        label = np.load(path + "Adv_labels_cifar_" + attack_used+ ".npy")  # load original labels of mnist adversarials
        #print(arr.shape)
        original_im = np.load(path + "Original_Image_cifar_" + attack_used+ ".npy")  # load and show original image if wanted # TODO remove ses
        original_im.reshape(arr.shape[0], 3, 32, 32)  # reshape
        model_file = path + "model_cifar.h5"  # load mnist model
        model = load_model(model_file)
        arr.reshape(arr.shape[0], 3, 32, 32)
    elif DATASET == "svhn":
        arr = np.load(path + "Adv_svhn_" + attack_used+ ".npy")  ###load mnist adversarials
        label = np.load(path + "Adv_labels_svhn_" + attack_used+ ".npy")  # load original labels of mnist adversarials
        print(arr.shape)
        original_im = np.load(
            path + "Original_Image_svhn_" + attack_used+ ".npy")  # load and show original image if wanted # TODO remove ses
        original_im.reshape(arr.shape[0], 3, 32, 32)  # reshape
        model_file = path + "model_svhn.h5"  # load mnist model
        model = load_model(model_file)
        arr.reshape(arr.shape[0], 3, 32, 32)
    #################save actual adv examples
    predicted_label = model.predict(arr)

    #if not np.array_equal(label[0], np.rint(predicted_label[0])):
    #list = np.array([arr[0]])  ### TODO implement adv success check?
    #label_list= np.array([label[0]])
    list =[]
    label_list=[]
    orig_im_list=[]
    for i in range(len(predicted_label)):
        if not np.array_equal(label[i], np.rint(predicted_label[i])): # if it is actually an adversarial example
            #print(arr[i].shape)
            list.append(arr[i])
            label_list.append(label[i])
            orig_im_list.append(original_im[i])
            #np.append(list, arr[i])
            #np.append(label_list,label[i])


    n_samples = 1024
    #print("list then original_im shape")
    #print(np.asarray(list).shape)
    print(original_im.shape)
    transfer_arr= list[0] - orig_im_list[0] # chose index 1
    print("transf arr shape ")
    print(transfer_arr.shape)

    for i in range(1, len(list)): # concatenate trans patterns
        transfer_arr = np.concatenate( (transfer_arr, list[i] - orig_im_list[i]) , axis=2)

    if DATASET == "mnist" or  DATASET =="fashion_mnist":
        transfer_arr= transfer_arr.reshape(transfer_arr.shape[2], 28,28, 1)
    elif DATASET == "cifar" or DATASET== "svhn":
        transfer_arr= transfer_arr.reshape(len(list), 32,32, 3)

    inputs_plus_trans= transfer_arr + X_test[:len(list)] #.detach().numpy

    if DATASET == "mnist":
        train(X_train, Y_train, X_test, Y_test, "mnist_aug", batch_size=128,
              epochs=EPOCHS)
        model2 = load_model('data/model_mnist_aug.h5')
    elif DATASET == "fashion_mnist":
        train(X_train, Y_train, X_test, Y_test, "fashion_mnist_aug", batch_size=128, epochs=EPOCHS) # TODO but changed input here
        model2 = load_model('data/model_fashion_mnist_aug.h5')
    elif DATASET == "cifar":
        train(X_train, Y_train, X_test, Y_test, "cifar_aug", batch_size=128,
              epochs=EPOCHS)
        model2 = load_model('data/model_cifar_aug.h5')
    elif DATASET == "svhn":
        train(X_train, Y_train, X_test, Y_test, "svhn_aug", batch_size=128,
              epochs=EPOCHS)  # TODO but changed input here
        model2 = load_model('data/model_svhn_aug.h5')
    ##########construct adv examples of the trained model using the changed dataset
    predicted_trans_label = model2.predict(inputs_plus_trans) # predictions of all X_test inputs modified with first adversarial example pattern
    trans_counter=0
    for i in range (len(predicted_trans_label)):
        if np.argmax(label_list[i]) != np.argmax(predicted_trans_label[i]):
            trans_counter+= 1
    print("transferability rate is:")
    print(trans_counter / len(predicted_trans_label))
    trans1= trans_counter / len(predicted_trans_label)
    # print("distance:")
    # print(distance)

    ################ generate adv examples on new model and test their transferability to mnist
    import craft_adv_examples
    sess = tf.Session()
    K.set_session(sess)
    path = "data/"

    ##### FGSM WITHOUT THIS, include for cw-lid####################
    # model3 = get_model("mnist_aug", softmax=False)
    # model3.compile(
    #     loss=cross_entropy,
    #     optimizer='adadelta',
    #     metrics=['accuracy']
    #     )
    if DATASET =="mnist":
        model_file = path + "model_mnist_aug.h5" #load mnist model
        model3 = load_model(model_file)
        # model3.load_weights(model_file)
        _, acc = model3.evaluate(X_test, Y_test, batch_size=100, verbose=0)
        # show acc?
        craft_adv_examples.craft_one_type(sess, model=model3, X=X_train, Y=Y_train, dataset='mnist_aug',
                                          attack=attack_used, batch_size=100)  # TODO put back in
        arr = np.load(path + "Adv_mnist_aug_" + attack_used+ ".npy")  ###load mnist aug adversarials
        label = np.load(path + "Adv_labels_mnist_aug_" + attack_used+ ".npy")  # load original labels of mnist aug adversarials
        original_im = np.load(path + "Original_Image_mnist_aug_" + attack_used+ ".npy")  # load and show original image if wanted
        original_im.reshape(arr.shape[0], 1, 28, 28)  # take one example adversarial pattern

    elif DATASET == "fashion_mnist":
        model_file = path + "model_fashion_mnist_aug.h5" #load mnist model
        model3 = load_model(model_file)
        # model3.load_weights(model_file)
        _, acc = model3.evaluate(X_test, Y_test, batch_size=100, verbose=0)
        # show acc?
        craft_adv_examples.craft_one_type(sess, model=model3, X=X_train, Y=Y_train, dataset='fashion_mnist_aug',
                                          attack=attack_used, batch_size=100)  # TODO put back in
        arr = np.load(path + "Adv_fashion_mnist_aug_" + attack_used+ ".npy")  ###load mnist aug adversarials
        label = np.load(path + "Adv_labels_fashion_mnist_aug_" + attack_used+ ".npy")  # load original labels of mnist aug adversarials
        original_im = np.load(path + "Original_Image_fashion_mnist_aug_" + attack_used+ ".npy")  # load and show original image if wanted
        original_im.reshape(arr.shape[0], 1, 28, 28)  # take one example adversarial pattern

    elif DATASET == "cifar":
        model_file = path + "model_cifar_aug.h5"
        model3 = load_model(model_file)
        # model3.load_weights(model_file)
        _, acc = model3.evaluate(X_test, Y_test, batch_size=100, verbose=0)
        # show acc?
        craft_adv_examples.craft_one_type(sess, model=model3, X=X_train, Y=Y_train, dataset='cifar_aug',
                                          attack=attack_used, batch_size=100)  # TODO put back in
        arr = np.load(path + "Adv_cifar_aug_" + attack_used+ ".npy")  ###load mnist aug adversarials
        label = np.load(path + "Adv_labels_cifar_aug_" + attack_used+ ".npy")  # load original labels of mnist aug adversarials
        original_im = np.load(path + "Original_Image_cifar_aug_" + attack_used+ ".npy")  # load and show original image if wanted
        original_im.reshape(arr.shape[0], 3, 32, 32)  # take one example adversarial pattern
    elif DATASET == "svhn":
        model_file = path + "model_svhn_aug.h5"
        model3 = load_model(model_file)
        # model3.load_weights(model_file)
        _, acc = model3.evaluate(X_test, Y_test, batch_size=100, verbose=0)
        # show acc?
        craft_adv_examples.craft_one_type(sess, model=model3, X=X_train, Y=Y_train, dataset='svhn_aug',
                                          attack=attack_used, batch_size=100)  # TODO put back in
        arr = np.load(path + "Adv_svhn_aug_" + attack_used+ ".npy")  ###load mnist aug adversarials
        label = np.load(path + "Adv_labels_svhn_aug_" + attack_used+ ".npy")  # load original labels of mnist aug adversarials
        original_im = np.load(path + "Original_Image_svhn_aug_" + attack_used+ ".npy")  # load and show original image if wanted
        original_im.reshape(arr.shape[0], 3, 32, 32)  # take one example adversarial pattern

    list = []  ### TODO implement adv success check?

    predicted_label = model3.predict(arr)

    for i in range(len(predicted_label)):
        if not np.array_equal(label[i], np.rint(predicted_label[i])): # if it is actually an adversarial example
            list.append(arr[i])
    #
    # if arr[1] in list: # TODO change index if not adversarial
    #     print("exist")
    # else:
    #     print("not exist")
    list =[]
    label_list=[]
    orig_im_list=[]
    for i in range(len(predicted_label)):
        if not np.array_equal(label[i], np.rint(predicted_label[i])): # if it is actually an adversarial example
            list.append(arr[i])
            label_list.append(label[i])
            orig_im_list.append(original_im[i])
    transfer_arr= list[0] - orig_im_list[0] # chose index 1
    print("transf arr shape ")
    print(transfer_arr.shape)
    #### TODO augmented dataset
    #testX = data[:10000] # use augmented set as test for transf  #.detach().numpy()
    #testy = labels[:10000] #.detach().numpy()


    for i in range(1, len(list)): # concatenate trans patterns
        transfer_arr = np.concatenate( (transfer_arr, list[i] - orig_im_list[i]) , axis=2)

    if DATASET == "mnist":
        transfer_arr = transfer_arr.reshape(len(list), 28, 28, 1)
        (real_X_train, real_y_train), (real_X_test, real_y_test) = mnist.load_data()
        real_X_train = real_X_train.reshape(-1, 28, 28, 1)
        real_X_test = real_X_test.reshape(-1, 28, 28, 1)
    elif DATASET =="fashion_mnist":
        transfer_arr = transfer_arr.reshape(len(list), 28, 28, 1)
        (real_X_train, real_y_train), (real_X_test, real_y_test) = fashion_mnist.load_data()
        real_X_train = real_X_train.reshape(-1, 28, 28, 1)
        real_X_test = real_X_test.reshape(-1, 28, 28, 1)
    elif DATASET == "cifar":
        transfer_arr = transfer_arr.reshape(len(list), 32, 32, 3)
        (real_X_train, real_y_train), (real_X_test, real_y_test) = cifar10.load_data()
        real_X_train = real_X_train.reshape(-1, 32, 32, 3)
        real_X_test = real_X_test.reshape(-1, 32, 32, 3)
    elif DATASET == "svhn":
        transfer_arr = transfer_arr.reshape(len(list), 32, 32, 3)
        training = sio.loadmat(os.path.join(PATH_DATA, 'svhn_train.mat'))
        testing = sio.loadmat(os.path.join(PATH_DATA, 'svhn_test.mat'))
        real_X_train = np.transpose(training['X'], axes=[3, 0, 1, 2])
        real_X_test = np.transpose(testing['X'], axes=[3, 0, 1, 2])
        # reshape (n_samples, 1) to (n_samples,) and change 1-index
        # to 0-index
        real_y_train = np.reshape(training['y'], (-1,)) - 1
        real_y_test = np.reshape(testing['y'], (-1,)) - 1

    # cast pixels to floats, normalize to [0, 1] range
    real_X_train = real_X_train.astype('float32')
    real_X_test = real_X_test.astype('float32')
    real_X_train = (real_X_train / 255.0) - (1.0 - CLIP_MAX)
    real_X_test = (real_X_test / 255.0) - (1.0 - CLIP_MAX)

    # one-hot-encode the labels
    real_Y_train = np_utils.to_categorical(y_train, 10)
    real_Y_test = np_utils.to_categorical(y_test, 10)


    inputs_plus_trans= transfer_arr + real_X_test[:len(list)]
    if DATASET == "mnist":
        model = load_model("../lid_adversarial_subspace_detection-master/data/model_mnist.h5")
    elif DATASET == "fashion_mnist":
        model = load_model("../lid_adversarial_subspace_detection-master/data/model_fashion_mnist.h5")
    elif DATASET == "cifar":
        model = load_model("../lid_adversarial_subspace_detection-master/data/model_cifar.h5")
    elif DATASET == "svhn":
        model = load_model("../lid_adversarial_subspace_detection-master/data/model_svhn.h5")


    predicted_trans_label = model.predict(inputs_plus_trans) # predictions of all X_test of mnist inputs modified with first adversarial example pattern
    trans_counter=0
    for i in range (len(predicted_trans_label)):
        if np.argmax(label_list[i]) != np.argmax(predicted_trans_label[i]):
            trans_counter+= 1
    with open("results.txt", "a") as myfile:
        #"transferability rate is: " + str(trans_counter / len(predicted_trans_label)) +
        myfile.write("distance " + str(distance) + "transf orig: "+ str(trans1) )
print(DATASET)

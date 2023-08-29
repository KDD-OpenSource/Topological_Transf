import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist, mnist
from torchvision import transforms
import kmapper
import torchvision
from torch.utils.data import DataLoader
import umap
from sklearn.cluster._dbscan import DBSCAN
from sklearn.manifold import Isomap
import networkx as nx
import netlsd
import tensorflow.compat.v1 as tf
import numpy as np
import kmapper as km
from dataset import CatDogDataset, split_trainings_data_into_train_and_val
from keras.models import load_model


def data_aug_for_topology_change(image):
    image_array = np.random.randint(0, 28, (10000,2)).astype(np.uint8)
    for i in range(len(image_array)): #set a random pixel to 255 for each image instance
        image[i][image_array[i][0]][image_array[i][1]] =255
    print("image shape after aug")
    print(image.shape)
    return image.numpy()


#print("----")
#g = nx.erdos_renyi_graph(100, 0.01) # create a random graph with 100 nodes

#descriptor2 = netlsd.heat(g) # compute the signature
#print(descriptor2)

#distance = netlsd.compare(descriptor2, descriptor2)
#print("----")

########################
######################## fashionMNIST
# batch_size= 64
# max_samples = 1024
#
# def get_data_loaders(train_batch_size, val_batch_size):
#     fashion_mnist = torchvision.datasets.FashionMNIST(download=True, train=True, root=".").train_data.float()
#
#     data_transform = transforms.Compose([transforms.Resize((224, 224)),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize((fashion_mnist.mean() / 255,),
#                                                               (fashion_mnist.std() / 255,))])
#
#     train_loader = DataLoader(
#         torchvision.datasets.FashionMNIST(download=True, root=".", transform=data_transform, train=True),
#         batch_size=train_batch_size, shuffle=True)
#
#     val_loader = DataLoader(
#         torchvision.datasets.FashionMNIST(download=False, root=".", transform=data_transform, train=False),
#         batch_size=val_batch_size, shuffle=False)
#     return train_loader, val_loader
#
# train_loader, val_loader = get_data_loaders(batch_size, int(0.25* batch_size))
# n_samples = 1000
########################
# #################
#print(data.shape)

#####load cat dog dataset
#set = CatDogDataset('../loads_cats_dogs/Animals_10/', cat_dir='cat/', dog_dir='dog/', samplesize=400) #TODO Specify path of Dataset
# train_set, test_set, train_set_len, test_set_len = split_trainings_data_into_train_and_val(set, batchsize=400) # TODO assert batchsize
# for i, batch in enumerate(train_set, 0): #TODO change that batchsize can be smaller
#     inputs, labels = batch
# X_train = inputs.numpy()
# y_train = labels.numpy()
# for i, batch in enumerate(test_set, 0):
#     testinputs, testlabels = batch
# X_test = testinputs.numpy()
# y_test= testlabels.numpy()
# data = X_train
##################################
#Transf script
data = torchvision.datasets.MNIST(download=True, train=True, root=".").train_data.float() #Goal: augment mnist such that topology changes measurably
data = data.reshape(-1, 28, 28, 1)
print("data shape")
print(data.shape)
labels= torchvision.datasets.MNIST(download=True, train=True, root=".").targets
data = data_aug_for_topology_change(data)

#data =  data[:5] # TODO use data augmentation here to change structure of graph - test whether netlsd captures those changes



data_aug = data.reshape(data.shape[0], data.shape[1]* data.shape[2])  #* data.shape[3])
#######################################
#TDA
# mapper = km.KeplerMapper(verbose=2)
# projected_data = mapper.fit_transform(data_aug, projection=[Isomap(n_components=100, n_jobs=-1) ,umap.UMAP()])
# #cluster data
# graph = mapper.map(
#     projected_data,
#     clusterer=DBSCAN(),
#     cover=km.Cover(35, 0.4),
# )
# print("graph:")
# nx_graph = kmapper.adapter.to_nx(graph)
# print("----------------------")
#
# fileID = "projection_augmented_mnist"
#
# #visualize the graph
# mapper.visualize(graph,
#                  path_html= "mapper_ex_" + fileID + ".html",
#                  title= fileID)
#
# descriptor1= netlsd.heat(nx_graph)
# print(descriptor1)
# ############################
# data2 = torchvision.datasets.MNIST(download=True, train=True, root=".").train_data.float() # compare with MNIST
# print(data2.shape)
# data2 = data2.reshape(data2.shape[0], data2.shape[1]* data2.shape[2])
#
# mapper2 = km.KeplerMapper(verbose=2)
# projected_data2 = mapper2.fit_transform(data2, projection=[Isomap(n_components=100, n_jobs=-1) ,umap.UMAP()])
# #cluster data
# graph2 = mapper2.map(
#     projected_data2,
#     clusterer=DBSCAN(),
#     cover=km.Cover(35, 0.4),
# )
#
# fileID = "projection"
#
# #visualize the graph
# mapper.visualize(graph2,
#                  path_html= "mapper_ex_" + fileID + ".html",
#                  title= fileID)
#
# nx_graph2 = kmapper.adapter.to_nx(graph2)
#
# #descriptor2 = netlsd.heat(projected_data2)
# descriptor2 = netlsd.heat(nx_graph2) # compute the signature
# print(descriptor2)
#
# distance = netlsd.compare(descriptor1, descriptor2) # compare the signatures using l2 distance
# print(distance)
#### Test transferability of modified data set and the actual topological distance proxied by netlsd

path = "../lid_adversarial_subspace_detection-master/data/"
arr= np.load(path + "Adv_mnists_cw-l2.npy") ###load mnist adversarials
label =np.load( path + "Adv_labels_mnist_cw-l2.npy") #load original labels of mnist adversarials
print(arr.shape)
model_file = path + "model_mnist.h5" #load mnist model

#################save actual adv examples
model = load_model(model_file)
arr.reshape(arr.shape[0],1,28,28)
predicted_label = model.predict(arr)


list = []
for i in range(len(predicted_label)):
    if not np.array_equal(label[i], np.rint(predicted_label[i])): # if it is actually an adversarial example
        list.append(arr[i])

n_samples = 1024
original_im= np.load(path + "Original_Image_mnists_cw-l2.npy") # load and show original image if wanted # TODO remove ses
original_im.reshape(arr.shape[0],1,28,28) # take one example adversarial pattern
transfer_arr= list[1] - original_im[1]
#### TODO augmented dataset
testX = data[:10000] # use augmented set as test for transf
testy = labels[:10000].numpy()

for i in range(1, len(arr)): # concatenate trans patterns
    transfer_arr = np.concatenate( (transfer_arr, arr[i] - original_im[i]) , axis=2)
transfer_arr= transfer_arr.reshape(transfer_arr.shape[2], 28,28,1)
inputs_plus_trans= transfer_arr + testX
model2 = load_model(path + 'model_mnist_aug.h5')
predicted_trans_label = model2.predict(inputs_plus_trans) # predictions of all X_test inputs modified with first adversarial example pattern
trans_counter=0
for i in range (len(predicted_trans_label)):
    if testy[i] != np.argmax(predicted_trans_label[i]):
        trans_counter+= 1
print("transferability rate is:")
print(trans_counter / len(predicted_trans_label))






#distance = np.linalg.norm(desc1 - desc2) # equivalent

################dataset + model for adv exs

############load dataset
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data() # class names are not included, need to create them to plot the images
# #class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# # scale the values to a range of 0 to 1 of both data sets
# train_images= np.expand_dims(train_images, 3)
# train_images = tf.image.resize(train_images, [32,32])
# train_images = np.concatenate ((train_images,train_images,train_images), axis=3)
# test_images= np.expand_dims(test_images, 3)
# test_images = tf.image.resize(test_images, [32,32])
# test_images = np.concatenate((test_images, test_images, test_images), axis=3)
#
#
# model = tf.keras.applications.VGG16(
#         include_top=False,
#         weights="imagenet",
#         input_tensor=None,
#         input_shape=(32,32,3),
#         pooling=None
#     )
#
# model.fit(train_images, train_labels, epochs=10, validation_split=0.2)
# test_loss, test_acc = model.evaluate(test_images, test_labels)








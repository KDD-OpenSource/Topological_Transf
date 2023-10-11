## Code for paper "On the Independence of Adversarial Transferability to Topological Changes in the Dataset". 

### Original code base for generating Adversarial Examples: Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality ICLR 2018, https://arxiv.org/abs/1801.02613

#### Dependencies:
python 3.5, tqdm, tensorflow = 1.8, Keras >= 2.0, cleverhans >= 1.0.0 (may need extra change to pass in keras learning rate)

#### Kernal Density and Bayesian Uncertainty are from https://github.com/rfeinman/detecting-adversarial-samples ("Detecting Adversarial Samples from Artifacts" (Feinman et al. 2017))
---------------------------

In main.py, change parameters AMOUNT_PIX_SWAP,EPOCHS,DATASET and attack_used to the parameters of your liking, then run python main.py. Might require trained models which can be obtained by running for example this command for mnist dataset
### 1. Pre-train DNN models:
python train_model.py -d mnist -e 50 -b 128
To craft adversarials, run:
### 2. Craft adversarial examples:
python craft_adv_samples.py -d cifar -a cw-l2 -b 120

----------------------------------------
main.py will test how attacks generated on the original model will transfer, but will also check the transferability for adversarials generated from the model trained on the topologically altered dataset then applied to the original model. (Both transfer directions)

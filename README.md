# CorrNet

This is an implementation of Correlational Neural Network (CorrNet) described in the following paper :

Sarath Chandar, Mitesh M Khapra, Hugo Larochelle, Balaraman Ravindran. [Correlational Neural Networks](http://arxiv.org/abs/1504.07225). To appear in Neural Computation, 2015.

Please site this paper if you are using this code for any of your publications.

## Dependencies

To run the representation learning code, you need Python and Theano.

To run the MNIST example, you need scikit-learn.

## Running MNIST example

Refer section 5 in the paper for details about the two-view setup for MNIST and the transfer learning experiment. First you should download the dataset from [this link](https://drive.google.com/file/d/0B9dqzboiV5u-OW1GMW5mVG1UYzg/view?usp=sharing) and extract all the files to some directory say MNIST_DIR. You also need a target directory where the models will be saved, say TGT_DIR.

In terminal, go to mnistExample folder.

To create the dataset for training, run the following command:

```
$ python create_data.py MNIST_DIR/
```

Next, to train the CorrNet, run the following command.

```
$ python train_corrnet.py MNIST_DIR/ TGT_DIR/
```

To project the data to the learnt space, run the following command.

```
$ python project_corrnet.py MNIST_DIR/ TGT_DIR/
```
To evaluate the learnt model for transfer learning task, run the following command.

```
$ python evaluate.py tl TGT_DIR/
```

With batch_size=100, training_epochs=50, l_rate=0.01, optimization="rmsprop", tied=True, n_hidden=50, lambda=2, hidden_activation=sigmoid, output_activation=sigmoid, loss_fn = "squarrederror", you should get 77.05% accuracy for view1 to view2 and 78.81% accuracy for view2 to view1.

To compute sum correlation in the projected space, run the following command.

```
$ python evaluate.py corr TGT_DIR/
```

With the same configuration as above, you should get 42.57 as test correlation.

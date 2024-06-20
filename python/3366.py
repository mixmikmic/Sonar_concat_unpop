# # Convolutional Neural Networks
# 
# ### Need for Convolution Neural Network
# In the previous tutorials we saw learned about the Feed Forward Neural Networks where each neuron in a network learns from every neuron in the previous layer. This architecture of the network assumes independence between the features of the data. This helps in learning the abstract global representation. However, this architure cannot learn local representations of the data because there is no interaction between the neighboring neurons. For Example we are analyzing an image or a video or a spech sequence we need to learn the local representations of the and build on those local representations to form a global representation.
# 
# ### Convolution and Pooling layer
# 
# We attain this function by imposing sparsity on the layers such that most of the weight are zeros exept few. We attain sparcity by forcig each neuron to learn only from new neighboring inputs. Let $k$ be number of neighnors we allow a neuron to learn from. If we take $k=3$ then each neuron only get input from the the 3 previous neurons. Forcibly reducing the inputs of the neuron to 3 will allow the neuron to learn the local representations of the image. For an image case that initial convolution layer learns small edges from the image. Another advantage of this technique is good old reusuability. Once is neuron is trained to learn an edge from an image, we can use the same neuron to detect the edge wherever it is present in the image by moving it around the entire image. We can see this in two ways we can think of this as a single neuron moving around the image to detect the features or many neurons sharing the same weights. This operaiton is called convolution.
# 
# Let's look at it mathematically. Let  $x = [x^{(1)}, x^{(2)},......,x^{(d)}]$ d dimensional input vector and $W = [w_1, w_2,.......,w_3]$ be the weight vector of the neuron. This operation can be mathematically represented as 
# $$z^{(j)} = \sum_{i=1}^{k}x^{j+i-1}W_k, \forall j=[1,2,…,𝑑−𝑘+1] $$
# 
# #### Receptive Field
# 
# The value k we used before that represents the number of inputs the neuron learns from can be seen as the number of signals it recieves from the previous layer. Therefore, this k is called the receptive field. It can also be viewed in a different way. If k is 3 then the neurons receptive field is 3 from the previous layer. What about the neuron's receptive field with respect to the layer before that? The previous layer has each neuron recieved from 3 inputs from the layer before. Thefore the current neuron recieves 9 inputs from the 2 layers before.
# 
# #### Pooling Operation
# 
# From the previous section we learned that each neuron recieves input from k neighboring k values in the input and produces one ouput. As the neuron moves in the entire layer, it produces d-k+1 outputs where d is the number of dimensions in the input. Each neuron produces d-k+1 oututs, so if we have n neurons in each layer then we have $n*(d-k+1)$ outputs. For Example for a 20 neuron layer, if the input is 100 dimensional, for k=3 we have 1960 outputs. from 100 to 1960 is a huge jump and this keeps on increasing and will result in a computational instability. To prevent that we have a special operation called pooling. We downsample the entire output space into lower dimension by taking a single value for a given pooling length. depending on the value we consider we call it either as avg pooling or max pooling. 
# 
# #### Stride
# 
# We can reduce the number of outputs by using strides without using pooling. A stride operation is equivalent to a hop. Stride is the number of hops we make from convolution to convolution. We can use longer strides where we don't want outputs to read from common inputs.
# 
# ### Convolution layer in YANN
# We can add a convolution layer to our network using add_layer function with the following syntax:
# <code> net.add_layer ( type = "conv_pool",
#                 origin = "input",
#                 id = "conv_pool_1",
#                 num_neurons = 
#                 filter_size = (,),
#                 pool_size = (,),
#                 activation = ('maxout', 'maxout', 2),
#                 batch_norm = True,
#                 regularize = True,
#                 verbose = verbose
#             )
#             </code>
# 

from yann.network import network
from yann.utils.graph import draw_network
from yann.special.datasets import cook_mnist
def lenet5 ( dataset= None, verbose = 1, regularization = None ):             
    """
    This function is a demo example of lenet5 from the infamous paper by Yann LeCun. 
    This is an example code.  
    
    Warning:
        This is not the exact implementation but a modern re-incarnation.

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.65, 0.97, 30),      
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                        }

    dataset_params  = {
                            "dataset"   : dataset,
                            "svm"       : False, 
                            "n_classes" : 10,
                            "id"        : 'data'
                      }

    visualizer_params = {
                    "root"       : 'lenet5',
                    "frequency"  : 1,
                    "sample_size": 144,
                    "rgb_filters": True,
                    "debug_functions" : False,
                    "debug_layers": False,  # Since we are on steroids this time, print everything.
                    "id"         : 'main'
                        }       

    # intitialize the network
    net = network(   borrow = True,
                     verbose = verbose )                       
    
    # or you can add modules after you create the net.
    net.add_module ( type = 'optimizer',
                     params = optimizer_params, 
                     verbose = verbose )

    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )

    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    )
    # add an input layer 
    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose, 
                    datastream_origin = 'data', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    mean_subtract = False )
    
    # add first convolutional layer
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv_pool_1",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (3,3),
                    activation = ('maxout', 'maxout', 2),
                    # regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_1",
                    id = "conv_pool_2",
                    num_neurons = 50,
                    filter_size = (3,3),
                    pool_size = (1,1),
                    activation = 'relu',
                    # regularize = True,
                    verbose = verbose
                    )      


    net.add_layer ( type = "dot_product",
                    origin = "conv_pool_2",
                    id = "dot_product_1",
                    num_neurons = 1250,
                    activation = 'relu',
                    # regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "dot_product_1",
                    id = "dot_product_2",
                    num_neurons = 1250,                    
                    activation = 'relu',  
                    # regularize = True,    
                    verbose = verbose
                    ) 
    
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "dot_product_2",
                    num_classes = 10,
                    # regularize = True,
                    activation = 'softmax',
                    verbose = verbose
                    )

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "softmax",
                    objective = "nll",
                    datastream_origin = 'data', 
                    regularization = regularization,                
                    verbose = verbose
                    )
                    
    learning_rates = (0.05, .0001, 0.001)  
    net.pretty_print()  
    draw_network(net.graph, filename = 'lenet.png')   

    net.cook()

    net.train( epochs = (20, 20), 
               validate_after_epochs = 1,
               training_accuracy = True,
               learning_rates = learning_rates,               
               show_progress = True,
               early_terminate = True,
               patience = 2,
               verbose = verbose)

    print(net.test(verbose = verbose))
data = cook_mnist()
dataset = data.dataset_location()
lenet5 ( dataset, verbose = 2)


# # Auto Encoders
# Auto Encoders are unsupervised learning technique where we conert an image into code word and the image can be regenerated from the code word. Auto encoder is a combination of two convolutional networks, The encoder and The decoder. Encoder is a convolutional network which gives the code word at the end. decoder is a network with deconvolution layers and it converts the codeword to image by using deconvolution layers. Deconvolution layers have the transposed weights of convolution layers and they use fractional striding to give the effect of unpooling. 
# 
# An example Auto Encoder can be seen below.
# 

from yann.network import network
def convolutional_autoencoder ( dataset= None, verbose = 1 ):
    """
    This function is a demo example of a deep convolutional autoencoder. 
    This is an example code. You should study this code rather than merely run it.  
    This is also an example for using the deconvolutional layer or the transposed fractional stride
    convolutional layers.

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    dataset_params  = {
                            "dataset"   : dataset,
                            "type"      : 'x',
                            "id"        : 'data'
                    }

    visualizer_params = {
                    "root"       : '.',
                    "frequency"  : 1,
                    "sample_size": 32,
                    "rgb_filters": False,
                    "debug_functions" : False,
                    "debug_layers": True,  
                    "id"         : 'main'
                        }  
                      
    # intitialize the network    
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.65, 0.95, 30),      
                "regularization"      : (0.0001, 0.0001),       
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                    }
    net = network(   borrow = True,
                     verbose = verbose )                       

    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )
    
    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    ) 
    net.add_module ( type = 'optimizer',
                     params = optimizer_params,
                     verbose = verbose )
    # add an input layer 
    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose, 
                    origin = 'data', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    mean_subtract = True )

    
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (1,1),
                    activation = 'tanh',
                    regularize = True,   
                    #stride = (2,2),                          
                    verbose = verbose
                    )

    net.add_layer ( type = "flatten",
                    origin = "conv",
                    id = "flatten",
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "flatten",
                    id = "hidden-encoder",
                    num_neurons = 1200,
                    activation = 'tanh',
                    dropout_rate = 0.5,                    
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "hidden-encoder",
                    id = "encoder",
                    num_neurons = 128,
                    activation = 'tanh',
                    dropout_rate = 0.5,                        
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "encoder",
                    id = "decoder",
                    num_neurons = 1200,
                    activation = 'tanh',
                    input_params = [net.dropout_layers['encoder'].w.T, None],
                    # Use the same weights but transposed for decoder. 
                    learnable = False,                    
                    # because we don't want to learn the weights of somehting already used in 
                    # an optimizer, when reusing the weights, always use learnable as False   
                    dropout_rate = 0.5,                                         
                    verbose = verbose
                    )           

    net.add_layer ( type = "dot_product",
                    origin = "decoder",
                    id = "hidden-decoder",
                    num_neurons = net.layers['flatten'].output_shape[1],
                    activation = 'tanh',
                    input_params = [net.dropout_layers['hidden-encoder'].w.T, None],
                    # Use the same weights but transposed for decoder. 
                    learnable = False,                    
                    # because we don't want to learn the weights of somehting already used in 
                    # an optimizer, when reusing the weights, always use learnable as False    
                    dropout_rate = 0.5,                                        
                    verbose = verbose
                    )                                            

    net.add_layer ( type = "unflatten",
                    origin = "hidden-decoder",
                    id = "unflatten",
                    shape = (net.layers['conv'].output_shape[2],
                             net.layers['conv'].output_shape[3],
                             20),
                    verbose = verbose
                    )

    net.add_layer ( type = "deconv",
                    origin = "unflatten",
                    id = "deconv",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (1,1),
                    output_shape = (28,28,1),
                    activation = 'tanh',
                    input_params = [net.dropout_layers['conv'].w, None],        
                    learnable = False,              
                    #stride = (2,2),
                    verbose = verbose
                    )

    # We still need to learn the newly created biases in the decoder layer, so add them to the 
    # Learnable parameters list before cooking

    net.active_params.append(net.dropout_layers['hidden-decoder'].b)
    net.active_params.append(net.dropout_layers['decoder'].b)    
    net.active_params.append(net.dropout_layers['deconv'].b)
    

    net.add_layer ( type = "merge",
                    origin = ("input","deconv"),
                    id = "merge",
                    layer_type = "error",
                    error = "rmse",
                    verbose = verbose)

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "merge", # this is useless anyway.
                    layer_type = 'value',
                    objective = net.layers['merge'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )          

    learning_rates = (0.04, 0.0001, 0.00001)  
    net.cook( objective_layers = ['obj'],
              datastream = 'data',
              learning_rates = learning_rates,
              verbose = verbose
              )

    # from yann.utils.graph import draw_network
    # draw_network(net.graph, filename = 'autoencoder.png')    
    net.pretty_print()
    net.train( epochs = (10, 10), 
               validate_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = True,
               verbose = verbose)

if __name__ == '__main__':
    import sys
    print " creating a new dataset to run through"
    from yann.special.datasets import cook_mnist_normalized_zero_mean as cook_mnist  
    data = cook_mnist (verbose = 2)
    dataset = data.dataset_location()

    convolutional_autoencoder ( dataset , verbose = 2 )


# # Multi-layer Neural Network
# By virture of being here, it is assumed that you have gone through the [Quick Start](http://yann.readthedocs.io/en/master/index.html#quick-start). To recap the Quicks tart tutorial, We imported MNIST dataset and trained a  Logistic Regression  which produces a linear classification boundary. It is impossible to learn complex functions like XOR with linear classification boundary.
# 
# A Neural Network is a function approximator consisting of several neurons organized in a layered fashion. Each neuron takes input from previous layer, performs some mathematical calculation and sends output to next layer. A neuron produces output only if the result of the calculation it performs is greater than some threshold. This threshold function is called activation function. Depending on the type of the task different activation functions can be used. Some of the most commonly used activation functions are sigmoid, tanh, ReLu and maxout. It is inspired from the functioning of human brain where one neuron sends signal to other neuron only if the electical signal in the first neuron is greater than some threshold.
# 
# A Feed Forward Neural network/ multi-layer perceptron has an input layer, an output layer and some hidden layers. The actual magic of the neural networks happens in the hidden layers and they represent the function the network is trying to approximate. Output layer is generally a softmax function that converts the inputs into probabilities.  Let us look at the mathematical representation of the hidden layer and output layer
# 
# #### Hidden layer:
# let $[a_{i-1}^1], a_{i-1}^2, a_{i-1}^3 ........ a_{i-1}^n]$ be the activations of the previous layer $i-1$
# $$h_i = w_i^0 + w_i^1a_{i-1}^1 + w_i^2a_{i-1}^2 + ...... + w_i^na_{i-1}^n$$
# $$a_i = act(h_i)$$
#  Where i is the layer number,
#        $[w_i^1, w_i^2, w_i^3, ......... w_i^n]$ be the parameters between the $i^{th}$ and $(i-1)^{th}$ layer, $w_i^0$ is the bias which is the input when there is no activation from the previous layer,
#        1,2....n are the dimensions of the layer,
#        $a_i$ is the activation at the layer, and $act()$ is the activation function for that layer. 
#        
# #### Output layer:
# let our network has l layers
# $$z = w_i^0 + w_i^1a_{i-1}^1 + w_i^2a_{i-1}^2 + ...... + w_i^na_{i-1}^n$$
# $$a = softmax(z)$$
# $$correct class = argmax(a)$$
# 
# Where a represents the output probabilities, z represents the weighted activations of the previous layer.
# 
# ### Neural Network training:-
# Neural Network has a lot of parameters to learn. Consider a neural network with 2 layers of each 100 neurons and input dimension of 1024 and 10 outputs. Then the number of parameters to learn is 1024 * 100 * 100 * 10 i.e., 102400000 parameters. Learning these many parameters is a complex task because for each parameter we need to calculate the gradient of error function and update the parameters with that gradient. The computational instability of this process is the reason for neural networks to loose it's charm quickly. There is a technique called Back propagation that solved this problem. The following section gives a brief insight into the backpropagation technique.
# 
# ## Back Propagation:
# YANN handles the Back propagation by itself. But, it does not hurt to know how it works. A neural network can be represented mathematically as $$O = f_1(W_l(f_2(W_{l-1}f_3(..f_n(WX)..)))$$ where $f_1, f_2, f_3$ are activation functions. 
# An Error function can be represented as $$E(f_1(W_l(f_2(W_{l-1}f_3(..f_n(WX)..))))$$ where $E()$ is some error function. The gradient of $W_l$ is given by:
# 
# $$g_l = \frac{\partial E(f_1(W_lf_2(W_{l-1}f_3(..f_n(WX)..))))}{\partial W_l} $$
# Applying chain rule:
# $$g_l = \frac{\partial E(f_1())}{\partial f_1}\frac{\partial f_1}{\partial W_l}
# $$
# The gradient of error w.r.t $W_{l-1}$ after applying chain rule:
# $$g_l = \frac{\partial E(f_1())}{\partial f_1}\frac{\partial f_1(W_lf_2())}{\partial f_2}\frac{\partial f_2()}{\partial W_2}
# $$
# 
# In the above equations the first term $\frac{\partial E(f_1())}{\partial f_1}$ remains same for both gradients. Similarly for rest of the parameters we reuse the terms from the previous gradient calculation. This process drastically reduces the number of calculations in Neural Network training.  
# 
# Let us take this one step further and create a neural network with two hidden layers. We begin as usual by importing the network class and creating the input layer.
# 

from yann.network import network
from yann.special.datasets import cook_mnist

data = cook_mnist()
dataset_params  = { "dataset": data.dataset_location(), "id": 'mnist', "n_classes" : 10 }

net = network()
net.add_layer(type = "input", id ="input", dataset_init_args = dataset_params)


# In Instead of connecting this to a classfier as we saw in the [Quick Start](http://yann.readthedocs.io/en/master/index.html#quick-start) , let us add a couple of fully connected hidden layers. Hidden layers can be created using layer type = dot_product.
# 

net.add_layer (type = "dot_product",
               origin ="input",
               id = "dot_product_1",
               num_neurons = 800,
               regularize = True,
               activation ='relu')

net.add_layer (type = "dot_product",
               origin ="dot_product_1",
               id = "dot_product_2",
               num_neurons = 800,
               regularize = True,
               activation ='relu')


# Notice the parameters passed. ***num_neurons*** is the number of nodes in the layer. Notice also how we modularized the layers by using the ***id*** parameter. ***origin*** represents which layer will be the input to the new layer. By default yann assumes all layers are input serially and chooses the last added layer to be the input. Using ***origin***, one can create various types of architectures. Infact any directed acyclic graphs (DAGs) that could be hand-drawn could be implemented. Let us now add a classifier and an objective layer to this.
# 

net.add_layer ( type = "classifier",
                id = "softmax",
                origin = "dot_product_2",
                num_classes = 10,
                activation = 'softmax',
                )

net.add_layer ( type = "objective",
                id = "nll",
                origin = "softmax",
                )


# The following block is something we did not use in the Quick Start tutorial. We are adding optimizer and optimizer parameters to the network. Let us create our own optimizer module this time instead of using the yann default. For any ***module*** in yann, the initialization can be done using the ***`add_module`*** method. The ***`add_module`*** method typically takes input ***`type`*** which in this case is ***`optimizer`*** and a set of intitliazation parameters which in our case is ***`params = optimizer_params`***. Any module params, which in this case is the ***`optimizer_params`*** is a dictionary of relevant options. If you are not familiar with the optimizers in neural network, I would suggest you to go through the [Optimizers to Neural network](./Optimization%20for%20Neural%20networks.ipynb) series of tutorials to get familiar with the effect of differnt optimizers in a Nueral Network.
# 
# A typical ***`optimizer setup`*** is:
# 

optimizer_params =  {
            "momentum_type"       : 'polyak',
            "momentum_params"     : (0.9, 0.95, 30),
            "regularization"      : (0.0001, 0.0002),
            "optimizer_type"      : 'rmsprop',
            "id"                  : 'polyak-rms'
                    }
net.add_module ( type = 'optimizer', params = optimizer_params )


# We have now successfully added a Polyak momentum with RmsProp back propagation with some  and  co-efficients that will be applied to the layers for which we passed as argument ***`regularize = True`***. For more options of parameters on optimizer refer to the [optimizer documentation](http://yann.readthedocs.io/en/master/yann/modules/optimizer.html) . This optimizer will therefore solve the following error:
# 
# where  is the error,  is the sigmoid layer and  is the ith layer of the network.  
# 

learning_rates = (0.05, 0.01, 0.001)


# The ***`learning_rate`***, supplied here is a tuple. The first indicates a annealing of a linear rate, the second is the initial learning rate of the first era, and the third value is the leanring rate of the second era. Accordingly, epochs takes in a tuple with number of ***`epochs`*** for each era.
# 
# Noe we can cook, train and test as usual:
# 

net.cook( optimizer = 'polyak-rms',
          objective_layer = 'nll',
          datastream = 'mnist',
          classifier = 'softmax',
          )

net.train( epochs = (20, 20),
           validate_after_epochs = 2,
           training_accuracy = True,
           learning_rates = learning_rates,
           show_progress = True,
           early_terminate = True)


# This time, let us not let it run the forty epochs, let us cancel in the middle after some epochs by hitting ^c. Once it stops lets immediately test and demonstrate that the ***`net`*** retains the parameters as updated as possible. 
# Some new arguments are introduced here and they are for the most part easy to understand in context. ***`epoch`*** represents a ***`tuple`*** which is the number of epochs of training and number of epochs of fine tuning epochs after that. There could be several of these stages of finer tuning. Yann uses the term ‘era’ to represent each set of epochs running with one learning rate. ***`show_progress`*** will print a progress bar for each epoch. ***`validate_after_epochs`*** will perform validation after such many epochs on a different validation dataset. 
# 
# Once done, lets run ***`net.test()`***:-
# 

net.test()


# The full code for this tutorial with additional commentary can be found in the file ***`pantry.tutorials.mlp.py`***. If you have toolbox cloned or downloaded or just the tutorials downloaded, Run the code as,
# 

from yann.pantry.tutorials.mlp import mlp
mlp(dataset = data.dataset_location())


# or simply ,
# <pre><code>python pantry/tutorials/mlp.py </code></pre>
# 
# from the toolbox root or path added to toolbox. The ***`__init__`*** program has all the required tools to create or load an already created dataset. Optionally as command line argument you can provide the location to the dataset.
# 







# # Generative Adversarial Networks(GAN)
# GAN is one of the areas in the Neural Networks with a very fast pace of reasearch. Every week there is new GAN. To explain the concept of GAN, let's use a small anecdote to stage this concept. In old movies to sketch a criminal there will be an artist and a witness. Witness tells artist some details and witness validates his art and says if it is correct or not. If the imageis not similar to the criminal, artist will redraw it again with further changes. This process will be repeated until artist produces an image which is accepted by the witness. In other words witness unable to differentiate the artists imaginary art from the crimial. At this point they stop.
# 
# GAN works similar to this idea. We have a generator network that generates random images and a Descriminator network that clssifies whether that image is fake or real. If the image is fake the descriminator discards the image and if image is real, it accepts it. This process continues until generator generates all real images. The generator is a decoder network from the autoencoder we discussed in the tutorial before. We take a random codeword and we pass it to the generator network to generate image. We take that generated image and feed it to descriminator to tell if it is a real or fake image. To achieve that we always keep our descriminator a step ahead.
# 
# The following code shows the implementation of GAN using YANN:
# For GAN in YANN we need to use the yann.special.gan package which has similar functionalities like a network.
# 

from yann.special.gan import gan 
from theano import tensor as T 

def shallow_gan_mnist ( dataset= None, verbose = 1 ):
    """
    This function is a demo example of a generative adversarial network. 
    This is an example code. You should study this code rather than merely run it.  

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.

    Notes:
        This method is setup for MNIST.
    """
    optimizer_params =  {        
                "momentum_type"       : 'polyak',             
                "momentum_params"     : (0.65, 0.9, 50),      
                "regularization"      : (0.000, 0.000),       
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                        }


    dataset_params  = {
                            "dataset"   : dataset,
                            "type"      : 'xy',
                            "id"        : 'data'
                    }

    visualizer_params = {
                    "root"       : '.',
                    "frequency"  : 1,
                    "sample_size": 225,
                    "rgb_filters": False,
                    "debug_functions" : False,
                    "debug_layers": True,  
                    "id"         : 'main'
                        }  
                      
    # intitialize the network
    net = gan (      borrow = True,
                     verbose = verbose )                       
    
    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )    
    
    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    ) 

    #z - latent space created by random layer
    net.add_layer(type = 'random',
                        id = 'z',
                        num_neurons = (100,32), 
                        distribution = 'normal',
                        mu = 0,
                        sigma = 1,
                        verbose = verbose)
    
    #x - inputs come from dataset 1 X 784
    net.add_layer ( type = "input",
                    id = "x",
                    verbose = verbose, 
                    datastream_origin = 'data', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    mean_subtract = False )

    net.add_layer ( type = "dot_product",
                    origin = "z",
                    id = "G(z)",
                    num_neurons = 784,
                    activation = 'tanh',
                    verbose = verbose
                    )  # This layer is the one that creates the images.
        
    #D(x) - Contains params theta_d creates features 1 X 800. 
    net.add_layer ( type = "dot_product",
                    id = "D(x)",
                    origin = "x",
                    num_neurons = 800,
                    activation = 'relu',
                    regularize = True,                                                         
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    id = "D(G(z))",
                    origin = "G(z)",
                    input_params = net.dropout_layers["D(x)"].params, 
                    num_neurons = 800,
                    activation = 'relu',
                    regularize = True,
                    verbose = verbose
                    )


    #C(D(x)) - This is the opposite of C(D(G(z))), real
    net.add_layer ( type = "dot_product",
                    id = "real",
                    origin = "D(x)",
                    num_neurons = 1,
                    activation = 'sigmoid',
                    verbose = verbose
                    )

    #C(D(G(z))) fake - the classifier for fake/real that always predicts fake 
    net.add_layer ( type = "dot_product",
                    id = "fake",
                    origin = "D(G(z))",
                    num_neurons = 1,
                    activation = 'sigmoid',
                    input_params = net.dropout_layers["real"].params, # Again share their parameters                    
                    verbose = verbose
                    )

    
    #C(D(x)) - This is the opposite of C(D(G(z))), real
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "D(x)",
                    num_classes = 10,
                    activation = 'softmax',
                    verbose = verbose
                   )
    
    # objective layers 
    # discriminator objective 
    net.add_layer (type = "tensor",
                    input =  - 0.5 * T.mean(T.log(net.layers['real'].output)) - \
                                  0.5 * T.mean(T.log(1-net.layers['fake'].output)),
                    input_shape = (1,),
                    id = "discriminator_task"
                    )

    net.add_layer ( type = "objective",
                    id = "discriminator_obj",
                    origin = "discriminator_task",
                    layer_type = 'value',
                    objective = net.dropout_layers['discriminator_task'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )
    #generator objective 
    net.add_layer (type = "tensor",
                    input =  - 0.5 * T.mean(T.log(net.layers['fake'].output)),
                    input_shape = (1,),
                    id = "objective_task"
                    )
    net.add_layer ( type = "objective",
                    id = "generator_obj",
                    layer_type = 'value',
                    origin = "objective_task",
                    objective = net.dropout_layers['objective_task'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )   

    #softmax objective.    
    net.add_layer ( type = "objective",
                    id = "classifier_obj",
                    origin = "softmax",
                    objective = "nll",
                    layer_type = 'discriminator',
                    datastream_origin = 'data', 
                    verbose = verbose
                    )
    
    from yann.utils.graph import draw_network
    draw_network(net.graph, filename = 'gan.png')    
    net.pretty_print()
    
    net.cook (  objective_layers = ["classifier_obj", "discriminator_obj", "generator_obj"],
                optimizer_params = optimizer_params,
                discriminator_layers = ["D(x)"],
                generator_layers = ["G(z)"], 
                classifier_layers = ["D(x)", "softmax"],                                                
                softmax_layer = "softmax",
                game_layers = ("fake", "real"),
                verbose = verbose )
                    
    learning_rates = (0.05, 0.01 )  

    net.train( epochs = (20), 
               k = 2,  
               pre_train_discriminator = 3,
               validate_after_epochs = 1,
               visualize_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = True,
               verbose = verbose)
                           
    return net

if __name__ == '__main__':
    
    from yann.special.datasets import cook_mnist_normalized_zero_mean as c 
    # from yann.special.datasets import cook_cifar10_normalized_zero_mean as c
    print " creating a new dataset to run through"
    data = c (verbose = 2)
    dataset = data.dataset_location() 

    net = shallow_gan_mnist ( dataset, verbose = 2 )


# ## Momentum:
# 
# ### Classical / Polyak momentum
# We discussed about the importance of learning rates and need for annealing the learning rates with Stochastic Gradient Descent. SGD is a convex optimization technique and it converges to stationary point. A stationary point may not be a convergence point for every problem. A solution space with sparse gradients will have a type of stationary points called saddle points. Saddle points are points with very low gradient or near zero gradient for few iterations and high gradients after that. 
# 
# This problem can be solved by updating parameters with a historical average of the gradients instead of the current gradient. Momentum update is calculated by taking a portion of the previous momentum and subtracting the current gradient (scaled by learning rate) from it. an equation for regular momentum update is shown below.
# $$v_{t+1} = \mu v_t - \eta f'(\theta _t)$$
# $$\theta _{t+1} = \theta _t + v_{t+1}$$
# 
# $v_t$ is the momentum update in the previous time step and $\theta _t$ is the current gradient. $\eta$ is the learning rate and $\mu$ is the momentum coefficient. If you consider previous momentum as a vector and currect gradient as another vector, we are adjusting the direction of the momentum vector using the current gradient.
# 
# ### Nestrov Accelerated Gradient(NAG)
# Nestrov Accelerated Gradient/Nestrov Momentum is another type of momentum update which is highly used. In polyak momentum we calculated the gradient and then adjusted the momentum with the gradient. However in Nestrov momentum we adjust the gradient before calculating the momentum. Nestrov Momentum can be respresented mathematically using the following equations:
# $$v_{t+1} = \mu v_t - \eta f'(\theta _t + \mu v_t)$$
# $$\theta _{t+1} = \theta _t + v_{t+1}$$
# 
# ## Momentum in YANN:
# YANN supports Polyak and Nestrov momentum. Momentum can be added to the network using the following
# 
# optimizer_params =  {         
# 
#         "momentum_type"   : <option>  'false' <no momentum>, 'polyak', 'nesterov'.
#         "momentum_params" : (<option in range [0,1]>, <option in range [0,1]>, <int>)
#         "optimizer_type" : <option>, 'sgd', 'adagrad', 'rmsprop', 'adam'.
#         "id"        : id of the optimizer
#             } 
# ***`momentum_type`*** is the type of momentum(polyak/nestrov) to be used. If you don't want to use any momentum you can give false to it. The default value for momentum_type is false. ***`momentum_params`*** is a 3 tuple and takes values for momentum coeffient at start,at end, at what epoch to end momentum increase and it takes a default value of (0.5, 0.95,50). ***`Optimizer_type`*** is the type of the optimizer to be used. YANN supports sgd, adagrad, rmsprop and adam. We discussed about sgd in the previous tutorial and we will discuss other optimizers later in this chapter.
# 
# Let's use the MLP network we created in the previous tutorial and train it with no momentum, polak momentum and nestrov momentum and analyze the results
# 

from yann.network import network
from yann.special.datasets import cook_mnist
import matplotlib.pyplot as plt

def get_cost():
    costs = []
    with open('./resultor/costs.txt') as costf:
        costs = [float(cost.rstrip()) for cost in costf]
    return costs

def plot_costs(costs, labels):
    for cost, label in zip(costs, labels):
        plt.plot(cost,label=label)
    plt.legend()
    plt.show()
    
costs = []

data = cook_mnist()
dataset_params  = { "dataset": data.dataset_location(), "id": 'mnist', "n_classes" : 10 }
def mlp(dataset_params, optimizer_params, optimizer_id):
    net = network()
    net.add_layer(type = "input", id ="input", dataset_init_args = dataset_params)
    net.add_layer (type = "dot_product",
                   origin ="input",
                   id = "dot_product_1",
                   num_neurons = 800,
                   regularize = True,
                   activation ='relu')

    net.add_layer (type = "dot_product",
                   origin ="dot_product_1",
                   id = "dot_product_2",
                   num_neurons = 800,
                   regularize = True,
                   activation ='relu')
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "dot_product_2",
                    num_classes = 10,
                    activation = 'softmax',
                    )

    net.add_layer ( type = "objective",
                    id = "nll",
                    origin = "softmax",
                    )
    
    net.add_module ( type = 'optimizer', params = optimizer_params )
    learning_rates = (0.05, 0.01, 0.001)
    net.cook( verbose = 0,
             optimizer = optimizer_id,
              objective_layer = 'nll',
              datastream = 'mnist',
              classifier = 'softmax',
              )
    net.train(verbose=0,
              epochs = (20, 20),
           validate_after_epochs = 2,
           training_accuracy = True,
           learning_rates = learning_rates,
           show_progress = True,
           early_terminate = True)
    return net


# The above function returns a trained network given the dataset_params and optimizer_params. Let's train this network without any momentum
# 

optimizer_params =  {
                "momentum_type"       : 'false',
                "momentum_params"     : (0.9, 0.95, 30),
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'sgd',
                "id"                  : 'sgd'
                        }
net = mlp(dataset_params, optimizer_params, 'sgd')
costs.append(get_cost())


labels = ['no momentum']
plot_costs(costs, labels)


# Let's train the same network with polyak momentum
# 

optimizer_params =  {
                "momentum_type"       : 'polyak',
                "momentum_params"     : (0.9, 0.95, 30),
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'sgd',
                "id"                  : 'sgd-polyak'
                        }
net = mlp(dataset_params, optimizer_params, 'sgd-polyak')
costs.append(get_cost())


labels = ['no momentum', 'polyak']
plot_costs(costs, labels)


# Let's train the same network with Nestrov momentum
# 

optimizer_params =  {
                "momentum_type"       : 'nesterov',
                "momentum_params"     : (0.9, 0.95, 30),
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'sgd',
                "id"                  : 'sgd-nesterov'
                        }
net = mlp(dataset_params, optimizer_params, 'sgd-nesterov')
costs.append(get_cost())


labels = ['no momentum', 'polyak', 'nesterov']
plot_costs(costs, labels)


# If you see the plot above you can observe that polyak and nesterov momentum helped to converge faster compared to the one with no momentum. In particular example, polyak and nesterov performed similarly. Hoewever in general nesterov performs better and is more stable.
# 

# #### Need for a new optimizer:
# Momentum solves the saddle point problem by adding a temporal average of the previous updates. One problem with momentum is that it accelerates the gradients for every direction(dimension). Imagine a scenario where we need have the minimum in the direction where we have less gradients and high gradients in other direction. Using momentum makes it faster in other direction and it maintains the slow pace in the direction we actually need to move.   
# 
# ## Adagrad
# 
# Adagrad is another optimizer we are borrowing from convex optimization. Adagrad promises convergence given a convex setting. We have learnt few optimization techniques before. If you recall, we used same learning rate for all parameters. Adagrad adapts the learning rate per parameter. Parameter update in Adagrad is done using the following equation.
# $$\theta _{j} = \theta _j +  \frac{\eta}{\sqrt{G_{j,j}}} g_j$$
# 
# The above equation has only one new component $\sqrt{G_{j,j}}$. $G_{j,j}$ is the sum of the squares of the previous gradients. This term acts as a smoothing factor.  If the gradient is constantly high in a direction then $G_{j,j}$ is also high which reduces the overall learning rate in that direction and vice versa.
# 
# Let's try the previous example using Adagrad.
# 

optimizer_params =  {
                "momentum_type"       : 'false',
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'adagrad',
                "id"                  : 'adagrad'
                        }
net = mlp(dataset_params, optimizer_params, 'adagrad')
costs.append(get_cost())


labels = ['sgd','sgd-polyak','sgd-nestrov', 'adagrad']
plot_costs(costs, labels)


# The above plot shows a comparison between sgd(with momentum) and adagrad where adagrad converged significantly faster compared to other algorithms. You can use polyak and nestrov momentum with Agadrad.
# 
# #### Problem with Adagrad:
# 
# We saw that the Agadrad got all those nice features from the denominator term $\sqrt{G_{j,j}}$. $G_{j,j}$. However, this term increases with time consequently, learning rate reduces with time. It can reach a point where through the gradient is high, low learning rate can stop the learning. In simple term Adagrad forces the convergence. Therefore this technique is not immune to Saddle point problem. Let's look at the most popular neural network training technique.
# 
# ## RMSPROP
# 
# rmsprop algorithm is proposed by Hinton which solves teh adagrad problem of forced convergence by taking weighted average of historical gradients so that the current update is only impacted by the past few gradients unlike every past gradient in Adagrad.
# $$H_t = \gamma H_{t-1} + (1-\gamma )g_j^2$$
# $$\theta _{j} = \theta _j +  \frac{\eta}{\sqrt{H_t}} g_j$$
# In the above equations $\gamma$ is called the forgetting factor because scales down the $H_{t-1}$, the historical gradient. As the historical gradient is scaled down using this forgetting factor it is immune to saddle points. As rmsprop implements weighted historic average it is also not affected much by sudden changes in the gradient. Therefore it is works well with on-line and non-stationary optimization settings.
# 
# Let's see rmsprop in action:
# 

optimizer_params =  {
                "momentum_type"       : 'false',
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'rmsprop',
                "id"                  : 'rmsprop'
                        }
net = mlp(dataset_params, optimizer_params, 'rmsprop')
costs.append(get_cost())


labels = ['sgd','sgd-polyak','sgd-nestrov', 'adagrad','rmsprop']
plot_costs(costs, labels)


# Above plot shows that rmsprop reaches convergence marginally faster compared to adagrad. 
# 
# ##### Need for a new Optimizer
# We have rmsprop that works well in on-line and non-stationary optimization setting ang we have adagrad that works well with sparse gradients. What if we have a single algorithm that achieves both? we are going to learn about an optimizer that achieves both in the next section
# 
# ## Adam
# Adam is relatively new algorithm. It got published in 2015. Adam is immune to sparse gradients and non-stationary optimization techniques. It has automatic annealing, It adapts learning rate to each parameter. Let's see the mathematics behind adam.
# 
# moment calculation:
# $$m_t = \beta_1 m_{t-1} + (1-\beta_1 )g_t$$
# $$v_t = \beta_2 v_{t-1} + (1-\beta_2 )g_t^2$$
# 
# Bias Update:
# $$\hat{m_t} = \frac{m_t}{1-\beta_1^t}$$
# $$\hat{v_t} = \frac{v_t}{1-\beta_2^t}$$
# 
# Parameter update:
# $$\theta _{t+1} = \theta _t +  \frac{\eta}{\sqrt{\hat{v_t}} + \epsilon } \hat{m_t}$$
# 
# As shown above adam has three steps ie., moment calculation, bias update and parameter update. In the moment calculation step, adam calculates first order and second order moments. first order moment is the temporal weighted average of the historical gradients and the second order moment is the temporal weighted average of square of historical gradients. $\beta _1$ and $\beta _2$ are coefficients similar to forgetting factor in rmsprop and they are close to 1. Authors who proposed this technique suggested to use 0.9 for $\beta _1$ and 0.999 for $\beta _2$. Therefore, $\beta _1$ and $\beta _2$ makes $m_t$ and $v_t$ biased towards the historical average because the weight $1-\beta_1$ and $1-\beta_2$ for the current gradient is near to zero. 
# At the start of the learning $m_0$ = 0; $v_0$ = 0 which makes the updates biased to zero. Therefore to correct that we use that bias correction term. AS $\beta _1$ < 1 after few iterations $\beta_1^t \approx$ 0 and the bias correction will not effect the moments. Parameter update is equivalent to rmpsprop with polyak momentum becase first order moment is equivalent to polyak momentum and the second order moment is the denominator term in rms prop are same.  $\epsilon$ is a small constand used to prevent divide-by-zero disaster.
# 
# Let's see Adam in Action:
# 

optimizer_params =  {
                "momentum_type"       : 'false',
                "regularization"      : (0.0001, 0.0002),
                "optimizer_type"      : 'adam',
                "id"                  : 'adam'
                        }
net = mlp(dataset_params, optimizer_params, 'adam')
costs.append(get_cost())


labels = ['sgd','sgd-polyak','sgd-nestrov', 'adagrad','rmsprop','adam']
plot_costs(costs, labels)


# The above plot shows that adam performs slightly better compared to rmsprop. 
# 
# To conclude this discussion of optimizers, this tutorial follows a path for introducing new optimizers. However that doesn't mean that adam works better than other techniques in every scenario. There may be many cases where adagrad may overperform rmsprop/adam. It is advised to analyze these technqiues and use them based on the requirement.
# 


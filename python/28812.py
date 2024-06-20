# # Inference in Mixture of Gaussians Model  
# 
# Our purpose here is to look at various ways of doing inference in mixture of Gaussians models. We would like to understand how each method compares with others in terms of effectiveness and efficiency.
# 
# Let us generate some data to test methods.
# 

import numpy as np
import scipy.stats as stat

seed = 123

# generate data
m1 = [0., 0.]
m2 = [4., -4.]
m3 = [-4., 4.]
x1 = stat.multivariate_normal.rvs(m1, size=100, random_state=seed)
x2 = stat.multivariate_normal.rvs(m2, size=100, random_state=seed)
x3 = stat.multivariate_normal.rvs(m3, size=100, random_state=seed)
x = np.vstack([x1, x2, x3])
y = np.zeros(300, dtype=np.int8)
y[100:200] = 1
y[200:300] = 2


# ## Gradient Ascent
# 
# First, we apply gradient ascent.
# 

get_ipython().magic('matplotlib inline')
plt.scatter(x[:, 0], x[:, 1], c=y)


def calc_grads(x, pk, m):
    nx = np.zeros((300, 3))
    for i in range(3):
        nx[:, i] = stat.multivariate_normal.pdf(x, mean=m[i], cov=1)

    pxk = nx*pk
    px = np.sum(pxk, axis=1)
    pk_x = pxk / px[:, np.newaxis]

    dpk = np.sum(nx / px[:, np.newaxis], axis=0) / x.shape[0]
    dm = np.zeros((3, 2))
    for i in range(3):
        dm[i] = np.sum(pk_x[:, i:(i+1)] * (x - m[i]), axis=0) / 2.
        
    return dpk, dm, pk_x


# randomly initialize parameters
pk = np.random.rand(3)
pk /= np.sum(pk)

m = np.zeros((3, 2))
m[0, :] = x[np.random.randint(x.shape[0])]
m[1, :] = x[np.random.randint(x.shape[0])]
m[2, :] = x[np.random.randint(x.shape[0])]

# gradient ascent
lr = 1e-2

i = 0
while True:
    i = i + 1
    dpk, dm, pk_x = calc_grads(x, pk, m)
    m += lr * dm
    pk += lr * dpk
    pk /= np.sum(pk)
    if np.sum(np.square(dm)) < 1e-9:
        break

print pk
print m
print "Complete in {0:d} iterations".format(i)


yp = np.argmax(pk_x, axis=1)
plt.scatter(x[:, 0], x[:, 1], c=yp)
plt.scatter(m[:, 0], m[:, 1], s=50, c='red', marker='o')


# ## Expectation Maximization
# 
# As our second method, we look at expectation maximization.
# 

def e_step(x, pk, m):
    nx = np.zeros((300, 3))
    for i in range(3):
        nx[:, i] = stat.multivariate_normal.pdf(x, mean=m[i], cov=1)

    pxk = nx*pk
    px = np.sum(pxk, axis=1)
    pk_x = pxk / px[:, np.newaxis]

    return pk_x

def m_step(x, pk_x):
    pk = np.sum(pk_x, axis=0) / np.sum(pk_x)
    m = np.zeros((3, 2))
    for i in range(3):
        m[i] = np.sum(pk_x[:, i:(i+1)] * x, axis=0) / np.sum(pk_x[:, i:(i+1)])
    return pk, m


# randomly initialize parameters
pk = np.random.rand(3)
pk /= np.sum(pk)

m = np.zeros((3, 2))
m[0, :] = x[np.random.randint(x.shape[0])]
m[1, :] = x[np.random.randint(x.shape[0])]
m[2, :] = x[np.random.randint(x.shape[0])]


i = 0
while True:
    i = i + 1
    old_m = m
    pk_x = e_step(x, pk, m)
    pk, m = m_step(x, pk_x)
    if np.sum(np.square(m - old_m)) < 1e-9:
        break
        
print pk
print m
print "Complete in {0:d} iterations".format(i)


pk_x = e_step(x, pk, m)
yp = np.argmax(pk_x, axis=1)
plt.scatter(x[:, 0], x[:, 1], c=yp)
plt.scatter(m[:, 0], m[:, 1], s=50, c='red', marker='o')


# ##Unsupervised Learning with Neural Networks
# Here, I would like to look at different methods for unsupervised learning in neural networks. I focus on latent variable models where we have observed data $X$ and latent variables $Z$. Our purpose is to learn both a generative, $p_{\theta}(x|z)$, and a recognition model, $q_{\phi}(z|x)$ from the data. A good generative model will assign high probability to our observed data $X$. Therefore, we want to maximize $\log p_{\theta}(X)$ which can be written as  
# $$
# \log p_{\theta}(X) = \sum_{z} q_{\phi}(z|x) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z|x)} + \sum_z q_{\phi}(z|x) \log \frac{q_{\phi}(z|x)}{p_{\theta}(z|x)} \= \mathcal{L}(\theta, \phi) + \text{KL}(q_{\phi}(z|x) || p_{\theta}(z|x))
# $$
# 
# Since $\text{KL}(.||.)$ is always nonnegative, $\mathcal{L}(\theta, \phi)$ is a lower bound on log likelihood, i.e., $\mathcal{L}(\theta, \phi) \leq p_{\theta}(X)$. Therefore, we can maximize log likelihood by maximizing $\mathcal{L}(\theta, \phi)$ with respect to $\theta$ and $\phi$. Note that we can write $\mathcal{L}(\theta, \phi)$ as follows using $p_{\theta}(x,z) = p_{\theta}(x|z)p_{\theta}(z)$
# $$
# \mathcal{L}(\theta, \phi) = \sum_{z} q_{\phi}(z|x) \log \frac{p_{\theta}(x|z)}{q_{\phi}(z|x)} + \sum_{z} q_{\phi}(z|x) \log \frac{p_{\theta}(z)}{q_{\phi}(z|x)}
# $$
# 
# This decomposition helps us see what maximizing $\mathcal{L}$ does. We see that the first term tries to bring the recognition and generative models closer to each other. While prior $p_{\theta}(z)$ acts as a regularizer for $q_{\phi}(z|x)$. (In the examples below, to keep things simple, we assume $p_{\theta}(z)$ is uniform.)
# 
# Let us calculate the derivatives of $\mathcal{L}$.
# $$
# \frac{\partial \mathcal{L}(\theta, \phi)}{\partial \theta} = \sum_{z} q_{\phi}(z|x) \nabla_{\theta} \log p_{\theta}(x,z) \= \mathbb{E}_{q}[\nabla_{\theta} \log p_{\theta}(x,z)]
# $$
# which can be approximated with samples from $q_{\phi}(z|x)$
# $$
# \frac{\partial \mathcal{L}(\theta, \phi)}{\partial \theta} \approx \frac{1}{L} \sum_{z_l \sim q} \nabla_{\theta} \log p_{\theta}(x,z_l)
# \tag 1
# $$
# Derivative with respect to $\phi$ is a bit more involved
# $$
# \frac{\partial \mathcal{L}(\theta, \phi)}{\partial \phi} = \sum_{z} \nabla_{\phi} q_{\phi}(z|x) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z|x)} + q_{\phi}(z|x) \nabla_{\phi} \log \frac{p_{\theta}(x,z)}{q_{\phi}(z|x)} \= \sum_{z} \nabla_{\phi} q_{\phi}(z|x) \log \frac{p_{\theta}(x,z)}{q_{\phi}(z|x)} - \nabla_{\phi} q_{\phi}(z|x) \= \sum_{z} \nabla_{\phi} q_{\phi}(z|x) [\log p_{\theta}(x,z) - \log q_{\phi}(z|x) - 1] \$$
# where we use $\sum_z \nabla q = \nabla \sum_z q = \nabla 1 = 0$ to get
# $$
# \frac{\partial \mathcal{L}(\theta, \phi)}{\partial \phi} = \sum_{z} \nabla_{\phi} q_{\phi}(z|x) [\log p_{\theta}(x,z) - \log q_{\phi}(z|x)] \$$
# We can turn this into an expectation over $q_{\phi}(z|x)$ by using $\nabla_{\phi} q_{\phi}(z|x) = q_{\phi}(z|x) \nabla_{\phi} \log q_{\phi}(z|x)$ (Williams, 1992; Mnih and Gregor, 2014)
# $$
# \frac{\partial \mathcal{L}(\theta, \phi)}{\partial \phi} = \sum_{z} q_{\phi}(z|x) \nabla_{\phi} \log q_{\phi}(z|x) [\log p_{\theta}(x,z) - \log q_{\phi}(z|x)] \= \mathbb{E}_q [\nabla_{\phi} \log q_{\phi}(z|x) [\log p_{\theta}(x,z) - \log q_{\phi}(z|x)]
# $$
# Again, this can be approximated with samples from $q_{\phi}(z|x)$
# $$
# \frac{\partial \mathcal{L}(\theta, \phi)}{\partial \phi} \approx \frac{1}{L} \sum_{z_l \sim q} [\nabla_{\phi} \log q_{\phi}(z_l|x) [\log p_{\theta}(x,z_l) - \log q_{\phi}(z_l|x)]
# \tag 2
# $$
# However, the above estimate is usually too high variance to be useful in practice. See Mnih and Gregor (2014) for various simple ways to reduce the variance of this estimate. For our simple example below, we will be able to use the above estimate directly. However, as we will see, this estimate will have high variance. 
# 

# ###Sampling with Neural Networks
# We would like to model both $p(x|z)$ and $q(z|x)$ with neural networks. We need to be able to sample from these distributions and calculate probabilities. How can we represent a probability distribution with function approximators such as neural networks? There seems to be two ways to do it. We can assume a functional form for the distribution and let the neural network output the parameters of this distribution. Or we can include stochastic units in the network and let the network directly produce samples from the distribution. However, if we adopt the latter approach, it becomes difficult to calculate probability of a produced sample. Because we have no analytical expression for the probability of a sample (in contrast to the first approach where we assume a functional form), it is unclear how we can calculate for example $\nabla_{\phi} \log q_{\phi}$. Therefore, we adopt the first approach here and let the network outputs represent the parameters of the distribution.
# 
# Specifically, we model $p_{\theta}(x|z)$ and $q_{\phi}(z|x)$ with neural networks and assume $x \in \mathbb{R}^D$ and $z \in \mathbb{R}^K$. Networks output the means of the Gaussian distributions from which the outputs are sampled (one can easily extend this to output variances of the distributions as well). Using a simple one layer network,
# $$
# \mu_z = W_{\phi}x + b_{\phi} \z \sim N(\mu_z, \mathbf{I})
# $$
# and
# $$
# \mu_x = W_{\theta}z + b_{\theta} \x \sim N(\mu_x, \mathbf{I})
# $$
# We also assume $p_{\theta}(z) \propto 1$ for the sake of simplicity. With the above expressions, it is straightforward to calculate the derivatives of $\mathcal{L}(\theta, \phi)$. 
# 
# Let us first generate some data to test our model. We are going to sample 400 points with each 100 drawn from a 4D Gaussian with a different mean. The data points are well separated in 4D space making this a rather simple problem.
# 

# generate data
import numpy as np
import scipy.stats as stat

np.random.seed(123)
m1 = [2., 2., 2., 2.]
m2 = [-2., -2., -2., -2.]
m3 = [-2., -2., 2., 2.]
m4 = [2., 2., -2., -2.]
x1 = stat.multivariate_normal.rvs(m1, size=150)
x2 = stat.multivariate_normal.rvs(m2, size=150)
x3 = stat.multivariate_normal.rvs(m3, size=150)
x4 = stat.multivariate_normal.rvs(m4, size=150)
tx = np.vstack([x1[0:100], x2[0:100], x3[0:100], x4[0:100]])
vx = np.vstack([x1[100:], x2[100:], x3[100:], x4[100:]])
ty = np.zeros(400, dtype=np.int8)
vy = np.zeros(200, dtype=np.int8)
ty[100:200] = 1
vy[50:100] = 1
ty[200:300] = 2
vy[100:150] = 2
ty[300:400] = 3
vy[150:200] = 3

# shuffle
rperm = np.random.permutation(400)
tx = tx[rperm]
ty = ty[rperm]

# randomly initialize parameters (we want all runs to start from the same initial parameters)
w_rec_init = np.random.randn(4, 2).T
b_rec_init = np.random.randn(2)
w_gen_init = np.random.randn(2, 4).T
b_gen_init = np.random.randn(4)


# ###Method 1: Gradient ascent on variational lower-bound
# Now we use gradient ascent to learn the parameters $\theta$ and $\phi$. We use equations 1 and 2 to estimate gradients.
# 

def calc_dl_dgen(xi, w_gen, b_gen, w_rec, b_rec, L=10):
    """Calculate derivative of L with respect to generative parameters
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
        L: Number of samples used to estimate derivatives
    Returns:
        ndarray: Derivative of L wrt to generative weight matrix
        ndarray: Derivative of L wrt to generative biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    dl_dw_gen = 0.0
    dl_db_gen = 0.0
    for l in range(L):
        # latent sample
        zi = mu_zi + np.random.randn(*mu_zi.shape)
        # generate x from latent sample
        mu_xi = np.dot(w_gen, zi) + b_gen
        # calculate derivatives
        dl_dw_gen += np.outer((xi - mu_xi), zi)
        dl_db_gen += (xi - mu_xi)
    dl_dw_gen /= L
    dl_db_gen /= L
    return dl_dw_gen, dl_db_gen

def calc_dl_drec(xi, w_gen, b_gen, w_rec, b_rec, L=10):
    """Calculate derivative of L with respect to recognition parameters
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
        L: Number of samples used to estimate derivatives
    Returns:
        ndarray: Derivative of L wrt to recognition weight matrix
        ndarray: Derivative of L wrt to recognition biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    dl_dw_rec = 0.0
    dl_db_rec = 0.0
    for l in range(L):
        # latent sample
        zi = mu_zi + np.random.randn(*mu_zi.shape)
        # run generation network
        mu_xi = np.dot(w_gen, zi) + b_gen
        # log p_{\theta}(x,z_l) - log q_{\phi}(z_l|x)
        d = np.sum(np.square(zi - mu_zi)) - np.sum(np.square(xi - mu_xi))
        dl_dw_rec += d*np.outer((zi - mu_zi), xi)
        dl_db_rec += d*(zi - mu_zi)
    dl_dw_rec /= L
    dl_db_rec /= L
    return dl_dw_rec, dl_db_rec

def calc_log_ll(x, w_gen, b_gen, w_rec, b_rec):
    log_ll = 0.0
    for i in range(x.shape[0]):
        xi = x[i]
        zp = np.dot(w_rec, xi) + b_rec
        xp = np.dot(w_gen, zp) + b_gen
        log_ll += -np.sum(np.square(xp - xi))
    return log_ll / x.shape[0]
    


# learn parameters using gradient ascent
w_gen_ga = w_gen_init.copy()  
b_gen_ga = b_gen_init.copy()  
w_rec_ga = w_rec_init.copy()  
b_rec_ga = b_rec_init.copy()  

np.random.seed(456)

epoch_count = 50
lr_gen = 1e-3
lr_rec = lr_gen * 1e-1

log_ll_ga = np.zeros(epoch_count+1)
log_ll_ga[0] = calc_log_ll(vx, w_gen_ga, b_gen_ga, w_rec_ga, b_rec_ga)
print "Initial val log ll: {0:f}".format(log_ll_ga[0])

dwrec_magnitude = np.zeros(tx.shape[0])
for e in range(epoch_count):
    for i in range(tx.shape[0]):
        xi = tx[i]
        dwgen, dbgen = calc_dl_dgen(xi, w_gen_ga, b_gen_ga, w_rec_ga, b_rec_ga, 1)
        w_gen_ga += lr_gen * dwgen
        b_gen_ga += lr_gen * dbgen
        dwrec, dbrec = calc_dl_drec(xi, w_gen_ga, b_gen_ga, w_rec_ga, b_rec_ga, 1)
        w_rec_ga += lr_rec * dwrec
        b_rec_ga += lr_rec * dbrec  
        
        dwrec_magnitude[i] = np.sum(np.square(dwrec))

    log_ll_ga[e+1] = calc_log_ll(vx, w_gen_ga, b_gen_ga, w_rec_ga, b_rec_ga)
    
print "Final val log ll: {0:f}, grad w_rec magnitude in last epoch: {1:f}+-{2:f}".format(log_ll_ga[-1], 
                                                                              np.mean(dwrec_magnitude), 
                                                                              np.std(dwrec_magnitude))


log_ll_ga


# Note the high variance of the gradient with respect to recognition weights.
# 

get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (10, 8)

def plot_z(x, y, w_rec, b_rec):
    z = np.zeros((x.shape[0], 2))
    for i in range(x.shape[0]):
        xi = x[i]
        zi = np.dot(w_rec, xi) + b_rec
        z[i] = zi
    plt.scatter(z[:,0], z[:,1], c=y)
  
# plot the 2D latent space
plot_z(tx, ty, w_rec_ga, b_rec_ga)


# ###Method 2: Variational autoencoder, i.e., reparameterization trick
# One way to calculate low variance estimates of the gradient with respect to recognition parameters is the reparameterization trick (Kingma and Welling, 2014). This trick allows us to calculate the derivative without needing to rewrite $\nabla_{\phi} q_{\phi}(z|x)$ as $q_{\phi}(z|x) \nabla_{\phi} \log q_{\phi}(z|x)$. Without the derivative of $\log q$, we get an estimate with much lower variance. This trick relies on being able to separate the stochastic component in sampling from a distribution. For example, in our case, $z$ are sampled from a normal distribution with mean given by the recognition model.
# $$
# z \sim N(\mu_z, \mathbf{I}) \$$
# where $\mu_z$ is some function of $x$, i.e., $\mu_z = h_{\phi}(x)$. If we rewrite $z$ as
# $$
# z = h_{\phi}(x) + \epsilon \\epsilon ~ N(0, \mathbf{I})
# $$
# we can rewrite the derivative of $\mathcal{L}$ with respect to $\phi$ as
# $$
# \nabla_{\phi} \mathcal{L}(\theta, \phi) = \nabla_{\phi} \mathbb{E}_q[\log p_{\theta}(x,z) - \log q_{\phi}(z|x)] \= \nabla_{\phi} \mathbb{E}_{\epsilon \sim N(0, \mathbf{I})}[\log p_{\theta}(x,h_{\phi}(x) + \epsilon) - \log q_{\phi}(h_{\phi}(x) + \epsilon|x)] \= \mathbb{E}_{\epsilon \sim N(0, \mathbf{I})}[\nabla_{\phi} \log p_{\theta}(x,h_{\phi}(x) + \epsilon) - \nabla_{\phi}  \log q_{\phi}(h_{\phi}(x) + \epsilon|x)] \$$
# where the second term inside the expectation is zero because it is independent of $\phi$ (mean of $q$ is $h_{\phi}(x)$ which cancels the other $h_{\phi}(x)$). Hence,
# $$
# \nabla_{\phi} \mathcal{L}(\theta, \phi) = \mathbb{E}_{\epsilon \sim N(0, \mathbf{I})}[\nabla_{\phi} \log p_{\theta}(x,h_{\phi}(x) + \epsilon)] \tag 3
# $$
# 
# Below, we use the above expression to calculate an estimate of the gradient with respect to recognition parameters. As we will see, this estimate has much lower variance. (We still use eqn. 1 to calculate the gradient with respect to $\theta$).
# 

def calc_dl_drec_reparameterized(xi, w_gen, b_gen, w_rec, b_rec, L=10):
    """Calculate derivative of L with respect to recognition parameters using the reparameterization trick
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
        L: Number of samples used to estimate derivatives
    Returns:
        ndarray: Derivative of L wrt to recognition weight matrix
        ndarray: Derivative of L wrt to recognition biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    dl_dw_rec = 0.0
    dl_db_rec = 0.0
    for l in range(L):
        # latent sample
        zi = stat.multivariate_normal.rvs(mu_zi)
        # run generation network
        mu_xi = np.dot(w_gen, zi) + b_gen
        # calculate derivatives
        dl_dw_rec += np.dot(w_gen.T, np.outer((xi - mu_xi), xi))
        dl_db_rec += np.dot(w_gen.T, (xi - mu_xi))
    dl_dw_rec /= L
    dl_db_rec /= L
    return dl_dw_rec, dl_db_rec


w_gen_rt = w_gen_init.copy()  
b_gen_rt = b_gen_init.copy()  
w_rec_rt = w_rec_init.copy()  
b_rec_rt = b_rec_init.copy()  

np.random.seed(456)

epoch_count = 100
lr_gen = 1e-3
lr_rec = lr_gen * 1e-1

log_ll_rt = np.zeros(epoch_count+1)
log_ll_rt[0] = calc_log_ll(x, w_gen_rt, b_gen_rt, w_rec_rt, b_rec_rt)
print "Initial log ll: {0:f}".format(log_ll_rt[0])

dwrec_magnitude = np.zeros(x.shape[0])
for e in range(epoch_count):
    for i in range(x.shape[0]):
        xi = x[i]
        dwgen, dbgen = calc_dl_dgen(xi, w_gen_rt, b_gen_rt, w_rec_rt, b_rec_rt, 1)
        w_gen_rt += lr_gen * dwgen
        b_gen_rt += lr_gen * dbgen
        dwrec, dbrec = calc_dl_drec_reparameterized(xi, w_gen_rt, b_gen_rt, w_rec_rt, b_rec_rt, 1)
        w_rec_rt += lr_rec * dwrec
        b_rec_rt += lr_rec * dbrec    
        
        dwrec_magnitude[i] = np.sum(np.square(dwrec))

    log_ll_rt[e+1] = calc_log_ll(x, w_gen_rt, b_gen_rt, w_rec_rt, b_rec_rt)

print "Final log ll: {0:f}, grad w_rec magnitude in last epoch: {1:f}+-{2:f}".format(log_ll_rt[-1], 
                                                                              np.mean(dwrec_magnitude), 
                                                                              np.std(dwrec_magnitude))


# plot the 2D latent space
plot_z(x, y, w_rec_rt, b_rec_rt)


# ###Method 3: Wake-sleep Algorithm
# Probably the earliest method for training unsupervised learning models like ours is the wake-sleep algorithm by Hinton et al. (1995). This is a heuristic algorithm, and it does not optimize a well-defined bound like the variational lower bound. The algorithm consists of two steps
# 1. **Wake step:** Freeze the recognition weights $\phi$. Get sample $z$ for data point $x$ and generate $\hat{x}$ from the generative model $p_{\theta}(x|z)$. Update $\theta$ such that $\hat{x}$ is close to $x$. In other words, we want the generative model to generate the $x$ we put in.
# 2. **Sleep step:** Freeze the generative weights $\theta$. Sample $z$ randomly and generate $x$ from the generative model $p_{\theta}(x|z)$. Run the recognition model $q_{\phi}(z|x)$ to get $\hat{z}$. Update $\phi$ such that $\hat{z}$ is close to $z$. In other words, we want the recognition model to be able to recognize a fantasy $x$.
# More formally, in the wake step we are maximizing the following objective
# $$
# \mathbb{E}_{q}[\log p_{\theta}(x|z)] \approx \frac{1}{L} \sum_{z_l \sim q_{\phi}} \log p_{\theta}(x|z_l)
# $$
# Note that this is the same objective with the one we get from the variational lower bound (the derivative with respect to $\theta$ is the same with equation 1). Wake step is the gradient ascent step with respect to $\theta$. In the sleep step, we are maximizing the following objective
# $$
# \mathbb{E}_{p}[\log q_{\phi}(z|x)] \approx \frac{1}{L} \sum_{x_l \sim p_{\theta}} \log q_{\phi}(x_l|z)
# $$
# In contrast to the wake step, the sleep step does not correspond to gradient ascent with respect to $\phi$. This is why we say that wake-sleep algorithm does not optimize a well-defined objective. Nevertheless, this algorithm works in practice. However, we would expect it to work worse than the methods we looked at above. 
# 

def wake_sleep(xi, w_gen, b_gen, w_rec, b_rec, L=10):
    """Apply one step of wake-sleep algorithm and get updates for generative and recognition parameters
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
        L: Number of samples used to estimate derivatives
    Returns:
        ndarray: Update for generative weight matrix
        ndarray: Update for generative biases
        ndarray: Update for recognition weight matrix
        ndarray: Update for recognition biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    dl_dw_gen = 0.0
    dl_db_gen = 0.0
    dl_dw_rec = 0.0
    dl_db_rec = 0.0
    for l in range(L):
        # latent sample
        zi = stat.multivariate_normal.rvs(mu_zi)
        # generate x from latent sample
        mu_xi = np.dot(w_gen, zi) + b_gen
        xi_pred = stat.multivariate_normal.rvs(mu_xi)
        # run recognition on predicted x
        mu_zi_pred = np.dot(w_rec, xi_pred) + b_rec
        zi_pred = stat.multivariate_normal.rvs(mu_zi)
        # calculate derivatives
        dl_dw_gen += np.outer((xi - xi_pred), zi)
        dl_db_gen += (xi - xi_pred)
        dl_dw_rec += np.outer((zi - zi_pred), xi_pred)
        dl_db_rec += (zi - zi_pred)
    dl_dw_gen /= L
    dl_db_gen /= L
    dl_dw_rec /= L
    dl_db_rec /= L
    return dl_dw_gen, dl_db_gen, dl_dw_rec, dl_db_rec


w_gen_ws = w_gen_init.copy()  
b_gen_ws = b_gen_init.copy()  
w_rec_ws = w_rec_init.copy()  
b_rec_ws = b_rec_init.copy()  

np.random.seed(456)

epoch_count = 100
lr_gen = 1e-4
lr_rec = lr_gen * 1e-1

log_ll_ws = np.zeros(epoch_count+1)
log_ll_ws[0] = calc_log_ll(x, w_gen_ws, b_gen_ws, w_rec_ws, b_rec_ws)
print "Initial log ll: {0:f}".format(log_ll_ws[0])

dwrec_magnitude = np.zeros(x.shape[0])
for e in range(epoch_count):
    for i in range(x.shape[0]):
        xi = x[i]
        dwgen, dbgen, dwrec, dbrec = wake_sleep(xi, w_gen_ws, b_gen_ws, w_rec_ws, b_rec_ws, 1)
        w_gen_ws += lr_gen * dwgen
        b_gen_ws += lr_gen * dbgen
        w_rec_ws += lr_rec * dwrec
        b_rec_ws += lr_rec * dbrec    
        
        dwrec_magnitude[i] = np.sum(np.square(dwrec))

    log_ll_ws[e+1] = calc_log_ll(x, w_gen_ws, b_gen_ws, w_rec_ws, b_rec_ws)

print "Final log ll: {0:f}, grad w_rec magnitude in last epoch: {1:f}+-{2:f}".format(log_ll_ws[-1], 
                                                                              np.mean(dwrec_magnitude), 
                                                                              np.std(dwrec_magnitude))


plot_z(x, y, w_rec_ws, b_rec_ws)


# ###Method 4: Classical autoencoder
# How is the model we looked at so far different from a classical [autoencoder](https://en.wikipedia.org/wiki/Autoencoder)? Our latent variable model is a probabilistic model. We are learning two probability distributions: generative distribution $p$ and recognition distribution $q$. However, a classical autoencoder is not a probabilistic model; the purpose is simply to learn a compressed representation of the input data. However, in terms of architecture, the classical autoencoder is almost identical to the model we looked at so far. In fact, the only difference is the sampling step where the latent variable $z$ is sampled from some mean $\mu_z$ that is produced by the recognition model. In a classical autoencoder, $\mu_z$ would be used directly as our latent representation $z$. In addition, for our model, we optimized the variational lower bound. In a classical autoencoder, on the other hand, we optimize reconstruction error. Hence, it seems unclear if and how the two models are related.
# 
# Let us look at the gradient expressions for classical autoencoder. Assume we have two functions: generation function $g_{\theta}(x)$ and recognition function $h_{\phi}(x)$. Then the purpose is to learn $g$ and $h$ that minimizes the following objective
# $$
# R(\theta, \phi) = \sum_n ||x_n - g_{\theta}(h_{\phi}(x_n))||_2^2
# $$
# Focusing on a single data point $x$, the derivatives with respect to $\theta$ and $\phi$ are as follows:
# $$
# \frac{\partial R(\theta, \phi)}{\partial \theta} = -(x - g_{\theta}(h_{\phi}(x)))\nabla_{\theta} g_{\theta}(h_{\phi}(x)) \\frac{\partial R(\theta, \phi)}{\partial \phi} = -(x - g_{\theta}(h_{\phi}(x))) g_{\theta}'(h_{\phi}(x)) \nabla_{\phi} h_{\phi}(x)
# $$
# 
# Now let us look at the derivatives of the variational lower bound again. Starting from equation 1 and making use of the reparameterization trick, we can write
# $$
# \frac{\partial \mathcal{L}(\theta, \phi)}{\partial \theta} \approx \frac{1}{L} \sum_{z_l \sim q} \nabla_{\theta} \log p_{\theta}(x,z_l) \= \frac{1}{L} \sum_{\epsilon \sim N(0,\mathbf{I})} \nabla_{\theta} \log p_{\theta}(x,h_{\phi}(x) + \epsilon)) \= -\frac{1}{L} \sum_{\epsilon \sim N(0,1)} \nabla_{\theta} ||x - g_{\theta}(h_{\phi}(x) + \epsilon))||_2^2 \= \frac{1}{L} \sum_{\epsilon \sim N(0,1)} (x - g_{\theta}(h_{\phi}(x) + \epsilon))\nabla_{\theta} g_{\theta}(h_{\phi}(x) + \epsilon)
# $$
# Here we used $g_{\theta}(z)$ to denote the function calculates $\mu_x$ from $z$. Note that if we ignore sampling $z$ and use $\mu_z$ directly, i.e., set $\epsilon=0$, the above expression is the same with the derivative of $R$ with respect to $\theta$ (ignoring sign difference because we are maximizing instead of minimizing here). 
# 
# Similarly, starting from equation 3, we can write the derivative of the variational lower bound with respect to $\phi$ as follows
# $$
# \nabla_{\phi} \mathcal{L}(\theta, \phi) = \mathbb{E}_{\epsilon \sim N(0, \mathbf{I})}[\nabla_{\phi} \log p_{\theta}(x,h_{\phi}(x) + \epsilon)] \= \frac{1}{L} \sum_{\epsilon \sim N(0,1)} \nabla_{\phi} \log p_{\theta}(x,h_{\phi}(x) + \epsilon) \= -\frac{1}{L} \sum_{\epsilon \sim N(0,1)} \nabla_{\phi} ||x - g_{\theta}(h_{\phi}(x) + \epsilon))||_2^2 \= \frac{1}{L} \sum_{\epsilon \sim N(0,1)} (x - g_{\theta}(h_{\phi}(x) + \epsilon)))g_{\theta}'(h_{\phi}(x) + \epsilon) \nabla_{\phi} h_{\phi}(x) \$$
# Again if we set $\epsilon=0$, this expression is the same with the derivative of $R$ with respect to $\phi$. Therefore, classical autoencoder is equivalent to a variational autoencoder where we ignore stochasticity in $z$, i.e., assume the recognition model is deterministic. However, note that we were able to derive this equivalency thanks to the following assumptions:
# * Generative model $p_{\theta}(x|z)$ is Gaussian with variance independent of $z$. This assumption made minimizing reconstruction error equivalent  to maximizing probability. 
# * Recognition model $q_{\phi}(z|x)$ is Gaussian with variance independent of $x$. This made it possible to write $z = h_{\phi}(x) + \epsilon$.
# 

# Now let us look at how well a classical autoencoder does on our toy problem. This should give us an idea about how much we lose if we ignore stochasticity in $z$. 
# 

def calc_dl_dgen_ae(xi, w_gen, b_gen, w_rec, b_rec):
    """Calculate derivative of R (reconstruction error for classical autoencoder) 
    with respect to generative parameters
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
    Returns:
        ndarray: Derivative of R wrt to generative weight matrix
        ndarray: Derivative of R wrt to generative biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    # generate x from latent z
    mu_xi = np.dot(w_gen, mu_zi) + b_gen
    # calculate derivatives
    dl_dw_gen = np.outer((mu_xi - xi), mu_zi)
    dl_db_gen = (mu_xi - xi)
    return dl_dw_gen, dl_db_gen

def calc_dl_drec_ae(xi, w_gen, b_gen, w_rec, b_rec):
    """Calculate derivative of R (reconstruction error for classical autoencoder) 
    with respect to recognition parameters
    Parameters:
        xi (ndarray): Training sample
        w_gen: Generative weights
        b_gen: Generative biases
        w_rec: Recognition weights
        b_rec: Recognition biases
    Returns:
        ndarray: Derivative of R wrt to recognition weight matrix
        ndarray: Derivative of R wrt to recognition biases
    """
    # run recognition network
    mu_zi = np.dot(w_rec, xi) + b_rec
    # run generation network
    mu_xi = np.dot(w_gen, mu_zi) + b_gen
    # calculate derivatives
    dl_dw_rec = np.dot(w_gen.T, np.outer((mu_xi - xi), xi))
    dl_db_rec = np.dot(w_gen.T, (mu_xi - xi))
    return dl_dw_rec, dl_db_rec


w_gen_ae = w_gen_init.copy()  
b_gen_ae = b_gen_init.copy()  
w_rec_ae = w_rec_init.copy()  
b_rec_ae = b_rec_init.copy()  

np.random.seed(456)

epoch_count = 100
lr_gen = 1e-3
lr_rec = lr_gen * 1e-1

log_ll_ae = np.zeros(epoch_count+1)
log_ll_ae[0] = calc_log_ll(x, w_gen_ae, b_gen_ae, w_rec_ae, b_rec_ae)
print "Initial log ll: {0:f}".format(log_ll_ae[0])

dwrec_magnitude = np.zeros(x.shape[0])
for e in range(epoch_count):
    for i in range(x.shape[0]):
        xi = x[i]
        dwgen, dbgen = calc_dl_dgen_ae(xi, w_gen_ae, b_gen_ae, w_rec_ae, b_rec_ae)
        w_gen_ae -= lr_gen * dwgen
        b_gen_ae -= lr_gen * dbgen
        dwrec, dbrec = calc_dl_drec_ae(xi, w_gen_ae, b_gen_ae, w_rec_ae, b_rec_ae)
        w_rec_ae -= lr_rec * dwrec
        b_rec_ae -= lr_rec * dbrec    
        
        dwrec_magnitude[i] = np.sum(np.square(dwrec))

    log_ll_ae[e+1] = calc_log_ll(x, w_gen_ae, b_gen_ae, w_rec_ae, b_rec_ae)
    
print "Final log ll: {0:f}, grad w_rec magnitude in last epoch: {1:f}+-{2:f}".format(log_ll_ae[-1], 
                                                                              np.mean(dwrec_magnitude), 
                                                                              np.std(dwrec_magnitude))


# plot the 2D latent space
plot_z(x, y, w_rec_ae, b_rec_ae)


# Let us compare the log likelihoods for each algorithm
plt.plot(range(81), log_ll_ga[20:])
plt.plot(range(81), log_ll_rt[20:])
plt.plot(range(81), log_ll_ws[20:])
plt.plot(range(81), log_ll_ae[20:])
plt.legend(['Gradient ascent', 'Variational autoencoder', 'Wake-sleep algorithm', 'Classical autoencoder'], loc='best')


# ####To-do
# Compare VAE and AE on MNIST.
# 

# #Comparing the error rates for Monte Carlo and Riemann Integration
# A commonly mentioned advantage of Monte Carlo integration is that the error rate does not depend on the dimensionality of the input space; it only depends on the number of samples used to estimate the integral. Let $\int_{\mathcal{H}} f(x)$ be the integral we want to estimate. In Monte Carlo integration, we use samples drawn uniformly from $\mathcal{H}$ to form our estimate as follows
# $$F_{mc} = \frac{1}{N}(\sum_i f(x_i))$$
# If we let $F$ to denote the true value of the integral, by the law of large numbers, $F_{mc}$ approaches $F$. Moreover, the variance of our estimate is $\frac{\sigma^2}{N}$ where $\sigma^2$ is the variance of $f(x)$. Hence, the error of Monte Carlo estimate varies with $O(n^{-1/2})$.
# 
# In Riemann integration, we simply partition $\mathcal{H}$ into $N$ equally spaced intervals and estimate the integral simply by
# $$F_{r} = \frac{1}{N} \sum_i f(x_i) |H|$$
# where $|H|$ is the size of the domain of integration. The error of Riemann estimate varies with $O(n^{-1})$, i.e., a better rate compared to Monte Carlo integration.
# 
# Let us first see that this is the case with a simple example. Let $f(x) = (1 - ||x||_2^2)$ where $x \in \mathbb{R}^D$. We use Monte Carlo and Riemann integration to estimate $\int_0^1 f(x)$ and look at how the error changes.
# 

import numpy as np
import matplotlib.pyplot as plt

def grid(xl, xu, N, D):
    """
    Create a grid of N evenly spaced points in D-dimensional space
    xl: lower bound of x
    xu: upper bound of x
    N: number of points per dimension
    D: number of dimensions
    """
    xr = np.linspace(xl, xu, N)
    g = np.zeros((N**D, D))
    for n in range(N**D):
        index = np.unravel_index(n, tuple([N]*D))
        g[n] = [xr[i] for i in index]
        
    return g

def f(x):
    return (1 - (np.sum(np.square(x)) / x.size))

def riemann(N, D):
    # riemann integration
    x = grid(0.0, 1.0, N, D)
    dx = 1.0 / (N**D)
    F_r = np.sum(np.apply_along_axis(f, 1, x) * dx)
    return F_r

def monte_carlo(N, D):
    # monte carlo integration
    x = np.random.rand(N**D, D)
    F_mc = np.sum(np.apply_along_axis(f, 1, x)) / (N**D)
    return F_mc

D = 1
N = np.logspace(1, 3, num=10, dtype=int)

F_r = np.zeros(10)
for i,n in enumerate(N):
    F_r[i] = riemann(n, D)
    
# error in riemann estimate
e_r = np.abs(F_r - 2.0/3.0)
print(e_r)


# Below we plot the error in Riemann estimate with respect to number of sample points. As it is clear from the figure, this error varies with $O(N^{-1})$.
# 

plt.plot(N, e_r)
plt.plot(N, (e_r[0]*N[0])/N)
plt.legend(['Error in Riemann estimate', '1/N'])
plt.show()


repeats = 1000
e_mc = np.zeros(10)
for i,n in enumerate(N):
    for r in range(repeats):
        e_mc[i] += np.abs(monte_carlo(n, D) - 2.0/3.0)

e_mc /= repeats
print(e_mc)


# Now let us look at the error in the Monte Carlo estimate. Again, as we have seen above, the error in Monte Carlo estimate varies with $O(n^{-1/2})$.
# 

plt.plot(N, e_mc)
plt.plot(N, (e_mc[0]*np.sqrt(N[0]))/np.sqrt(N))
plt.legend(['Error in Monte Carlo estimate', r"$\frac{1}{N^{-1/2}}$"])
plt.show()


# Then, why would anyone use Monte Carlo integration? Obviously, Riemann estimate decreases much more quickly as we increase the number of sample points. What is the deal with the error rate in Monte Carlo integration being independent of the number of dimensions? Do the Monte Carlo and Riemann errors change differently as we increase the number of dimensions? Let us look into that.
# 
# First, we need to understand how the Riemann error changes if we go to a higher dimensional space. Let $N$ denote the number of points along each axis; hence, if we have $D$ dimensions, we have $N^D$ sample points. In a D-dimensional space, does the error change as $O(1/N)$ or $O(1/(N^D))$? Let us try $D=2$.
# 

D = 2
N = np.array([5, 10, 20 , 50, 100])

F_rd = np.zeros(5)
for i,n in enumerate(N):
    F_rd[i] = riemann(n, D)
    
e_rd = np.abs(F_rd - 2.0/3.0)


plt.plot(N, e_rd)
plt.plot(N, (e_rd[0]*N[0])/(N))
plt.plot(N, (e_rd[0]*(N[0]**D))/(N**D))
plt.legend(['Error in Riemann estimate', '1/N', '1/N^D'])
plt.show()


# It looks more like $O(1/N)$. Let us look at $D=3$ too.
# 

D = 3
N = np.array([3, 6, 12 , 25, 50])

F_rd = np.zeros(5)
for i,n in enumerate(N):
    F_rd[i] = riemann(n, D)
    
e_rd = np.abs(F_rd - 2.0/3.0)


plt.plot(N, e_rd)
plt.plot(N, (e_rd[0]*N[0])/(N))
plt.plot(N, (e_rd[0]*(N[0]**D))/(N**D))
plt.legend(['Error in Riemann estimate', '1/N', '1/N^D'])
plt.show()


# Again it looks more like $O(1/N)$. Another way to look at that is to see how the error changes as we change the number of dimensions.
# 

D = [1,2,3,4,5]
N = 10

F_rn = np.zeros(5)
for i,d in enumerate(D):
    F_rn[i] = riemann(N, d)
    
e_rn = np.abs(F_rn - 2.0/3.0)


plt.plot(D, e_rn)
plt.show()


# As it is clear, if we keep the number of points per dimension constant, the error stays constant. In other words, Riemann error behaves as $O(1/N)$. That also means that if we keep the number of sample (grid) points constant, Riemann error should increase.
# 

D = np.array([1, 2, 4, 8, 16])
# we need to keep the number of samples constant across dimensions
N = np.array([65536, 256, 16, 4, 2])
F_rc = np.zeros(5)
for i,(n,d) in enumerate(zip(N,D)):
    F_rc[i] = riemann(n, d)
    
# error in riemann estimate
e_rc = np.abs(F_rc - (2.0 / 3.0))
print(e_rc)


plt.plot(D, e_rc)
plt.plot(D, (e_rc[0]*N[0])/N)
plt.legend(['Error in Riemann estimate', '1/N'])
plt.show()


# Though the increase seems to be larger than $O(1/N)$.
# 
# Nevertheless, for us the important point is to compare how the Monte Carlo error changes to Riemann error. Let us look at how the Monte Carlo error changes with respect to $N$ for $D=2$. Remember that $N$ is the number of points per dimension.
# 

repeats = 100
D = 2
N = np.array([5, 10, 20 , 50, 100])
e_mcd2 = np.zeros(5)
for i,n in enumerate(N):
    for r in range(repeats):
        e_mcd2[i] += np.abs(monte_carlo(n, D) - (2.0/3.0))

e_mcd2 /= repeats
print(e_mcd2)


plt.plot(N, e_mcd2)
plt.plot(N, (e_mcd2[0]*(N[0]))/N)
plt.legend(['Error in Monte Carlo estimate', r"$\frac{1}{\sqrt{N^D}}$"])
plt.show()


# Ahah! Note that the Monte Carlo error changes as $O(1/\sqrt{N^D})$ which is $O(1/N)$ in this case. In other words, Monte Carlo error depends on the total number of sample points, not on the number of points per dimension (as Riemann error does). For example, if we keep $N$ constant as we increase the number of dimensions, the Monte Carlo error should decrease. Remember, Riemann error stays the same as long as $N$ is constant.
# 

repeats = 100
D = np.array([1, 2, 3, 4, 5])
# we need to keep the number of samples constant across dimensions
N = 10
e_mcd = np.zeros(5)
for i,d in enumerate(D):
    for r in range(repeats):
        e_mcd[i] += np.abs(monte_carlo(N, d) - (2.0/3.0))

e_mcd /= repeats
print(e_mcd)


plt.plot(D, e_mcd)
D = np.array([1, 2, 3, 4, 5])
plt.plot(D, (e_mcd[0]*np.sqrt(N))/np.sqrt(N**D))
plt.legend(['Error in Monte Carlo estimate', r"$\frac{1}{\sqrt{N^D}}$"])
plt.show()


# Note that how the Monte Carlo error changes as $\frac{1}{\sqrt{N^D}}$. Therefore, one only needs a constant number of  sample points to achieve a constant error rate. In contrast, for Riemann integration, we need to keep the number of points per dimension constant, which means an exponentially increasing number of sample points as $D$ increases.
# 

# To recap, for Monte Carlo integration, the error varies with $O(1/\sqrt{N^D})$ where $N$ is the number of points per dimension. In contrast, for Riemann integration, the error varies with $O(1/N)$. Therefore, if we keep the total number of sample points constant, Riemann estimate will get worse as $D$ increases, while Monte Carlo error should stay the same. 
# 


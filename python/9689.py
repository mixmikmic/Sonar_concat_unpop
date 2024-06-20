get_ipython().magic('pylab inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Read the data
# 

# Read the data: Gender (1 for males, 2 for females), Age group (1 for 20-46, 2 for 46+), Brain weight (grams) and head size (cubic cm) for 237 adults.
# 
# Source: R.J. Gladstone (1905). "A Study of the Relations of the Brain to 
# to the Size of the Head", Biometrika, Vol. 4, pp105-123
# 

df = pd.read_csv("http://www.stat.ufl.edu/~winner/data/brainhead.dat", sep=" ", skipinitialspace=True, header=None)


df.head()


# Name the columns
# 

df.rename(columns={0:"gender",1:"age group",2:"head size", 3:"brain weight"}, inplace=True)


df.head()


# ## Plot the data
# 

# Plot the brain weight vs. head size for younger males only.
# 

youngmen = df[df["gender"]==1][df["age group"]==1]


plt.scatter(youngmen["head size"], youngmen["brain weight"])
plt.xlabel("head size (cm^3)")
plt.ylabel("brain weight (g)")
plt.title("Brain weight vs. head size")


# Extract the data for older men, plot allmen together:
# 

oldmen = df[df["gender"]==1][df["age group"]==2]


plt.scatter(youngmen["head size"], youngmen["brain weight"])
plt.scatter(oldmen["head size"], oldmen["brain weight"],color="r")
plt.xlabel("head size (cm^3)")
plt.ylabel("brain weight (g)")
plt.title("Brain weight vs. head size for men")
plt.legend(["age 20-46","age 46+"],loc="upper left")


youngwomen = df[df["gender"]==2][df["age group"]==1]
oldwomen = df[df["gender"]==2][df["age group"]==2]


plt.scatter(youngwomen["head size"], youngwomen["brain weight"])
plt.scatter(oldwomen["head size"], oldwomen["brain weight"],color="r")
plt.xlabel("head size (cm^3)")
plt.ylabel("brain weight (g)")
plt.title("Brain weight vs. head size for women")
plt.legend(["age 20-46","age 46+"],loc="upper left")


# Plot men and women together for comparison.
# 

plt.scatter(youngmen["head size"], youngmen["brain weight"])
plt.scatter(oldmen["head size"], oldmen["brain weight"],color="r")
plt.scatter(youngwomen["head size"], youngwomen["brain weight"], color="g")
plt.scatter(oldwomen["head size"], oldwomen["brain weight"],color="m")
plt.xlabel("head size (cm^3)")
plt.ylabel("brain weight (g)")
plt.title("Brain weight vs. head size for women")
plt.legend(["men 0-46","men 46+", "women 20-46","women 46+"],loc="upper left")


# We see that the correlation is about the same, but women, on average, have smaller heads and brains.
# 

# ## Distributions
# 

# Distribution of male head sizes
# 

allmen = df[df["gender"]==1]


allmen["head size"].describe()


allmen["brain weight"].describe()


plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.hist(allmen["head size"],bins=20,normed=True);
plt.xlabel("Head size (cm^3)")
plt.title("Distribution of male head sizes");
plt.subplot(1,2,2)
plt.hist(allmen["brain weight"],bins=20,normed=True);
plt.xlabel("Brain weight (g)")
plt.title("Distribution of male brain weights");


# Distributionof female head and brain sizes
# 

allwomen = df[df["gender"]==2]


allwomen["head size"].describe()


allwomen["brain weight"].describe()


plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.hist(allwomen["head size"],bins=20,normed=True);
plt.xlabel("Head size (cm^3)")
plt.title("Distribution of female head sizes");
plt.subplot(1,2,2)
plt.hist(allwomen["brain weight"],bins=20,normed=True);
plt.xlabel("Brain weight (g)")
plt.title("Distribution of female brain weights");


# ## Linear regression using scikit-learn
# 

from sklearn import linear_model


regr_men = linear_model.LinearRegression()


regr_men.fit(allmen["head size"].reshape(-1,1), allmen["brain weight"])


plt.scatter(allmen["head size"], allmen["brain weight"])
plt.plot(allmen["head size"], regr_men.predict(allmen["head size"].reshape(-1,1)), color='red',linewidth=3)
plt.title("Brain weight vs. head size for all men")
plt.legend(["data","linear fit"],loc="upper left")


regr_women = linear_model.LinearRegression()


regr_women.fit(allwomen["head size"].reshape(-1,1), allwomen["brain weight"])


plt.scatter(allwomen["head size"], allwomen["brain weight"])
plt.plot(allwomen["head size"], regr_women.predict(allwomen["head size"].reshape(-1,1)), color='red',linewidth=3)
plt.title("Brain weight vs. head size for all women")
plt.legend(["data","linear fit"],loc="upper left")





get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np
import scipy.special as sp


# Suppose we want to sample from the gamma distribution $\Gamma(\alpha,1)$ with shape parameter $\alpha$ and scale parameter $\beta=1$. The density function of $\Gamma(\alpha,1)$ is
# $$ p(x) = \frac{x^{\alpha -1}\mathrm{e}^{-x}}{\Gamma(\alpha)} $$
# 

x = np.linspace(0,8,100)
for a in [1,1.5,2]:
    plt.plot(x, x**(a-1)*np.exp(-x)/ sp.gamma(a), label="$\\alpha="+str(a)+"$")
plt.legend()
plt.xlabel("x")
plt.title("The gamma distribution $\Gamma(\\alpha,1)$");


# In order to sample this with the rejection sampling, we need an _instrumental distribution_. We assume we know how to draw samples from the instrumental distribution. We choose the general **exponential distribution**, with the density
# 
# $$ q_\lambda (x) = \lambda \mathrm{e}^{-\lambda x}$$
# 

# In order to maximize the acceptance probability (see lecture notes), we choose $\lambda = 1/\alpha$.
# 

from scipy.stats import expon
for a in [1,1.5,2]:
    plt.plot(x,(1.0/a)* expon.pdf(x, scale = 1/(1.0/a)), label="$\\lambda=%.2f$" % (1.0/a,))
plt.legend()
plt.xlabel("x")
plt.title("The exponential pdf $\lambda\mathrm{e}^{-\lambda x}$");


# Here's the main loop of rejection sampling, to retrieve one random value.
# 

def p(x):
    return x**(a-1)*np.exp(-x)/sp.gamma(a)
def q(x, lam):
    return lam * np.exp(-lam*x)

a = 2.0
lam = 1/a
M = (a**a) * np.exp(-(a-1))/sp.gamma(a)
while True:
    x = expon.rvs(scale=1.0/a)
    u = np.random.uniform()
    if u <= p(x)/(M*q(x,lam)):
        print x
        break


def rejsamp(n=1):
    retval = []
    ntrials = 0
    for i in range(n):
        while True:
            x = expon.rvs(scale=1/lam)
            u = np.random.uniform()
            ntrials += 1
            if u <= p(x)/(M*q(x,lam)):
                retval.append(x)
                break
    return ntrials, retval


a = 2.0
lam = 1/a
M = ((a-1)/(1-lam))**(a-1) * np.exp(-(a-1))/(lam*sp.gamma(a))
ntrials, outcomes = rejsamp(100000)


plt.hist(outcomes,bins=100, normed=True);
x = np.linspace(0,6,100)
plt.plot(x,p(x),"r", label="$p(x)$");
plt.plot(x,M*q(x,lam),"g", label="$Mq(x)$");
plt.xlim(0,6);
plt.legend();
plt.title("Acceptance rate: %.2f" % (1.0*len(outcomes)/ntrials,));


1/M


a = 2.0
lam = 0.1
M = ((a-1)/(1-lam))**(a-1) * np.exp(-(a-1))/(lam*sp.gamma(a))


ntrials, outcomes = rejsamp(10000)


plt.hist(outcomes,bins=100, normed=True);
x = np.linspace(0,6,100)
plt.plot(x,p(x),"r", label="$p(x)$");
plt.plot(x,M*q(x,lam),"g", label="$Mq(x)$");
plt.xlim(0,6);
plt.legend();
plt.title("Acceptance rate: %.2f" % (1.0*len(outcomes)/ntrials,))


a = 2.0
lam = 0.7
M = ((a-1)/(1-lam))**(a-1) * np.exp(-(a-1))/(lam*sp.gamma(a))


ntrials, outcomes = rejsamp(10000)


plt.hist(outcomes,bins=100, normed=True);
x = np.linspace(0,6,100)
plt.plot(x,p(x),"r", label="$p(x)$");
plt.plot(x,M*q(x,lam),"g", label="$Mq(x)$");
plt.xlim(0,6);
plt.legend();
plt.title("Acceptance rate: %.2f" % (1.0*len(outcomes)/ntrials,));





get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np


# Consider the Rosenbrock function
# $$f(x,y)=10(y-x^2)^2 + (1-x)^2$$
# with gradient
# $$\nabla f = \left[\begin{array}{c}
# 40x^3 - 40xy +2x - 2 \\20(y-x^2)
# \end{array}\right]$$
# and Hessian
# $$\nabla^2 f = \left[
# \begin{array}{c}
# 120x^2-40y+2 & -40x \\-40x & 20
# \end{array}\right]$$
# The only minimum is at $(x,y)=(1,1)$ where $f(1,1)=0$.
# 

def objfun(x,y):
    return 100*(y-x**2)**2 + (1-x)**2
def gradient(x,y):
    return np.array([-40*x*y + 40*x**3 -2 + 2*x, 20*(y-x**2)])
def hessian(x,y):
    return np.array([[120*x*x - 40*y+2, -40*x],[-40*x, 20]])


# Create a utility function that plots the contours of the Rosenbrock function.
# 

def contourplot(objfun, xmin, xmax, ymin, ymax, ncontours=50, fill=True):

    x = np.linspace(xmin, xmax, 300)
    y = np.linspace(ymin, ymax, 300)
    X, Y = np.meshgrid(x,y)
    Z = objfun(X,Y)
    if fill:
        plt.contourf(X,Y,Z,ncontours); # plot the contours
    else:
        plt.contour(X,Y,Z,ncontours); # plot the contours
    plt.scatter(1,1,marker="x",s=50,color="r");  # mark the minimum


# Here is a contour plot of the Rosenbrock function, with the global minimum marked with a red cross.
# 

conts = sorted(set([objfun(2,y) for y in np.arange(-2,5,0.25)]))
contourplot(objfun, -3,3, -2, 5, ncontours=conts,fill=False)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contours of $f(x,y)=100(y-x^2)^2 + (1-x)^2$");


# # Coordinate descent with fixed step length
# 

# First we write a function that uses the coordinate descent method. Initializes the solution at position `init`, moves along each dimension using the gradient vector with steps `steplength`, until the absolute difference between function values drops below `tolerance` or until the number of iterations exceeds `maxiter`.
# 
# The function returns the array of all intermediate positions, and the array of function values.
# 

def coordinatedescent(objfun, gradient, init, dim=2, tolerance=1e-6, maxiter=10000, steplength=0.01):
    p = np.array(init)
    iterno=0
    endflag = False
    parray = [p]
    fprev = objfun(p[0],p[1])
    farray = [fprev]
    eye = np.eye(dim)
    while iterno < maxiter: # main loop
        for d in range(dim): # loop over dimensions
            g = gradient(p[0],p[1])
            p = p - steplength*g[d]*eye[d]
            fcur = objfun(p[0], p[1])
            parray.append(p)
            farray.append(fcur)
            

        if abs(fcur-fprev)<tolerance:
            break
        fprev = fcur
        iterno += 1
    return np.array(parray), np.array(farray)


# Now let's see how the coordinate descent method behaves with the Rosenbrock function.
# 

p, f = coordinatedescent(objfun, gradient, init=[2,4], steplength=0.005,maxiter=10000)


# Plot the convergence of the solution. Left: The solution points (white) superposed on the contour plot. The star indicates the initial point. Right: The objective function value at each iteration.
# 

plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -1,3,0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");

plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");


# Suppose we increase the step size from $\alpha=0.005$ to $\alpha=0.01$, and the trajectory of the solution gets weird.
# 

p, f = coordinatedescent(objfun, gradient, init=[2,4], steplength=0.01,maxiter=500)


plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -2,3,0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");

plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");


# Now the step size is larger, so the new position ends up in a location where the gradient is larger. Therefore the next step is even larger, and we observe a large jump across the middle hill. There the steps get smaller again, and the solution approaches the global minimum from the back.
# 
# Try a different starting location, where the gradient is larger. Now, the same $\alpha$ is too large; the step size increases at each iteration and the calculation blows up.
# 

p, f = coordinatedescent(objfun, gradient, init=[2,6], steplength=0.01)


# We see that the function value increases rapidly at each iteration. The algorithm is unstable.
# 

f


# However, when $\alpha$ is decreased to $0.005$ again, we see that the solution converges after some oscillations.
# 

p, f = coordinatedescent(objfun, gradient, init=[2,6], steplength=0.005)


plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -2,3,0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");

plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");


# In general, the convergence depends sensitively on the $\alpha$ value as well as the local gradient value at the initial position. You can play with various initial positions and step lengths to see how it works.
# 

p, f = coordinatedescent(objfun, gradient, init=[2,5.1155], steplength=0.01)


plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -3,3,0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");

plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");





# # Introduction 
# 

# We have seen that Newton's method is very successful even with a troublesome function such as Rosenbrock. However, the true Hessian can be expensive and/or difficult to evaluate. _Quasi-Newton_ methods use approximations to the Hessian, easier to evaluate, and still converge quickly.
# 

get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np


# Consider the Rosenbrock function
# $$f(x,y)=10(y-x^2)^2 + (1-x)^2$$
# with gradient
# $$\nabla f = \left[\begin{array}{c}
# 40x^3 - 40xy +2x - 2 \\20(y-x^2)
# \end{array}\right]$$
# and Hessian
# $$\nabla^2 f = \left[
# \begin{array}{c}
# 120x^2-40y+2 & -40x \\-40x & 20
# \end{array}\right]$$
# The only minimum is at $(x,y)=(1,1)$ where $f(1,1)=0$.
# 

def objfun(x,y):
    return 10*(y-x**2)**2 + (1-x)**2
def gradient(x,y):
    return np.array([-40*x*y + 40*x**3 -2 + 2*x, 20*(y-x**2)])
def hessian(x,y):
    return np.array([[120*x*x - 40*y+2, -40*x],[-40*x, 20]])


# Create a utility function that plots the contours of the Rosenbrock function.
# 

def contourplot(objfun, xmin, xmax, ymin, ymax, ncontours=50, fill=True):

    x = np.linspace(xmin, xmax, 200)
    y = np.linspace(ymin, ymax, 200)
    X, Y = np.meshgrid(x,y)
    Z = objfun(X,Y)
    if fill:
        plt.contourf(X,Y,Z,ncontours); # plot the contours
    else:
        plt.contour(X,Y,Z,ncontours); # plot the contours
    plt.scatter(1,1,marker="x",s=50,color="r");  # mark the minimum


# Here is a contour plot of the Rosenbrock function, with the global minimum marked with a red cross.
# 

contourplot(objfun, -7,7, -10, 40, fill=False)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contours of $f(x,y)=10(y-x^2)^2 + (1-x)^2$");


# # Symmetric-rank-one (SR1) method
# 

# We write a function the SR1 quasi-Newton method to minimize a given objective function. Starts the solution at position `init`, moves along the quasi-Newton direction $-B_k^{-1}\nabla f$ until the absolute difference between function values drops below `tolerance` or until the number of iterations exceeds `maxiter`.
# 
# The approximate Hessian is initialized with an identity matrix.
# 
# The method is not self-starting: In order to calculate $B_1$, we need $x_1$ and $\nabla f_1$. So in the first step we do a simple gradient descent.
# 
# The step length $\alpha$ is not used here, effectively set to 1.
# 
# For efficiency, we use `np.linalg.solve` to determine the descent direction, instead of inverting the Hessian matrix. Inversion is no big deal in this 2D system, but that's a good habit to follow.
# 
# The function returns the array of all intermediate positions, and the array of function values.
# 

def sr1(objfun, gradient, init, tolerance=1e-6, maxiter=10000):
    x = np.array(init)
    iterno = 0
    B = np.identity(2)
    xarray = [x]
    fprev = objfun(x[0],x[1])
    farray = [fprev]
    gprev = gradient(x[0],x[1])
    xtmp = x - 0.01*gprev/np.sqrt(np.dot(gprev,gprev))
    gcur = gradient(xtmp[0],xtmp[1])
    s = xtmp-x
    y = gcur-gprev
    while iterno < maxiter:
        r = y-np.dot(B,s)
        B = B + np.outer(r,r)/np.dot(r,s)        
        x = x - np.linalg.solve(B,gcur)
        fcur = objfun(x[0], x[1])
        if np.isnan(fcur):
            break
        gprev = gcur
        gcur = gradient(x[0],x[1])
        xarray.append(x)
        farray.append(fcur)
        if abs(fcur-fprev)<tolerance:
            break
        fprev = fcur
        s = xarray[-1]-xarray[-2]
        y = gcur-gprev
        iterno += 1
    return np.array(xarray), np.array(farray)


# Now let's see how Newton's method behaves with the Rosenbrock function.
# 

p, f = sr1(objfun, gradient, init=[2,4])


f


# Plot the convergence of the solution. Left: The solution points (white) superposed on the contour plot. The star indicates the initial point. Right: The objective function value at each iteration.
# 

plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -1,3,0,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");

plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");


# The minimum is found in less than 20 iterations, with some zig-zagging around the minimum. Not as good as Newton's method, but definitely much better than steepest descent.
# 
# Now let's start at a more difficult location.
# 

p, f = sr1(objfun, gradient, init=[-1,9])


plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
contourplot(objfun, -3,3,-10,10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
plt.scatter(p[0,0],p[0,1],marker="*",color="w")
for i in range(1,len(p)):    
        plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");
plt.xlim(-3,3)
plt.ylim(-10,10)
        
plt.subplot(1,2,2)
plt.plot(f)
plt.xlabel("iterations")
plt.ylabel("function value");


# There are many large oscillations in the results. It may be the result of the denominator becoming almost zero, or the rsult of some mistake I've done. I'll return to this later.
# 




get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np
from time import sleep


# ##  Generate synthetic data
# 

N = 100 # count of data points
Xpos = np.random.multivariate_normal((1,1),[[1,0],[0,1]],int(N/2))
Xneg = np.random.multivariate_normal((-2,-2),[[1,0],[0,1]],int(N/2))


plt.scatter(Xpos[:,0],Xpos[:,1])
plt.scatter(Xneg[:,0],Xneg[:,1])
plt.legend(("Positive", "Negative"));


# Put the positive and negative instances together
# 

X = np.vstack((Xpos,Xneg))
y = np.hstack((np.ones(int(N/2)),np.zeros(int(N/2))))


# And shuffle them to remove any structure.
# 

ind = np.random.permutation(N)
X = np.hstack((np.ones((N,1)), X[ind,:])) # add the fixed column of ones for bias.
y = y[ind]


# ## Perceptron Learning Algorithm
# 

# Start with an arbitrary weight vector $w$.
# 

w = np.random.uniform(-1,1,3)
print(w)


xline = np.linspace(-4,3,100)
yline = -(w[1]*xline+w[0])/w[2]
plt.plot(xline, yline)
plt.scatter(Xpos[:,0],Xpos[:,1])
plt.scatter(Xneg[:,0],Xneg[:,1])
plt.legend(("wx=0","Positive", "Negative"));


xline = np.linspace(-4,3,100)

for repeat in range(50):
    predicted = [1 if i else 0 for i in np.dot(X,w)>0];
    misclassified = [i for i in range(N) if predicted[i]!=y[i]]
    if not misclassified:
        print("best solution found")
        break
    # Choose a random misclassified point
    i = np.random.choice(misclassified)
    # Update the weight vector
    w = w + 1*(y[i]-predicted[i])*X[i,:]
    #print(w)
    
    plt.figure()
    yline = -(w[1]*xline+w[0])/w[2]
    plt.plot(xline, yline)
    plt.scatter(Xpos[:,0],Xpos[:,1])
    plt.scatter(Xneg[:,0],Xneg[:,1])
    plt.legend(("wx=0","Positive", "Negative"))





# # Linear Regression to Quadratic Model Function
# 

# Suppose we apply linear regression to a model in the form
# $$f(x) = w_0 + w_1 x + w_2 x^2$$
# The function is not linear in $x$. However, the "linear" in "linear regression" refers to  linearity in parameters $w_i$. In that sense, linear regression can be applied. We just treat $x$ and $x^2$ as independent features.
# 

get_ipython().magic('matplotlib notebook')
import matplotlib.pylab as plt
plt.style.use("ggplot")
import numpy as np


# Generate synthetic data: 20 random $x$ values, and a quadratic function plus gaussian noise for $y$.
# 

x = np.random.uniform(-1,2,20)


y = x**2 - 2*x + 1 + np.random.normal(0,0.2,20)


# Plot the resulting data points.
# 

plt.scatter(x,y)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y");


# The original, nonlinear data has one feature, $x_1 = x$. We introduce a new feature  $x_2 = x^2$.
# 
# Visualize the $(x, x^2, y)$ data points with a 3D scatter plot.
# 

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x,x**2,y)


# By rotating the axes we find a viewpoint at which the points are on a line. Linear regression will give this hyperplane.
# 
# Apply linear regression:
# 

from sklearn import linear_model
reg = linear_model.LinearRegression()
X = np.vstack((x,x**2)).transpose()
reg.fit(X,y)


# Check how well the fit results agree with the parameters $w_0 = 1$, $w_1 = -2$, $w_2=1$.
# 

reg.intercept_, reg.coef_[0], reg.coef_[1]


# Now plot the data points together with the hyperplane resulting from the fit.
# 
# Rotate the figure to look at the plane from the side.
# 

fig = plt.figure()
ax = fig.gca(projection='3d')
Xf = np.arange(-1.1, 2.2, 0.25)
Yf = np.arange(-0.2, 4.25, 0.25)
Xf, Yf = np.meshgrid(Xf, Yf)
Zf = reg.intercept_ + reg.coef_[0]*Xf + reg.coef_[1]*Yf
surf = ax.plot_surface(Xf, Yf, Zf, alpha=0.2)
ax.scatter(x,x**2,y,c="r")


# A better visualization: Plot data points with stems coming out from the fit surface.
# 

yf = reg.predict(X)


fig = plt.figure()
 
ax = fig.gca(projection='3d')

Xf = np.arange(-1.1, 2.2, 0.25)
Yf = np.arange(-0.2, 4.25, 0.25)
Xf, Yf = np.meshgrid(Xf, Yf)
Zf = reg.intercept_ + reg.coef_[0]*Xf + reg.coef_[1]*Yf
surf = ax.plot_surface(Xf, Yf, Zf, alpha=0.2)

for i in range(len(X)):
    ax.plot([X[i,0], X[i,0]], [X[i,1],X[i,1]], [yf[i], y[i]], linewidth=2, color='r', alpha=.5)
ax.plot(X[:,0], X[:,1], y, 'o', markersize=8, 
        markerfacecolor='none', color='r')


# Now let's plot the 1D model function and the data points.
# 

plt.figure()
xx = np.linspace(-1,2)
ytrue = xx**2 - 2*xx + 1
yy = reg.coef_[1]*xx**2 + reg.coef_[0]*xx + reg.intercept_
plt.plot(xx, yy, "r")
#plt.plot(xx,ytrue, "g--")
plt.scatter(x,y)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y");





# Linear regression study with synthetic data.
# 

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as pl
from numpy.random import uniform, normal


# ## Synthetic data generation
# 

# Write a function that generates $M$ instances of data with $N$ features. The target function is linear in the features with some Gaussian noise added.
# 

def generate_data(M,N,theta,xmin=0,xmax=1,sigmay=1):
    """
    Generates synthetic data for linear regression.
  
    For each feature i, the feature values x_i are generated randomly between xmin[i] and xmax[i].
    The target variable y is a linear function of x_i with parameters theta, with a Gaussian noise added.
    
    y = theta_0 + theta_1 * x_1 + ... + theta_N * x_N + normal(0,sigmay)
    
    Parameters:
    
    M: Number of instances
    N: Number of features
    theta: the array of parameters (N+1 elements)
    xmin[i]: Minimum value of feature i. If scalar, same value is used for each feature.
    xmax[i]: Maximum value of feature i. If scalar, same value is used for each feature.
    sigmay: The standard deviation of the noise.
    
    Output:
    X_out: Design matrix
    y_out: Target array
    """
    if type(xmin)==int or type(xmin)==float:
        xmin = [xmin]*N
    if type(xmax)==int or type(xmax)==float:
        xmax = [xmax]*N

    assert len(xmin) == N
    assert len(xmax) == N
    assert len(theta) == N+1
    
    X_out = np.zeros((M,N+1))
    y_out = np.zeros(M)
    
    for m in range(M):
        x = [1] + [ uniform(xmin[i],xmax[i],1) for i in range(N) ]
        y = np.dot(x,theta) + normal(0,sigmay,1)
        X_out[m,:] = x
        y_out[m] = y
    
    return X_out, y_out


# Try it: Generate 50 data points with $y=2x+1+N(0,0.2)$
# 

X,y = generate_data(50,1,sigmay=0.2,theta=[1,2])


# Display the data points and the underlying model.
# 

pl.scatter(X[:,1],y)
pl.plot(X[:,1], np.dot(X,[1,2]))
pl.xlabel("$x_1$")
pl.ylabel("$y$")


# ## Linear regression
# 

# The linear model $h_\theta(X) = X\theta$
# 

def model(X, theta):
    """
    The linear model h(x) = theta_0 + theta_1*x_1 + theta_2*x_2 + ... theta_N*x_N
    X: The design matrix (M-by-(N+1), first column all 1's).
    theta: The parameter array (theta_0, theta_1, ..., theta_N)
    
    Returns:
    The array of model values, h = X.theta
    """
    assert X.shape[1] == len(theta)
    return np.dot(X,theta)


def cost(X, theta, y):
    """
    The cost function J(theta) = ||h(theta,X) - y||^2 / (2m)
    """
    m = len(y)
    diff = model(X,theta) - y
    return np.dot(diff, diff)/(2*m)


def gradient_descent(X, y, theta_0, alpha, eps = 1e-3, maxiter=1000):
    i = 0
    m = X.shape[0]
    J = [cost(X,theta_0,y)]
    while True:
        i += 1
        h = model(X, theta_0)
        theta = theta_0 - (alpha/m)*np.dot(np.transpose(h-y),X)
        J += [cost(X, theta, y)]
        diff = theta - theta_0
        if np.dot(diff,diff) < eps**2:
            break
        if i >= maxiter:
            print("Maximum iteration reached.")
            return None, J
        theta_0 = theta
    return theta, J


theta, J = gradient_descent(X,y,[2,3], 0.03)


pl.plot(J)


# Plot the data, original line, and the regression line.
# 

pl.scatter(X[:,1],y)
pl.plot(X[:,1], np.dot(X,[1,2]), label="original")
pl.plot(X[:,1], np.dot(X,th),label="regression")
pl.xlabel("$x_1$")
pl.ylabel("$y$")
pl.legend(loc="upper left");


# Solve using the normal equation and compare.
# 

from numpy.linalg import inv
theta_n = np.dot(np.dot(inv(np.dot(np.transpose(X),X)), np.transpose(X)),y)


pl.scatter(X[:,1],y)
pl.plot(X[:,1], np.dot(X,[1,2]), label="original")
pl.plot(X[:,1], np.dot(X,th),label="grad. desc.")
pl.plot(X[:,1], np.dot(X,theta_n),label="normal eq.")
pl.xlabel("$x_1$")
pl.ylabel("$y$")
pl.legend(loc="upper left");


# ## Contours of the cost function and iteration steps
# 

# Plot the contours of the cost function $J(\theta_0, \theta_1)$ in $(\theta_0,\theta_1)$ space.
# 

theta_0 = np.linspace(-3,4,50)
theta_1 = np.linspace(-2,6,50)
t0, t1 = np.meshgrid(theta_0,theta_1)
J = np.zeros((50,50))
for i in range(50):
    for j in range(50):
        J[i,j] = cost(X, [t0[i,j], t1[i,j]], y)


pl.contourf(t0,t1,J,100)
pl.xlabel(r"$\theta_0$")
pl.ylabel(r"$\theta_1$")
pl.title("Contours of " + r"$J( \theta_0, \theta_1)$");


# Modify the gradient descent function to output the intermediate parameters.
# 

def gradient_descent(X, y, theta_0, alpha, eps = 1e-5, maxiter=1000):
    i = 0
    m = X.shape[0]
    theta_array = [theta_0]
    J = [cost(X,theta_0,y)]
    while True:
        i += 1
        h = model(X, theta_0)
        theta = theta_0 - (alpha/m)*np.dot(np.transpose(h-y),X)
        J += [cost(X, theta, y)]
        theta_array += [theta]
        diff = abs(J[-1] - J[-2])
        if diff < eps:
            break
        if i >= maxiter:
            print("Maximum iteration reached.")
            break
        theta_0 = theta
    return np.array(theta_array), J


theta, J = gradient_descent(X,y,[3,5], 1)


pl.plot(J);


theta_0 = np.linspace(-3,4,50)
theta_1 = np.linspace(-2,6,50)
t0, t1 = np.meshgrid(theta_0,theta_1)
J = np.zeros((50,50))
for i in range(50):
    for j in range(50):
        J[i,j] = cost(X, [t0[i,j], t1[i,j]], y)


# Plot the contours and the gradient-descent steps.
# 

pl.contourf(t0,t1,J,100)
pl.xlabel(r"$\theta_0$")
pl.ylabel(r"$\theta_1$")
pl.title("Contours of " + r"$J( \theta_0, \theta_1)$");
pl.plot(theta[:,0],theta[:,1],"r.-");



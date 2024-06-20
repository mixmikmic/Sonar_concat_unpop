# # Test the projection convergence of our basis constructed using OMP
# 

import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')


# ### Collective OMP algorithm
# 
# We construct a measurement basis $W_m$ based on a given approximation basis $V_n$. Our ambient space $V$ is $H_0^1([0,1])$. Here $V_n$ is the sinusoids, normalised in $H_0^1([0,1])$, so $V_n = \mathrm{span}\{\phi_1,\ldots,\phi_n\}$, where $\phi_k = \frac{\sqrt{2}}{\pi k} \sin(k \pi x)$.
# 
# The measurements are assumed to be point evaluations, which have representer in $H_0^1$ of
# $$
# \omega_{x_0}(x) = \frac{1}{\sqrt{x_0 (1-x_0)}}
# \begin{cases}
# x (1 - x_0) & \text{for } x \le x_0 \(1 - x) x_0 & \text{for } x > x_0
# \end{cases}
# $$
# 
# This implementation of the algorithm looks at the best choice of $\omega$ from the dictionary $\mathcal{D}$ against the entire basis $V_n$, hence is called the _collective_ approach. That is, at each step of the algorithm choose
# $$
# \omega_k 
# = \mathrm{argmax}_{\omega\in\mathcal{D}} |\left\langle \omega, v - P_{W_{k-1}} v \right\rangle| 
# = \mathrm{argmax}_{\omega\in\mathcal{D}} \| P_{V_n} (\omega - P_{W_{k-1}} \omega ) \|
# = \mathrm{argmax}_{\omega\in\mathcal{D}} \sum_{i=1}^n |\left\langle \phi_i - P_{W_{k-1}}\phi_i, \omega \right\rangle|^2
# $$
# It is precisely the last expression on the right that is used in the code.
# 
# ### Lets look at $\beta(V_n, W_m)$ for our collective OMP basis and a random basis for comparison
# Note that this calculation is done for a small dictionary that only has $N=10^3$ elements, to save time
# 

N = 1e3
dictionary = pat.make_unif_dictionary(N)

ns = [10,20,40]
np.random.seed(3)
#n = 20
m = 200
bs_omp = np.zeros((len(ns), m))
bs_rand = np.zeros((len(ns), m))

Vn = pat.make_sin_basis(ns[-1])
Wms_omp = []
Wms_rand = []

for j, n in enumerate(ns):

    gbc = pat.CollectiveOMP(dictionary, Vn.subspace(slice(0,n)), verbose=True)
    Wm_omp = gbc.construct_to_m(m)
    Wms_omp.append(Wm_omp)
    Wm_omp_o = Wm_omp.orthonormalise()

    Wm_rand = pat.make_random_delta_basis(m)
    Wms_rand.append(Wm_rand)
    Wm_rand_o = Wm_rand.orthonormalise()

    BP_omp = pat.BasisPair(Wm_omp_o, Vn)
    BP_rand = pat.BasisPair(Wm_rand_o, Vn)
    for i in range(n, m):
        print('FB step ' + str(i))
        BP_omp_s = BP_omp.subspace(Wm_indices=slice(0,i), Vn_indices=slice(0,n)) #pat.BasisPair(Wm_omp_o.subspace(slice(0,i)), Vn.subspace(slice(0,n)))
        FB_omp = BP_omp_s.make_favorable_basis()
        bs_omp[j, i] = FB_omp.beta()

        BP_rand_s = BP_rand.subspace(Wm_indices=slice(0,i), Vn_indices=slice(0,n)) #pat.BasisPair(Wm_rand_o.subspace(slice(0,i)), Vn.subspace(slice(0,n)))
        FB_rand = BP_rand_s.make_favorable_basis()
        bs_rand[j, i] = FB_rand.beta()


sns.set_palette("deep")
cp = sns.color_palette()

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title='beta(Vn, Wm) against m for various n')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')

for i, n in enumerate(ns):
    plt.plot(range(m), bs_omp[i, :], label='omp Wm for n={0}'.format(n))#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))
    plt.plot(range(m), bs_rand[i, :], label='random Wm for n={0}'.format(n))#r'Random $W_m$, $n={{{0}}}$'.format(n))

ax.set(xlabel='m', ylabel='beta(Vn, Wm)')#r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=4)
plt.show()


# ## Take the case $n=20$, lets inspect the actual evaluation points for the basis $W_m$
# If $W_m = \mathrm{span}\{\omega_{x_1},\ldots,\omega_{x_m}\}$, where $\langle \omega_{x_k}, f\rangle = f(x_k)$, then what is our set of $x_k$?
# 

sns.set_palette("deep")
cp = sns.color_palette()

Wm_omp = Wms_omp[1]
Vn = Vn.subspace(slice(0, 20))
b_omp = bs_omp[1,:]
b_rand = bs_rand[1,:]

n=20
m=200

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$'.format(n))

plt.plot(range(n,m), b_omp[n:], label=r'OMP constructed $W_m$')
plt.plot(range(n,m), b_rand[n:], label=r'Random $W_m$')

ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=2)
plt.show()


# Plot the evaluation points in the Wm_rand basis 
# (note that the basis is infact orthonormalised so this isn't *quite* an accurate picture)
Wm_points = [vec.elements.values_array()[0].keys_array() for vec in Wm_omp.vecs]

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$ for OMP basis, with eval points'.format(n))
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$ and point locations')
plt.plot(range(n,n+40), b_omp[20:60], color=cp[0], label=r'$\beta(V_n, W_m)$ for OMP $W_m$')

plt.plot(n * np.ones(n-1), Wm_points[:n-1], 'o', color=cp[4], markersize=4, label='eval point')
plt.plot(n, Wm_points[n-1], 'o', color=cp[2], markersize=6, label='New eval point')
for m_plot in range(n, n+40-1):
    plt.plot((m_plot+1) * np.ones(m_plot), Wm_points[:m_plot], 'o', color=cp[4], markersize=4)
    plt.plot(m_plot+1, Wm_points[m_plot], 'o', color=cp[2], markersize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# Plot the evaluation points in the Wm_rand basis 
# (note that the basis is infact orthonormalised so this isn't *quite* an accurate picture)
Wm_points = [vec.elements.values_array()[0].keys_array() for vec in Wm_rand.vecs]
Wm_o_coeffs = [vec.elements.values_array()[0].values_array() for vec in Wm_rand_o.vecs]

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$ for random basis, with eval points'.format(n))
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$ and point locations')
plt.plot(range(n,n+40), b_rand[20:60], color=cp[1], label=r'$\beta(V_n, W_m)$ for random $W_m$')

plt.plot(n * np.ones(n-1), Wm_points[:n-1], 'o', color=cp[4], markersize=4, label='eval point')
plt.plot(n, Wm_points[n-1], 'o', color=cp[2], markersize=6, label='New eval point')
for m_plot in range(n, n+40-1):
    plt.plot((m_plot+1) * np.ones(m_plot), Wm_points[:m_plot], 'o', color=cp[4], markersize=4)
    plt.plot(m_plot+1, Wm_points[m_plot], 'o', color=cp[2], markersize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


bs_unif_int = np.zeros((len(ns), m))
Vn = pat.make_sin_basis(ns[-1])

Wms_unif_int = []

for j, n in enumerate(ns):
    for i in range(n, m):
        
        Wm_unif_int = pat.Basis([pat.FuncVector(params=[[x]],coeffs=[[1.0]],funcs=['H1UIDelta']) for x in np.linspace(0.0, 1.0, i, endpoint=False)+0.5/i])
        Wm_unif_int_o = Wm_unif_int.orthonormalise()

        BP_ui = pat.BasisPair(Wm_unif_int_o, Vn.subspace(slice(0,n)))
        FB_ui = BP_ui.make_favorable_basis()
        bs_unif_int[j, i] = FB_ui.beta()


n = ns[1]
Wm_omp = Wms_omp[1]
Vn = Vn.subspace(slice(0, 20))
b_omp = bs_omp[1,:]
b_rand = bs_rand[1,:]
b_ui = bs_unif_int[1,:]

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$'.format(n))

plt.plot(range(n,m), b_omp[n:], label=r'OMP constructed $W_m$')
plt.plot(range(n,m), b_rand[n:], label=r'Random $W_m$')
plt.plot(range(n,m), b_ui[n:], label=r'Uniformly spaced $W_m$')

ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=2)
plt.show()


# ## Lets look at some bases constructed on the cluster, and examine $m$ vs $n$ for a fixed minimum $\beta(V_n,W_m)$
# Here we have used a large dictionary, where $\mathcal{D}$ has $N = 10^6$ elements. Surprisingly we get very very good results: $\beta(V_n, W_m)\to 1$ quite quickly
# 

m=200
ns = [5, 10, 20, 40]#, 100]
bs_unif = np.zeros((len(ns), m))
bs_rand = np.zeros((len(ns), m))
bs_arb = np.zeros((len(ns), m))

gammas = np.arange(0., 1.1, 0.1)
m_gammas_unif = np.zeros((len(ns), len(gammas)))
m_gammas_rand = np.zeros((len(ns), len(gammas)))
m_gammas_arb = np.zeros((len(ns), len(gammas)))

for j, n in enumerate(ns):
    Vn = pat.make_sin_basis(n)
    
    omp_unif_x = np.load('omp_x_unif_{0}_10000.npy'.format(n))
    Wm_omp_unif = pat.Basis(vecs=[pat.FuncVector([[x]], [[1.0]], ['H1UIDelta']) for x in omp_unif_x])
    Wm_omp_unif_o = Wm_omp_unif.orthonormalise()

    omp_rand_x = np.load('omp_x_rand_{0}_10000.npy'.format(n))
    Wm_omp_rand = pat.Basis(vecs=[pat.Vector([[x]], [[1.0]], ['H1UIDelta']) for x in omp_rand_x])
    Wm_omp_rand_o = Wm_omp_rand.orthonormalise()

    Wm_arb = pat.make_random_delta_basis(m)
    Wm_arb_o = Wm_arb.orthonormalise()
    
    for i in range(n, m):
        BP_unif = pat.BasisPair(Wm_omp_unif_o.subspace(slice(0,i)), Vn)
        FB_unif = BP_unif.make_favorable_basis()
        bs_unif[j,i] = FB_unif.beta()

        BP_rand = pat.BasisPair(Wm_omp_rand_o.subspace(slice(0,i)), Vn)
        FB_rand = BP_rand.make_favorable_basis()
        bs_rand[j,i] = FB_rand.beta()
    
        BP_arb = pat.BasisPair(Wm_arb_o.subspace(slice(0,i)), Vn)
        FB_arb = BP_arb.make_favorable_basis()
        bs_arb[j,i] = FB_arb.beta()
    
    # Make the pivot data - the minimum m to reach some beta
    for i, gamma in enumerate(gammas):
        
        m_gammas_unif[j, i] = np.searchsorted(bs_unif[j,:], gamma)
        m_gammas_rand[j, i] = np.searchsorted(bs_rand[j,:], gamma)
        m_gammas_arb[j, i] = np.searchsorted(bs_arb[j,:], gamma)
        


sns.set_palette("deep")
cp = sns.color_palette()

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ for OMP with large dictionary ($N=10^6$)')

for i, n in enumerate(ns):
    
    plt.plot(range(m), bs_unif[i, :], label=r'OMP unif dict', color=cp[i])
    plt.plot(range(m), bs_rand[i, :], ':', label=r'OMP rand dict', color=cp[i])
    plt.plot(range(m), bs_arb[i, :], '--', label=r'Random $W_m$', color=cp[i])
    
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=2)
plt.show()

"""THIS PLOT BELOW IS INTERESTING BUT CONFUSING: COMMENTED OUT FOR NOW
axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'Minimum $m$ to attain $\gamma$'.format(n))

for i, n in enumerate(ns):
    
    plt.plot(gammas, m_gammas_unif[i, :], label=r'OMP unif dict', color=cp[i])
    #plt.plot(gammas, m_gammas_rand[i, :], ':', label=r'OMP rand dict', color=cp[i])
    plt.plot(gammas, m_gammas_arb[i, :], '--', label=r'Random $W_m$', color=cp[i])
    
ax.set(xlabel=r'$\gamma$', ylabel=r'$\mathrm{argmin}\{m : \beta(V_n, W_m) > \gamma \}$')
plt.legend(loc=2)
plt.show()
"""


# ## Ok, lets look quickly at the points generated with the large dictionary, versus the points generated above with the small dictionary
# We actually see that the results for $\beta(V_n,W_m)$ are almost identical for the two dictionaries, which is encouraging.
# 

sns.set_palette("deep")
cp = sns.color_palette()

Wm_omp = Wms_omp[1]
Vn = Vn.subspace(slice(0, 20))
b_omp = bs_omp[1,:]
n=20

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$, comparing results for different dictionary sizes'.format(n))

plt.plot(range(m), b_omp[:], label=r'Small dictionary')
plt.plot(range(m), bs_unif[2, :], label=r'Large dictionary')

ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=2)
plt.show()

# Plot the evaluation points in the Wm_omp basis - generated from a small dictionary
Wm_points = [vec.elements[0][0] for vec in Wm_omp.vecs]

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$ for small dictionary OMP basis, with eval points'.format(n))
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$ and point locations')
plt.plot(range(n,n+40), b_omp[20:60], color=cp[1], label=r'$\beta(V_n, W_m)$ for OMP $W_m$')

plt.plot(n * np.ones(n-1), Wm_points[:n-1], 'o', color=cp[4], markersize=4, label='eval point')
plt.plot(n, Wm_points[n-1], 'o', color=cp[2], markersize=6, label='New eval point')
for m_plot in range(n, n+40-1):
    plt.plot((m_plot+1) * np.ones(m_plot), Wm_points[:m_plot], 'o', color=cp[4], markersize=4)
    plt.plot(m_plot+1, Wm_points[m_plot], 'o', color=cp[2], markersize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# Now for the Wm_omp_unif basis - generated from a large dictionary
omp_unif_x = np.load('omp_x_unif_{0}_10000.npy'.format(n))
Wm_omp_unif = pat.Basis(vecs=[omp.Vector([x], [1.0], ['H1delta']) for x in omp_unif_x])

Wm_points = [vec.elements[0][0] for vec in Wm_omp_unif.vecs]

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$ for large dictionary OMP basis, with eval points'.format(n))
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$ and point locations')
plt.plot(range(n,n+40), bs_unif[2, 20:60], color=cp[1], label=r'$\beta(V_n, W_m)$ for OMP $W_m$')

plt.plot(n * np.ones(n-1), Wm_points[:n-1], 'o', color=cp[4], markersize=4, label='eval point')
plt.plot(n, Wm_points[n-1], 'o', color=cp[2], markersize=6, label='New eval point')
for m_plot in range(n, n+40-1):
    plt.plot((m_plot+1) * np.ones(m_plot), Wm_points[:m_plot], 'o', color=cp[4], markersize=4)
    plt.plot(m_plot+1, Wm_points[m_plot], 'o', color=cp[2], markersize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

print(bs_unif[2, 19:30])





# $\def \dot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \adot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \cD {\mathcal{D}}$
# $\def \bc {\mathbf{c}}$
# $\def \bv {\mathbf{v}}$
# $\def \bG {\mathbf{G}}$
# 
# # Greedy algorithms with a 4 dimensional field
# 
# Here we consider the solutions of the PDE $u_h(a(y))$ where $y\in [-1,1]^4$, $a(y) = \bar{a} + c \sum_{i=1}^4 y_i \chi_{D_i}(x)$, where the $D_i$ are partitions of the unit square in to the 4 even sub-squares.
# 
# We're given our measurement space $W_m = \mathrm{span}\{w_1,\ldots,w_m\}$. We have a series of measurements $\langle w_i, u\rangle_V$, and we write $w := P_{W_m} u$, the projection of $u$ in $W_m$. We try random and evenly-spaced measurements. Applications of the greedy algorithm reveal the dimensionality, and we then try domain-decomposition solutions.
# 

import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')

def make_soln(points, fem_div, a_bar=1.0, c=0.5, f=1.0, verbose=False):
    
    solns = []
    fields = []

    for p in points:
        field = pat.PWConstantSqDyadicL2(a_bar + c * p.reshape((2,2)))
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        solns.append(fem_solver.u)
        
    return solns, fields


# ### Generate the solution $u$ that we want to approximate
# 

fem_div = 8

a_bar = 0.1
c = 2.0

np.random.seed(2)

y = np.random.random((1,4))

u, a = make_soln(y, fem_div, a_bar=a_bar, c=c)
u = u[0]
a = a[0]

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
a.plot(ax, title='Example field $a(y)$ with $y\in\mathbb{R}^2$')
ax = fig.add_subplot(1, 2, 2, projection='3d')
u.plot(ax, title='FEM solution $u_h(a(y))$')
plt.show()


# ### Generate the basis $W_m$ of randomly placed local averages
# 

# local_width is the width of the measurement squares in terms of FEM mesh squares
width_div = 1
local_width = 2**width_div
spacing_div = 4

Wm_reg, Wloc_reg = pat.make_local_avg_grid_basis(width_div, spacing_div, fem_div, return_map=True)
Wm_reg = Wm_reg.orthonormalise()

m = Wm_reg.n
print('m =', m)

# We make the ambient spaces for Wm and Vn
np.random.seed(2)

Wm_rand, Wloc_rand = pat.make_pw_local_avg_random_basis(m=m, div=fem_div, width=local_width, return_map=True)
Wm_rand = Wm_rand.orthonormalise()

fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(Wloc_reg.values, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(Wloc_rand.values, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
ax.set_title('Measurement locations')
##plt.savefig('ddgrb_measurements.pdf')
plt.plot()


# ### Generate the dictionary of snapshots
# 
# Note that the $y$ that form the dictionary are on the regular ```dict_n``` $\times$ ```dict_N``` grid in $[0,1]^2$.
# 

dict_N = 10
dict_grid = np.linspace(0.0, 1.0, dict_N, endpoint=False)
y1s, y2s, y3s, y4s = np.meshgrid(dict_grid, dict_grid, dict_grid, dict_grid)

y1s = y1s.flatten()
y2s = y2s.flatten()
y3s = y3s.flatten()
y4s = y4s.flatten()

dict_ys = np.stack([y1s, y2s, y3s, y4s]).T
print('Making dictionary of length', len(dict_ys))
dictionary, dictionary_fields = make_soln(dict_ys, fem_div, a_bar=a_bar, c=c)


# In the following, 
#  - __```g```__ uses Algorithm 1
#  -  __```mbg_rand```__ uses Algorithm 2 with the randomly generated $W_m$ (```Wm_rand``` from above)
#  -  __```mbg_reg```__ uses Algorithm 2 with the regular $W_m$ (```Wm_reg``` from above)
#  -  __```mbgp_rand```__ uses Algorithm 3 with the random $W_m$
#  -  __```mbgp_reg```__ uses Algorithm 3 with the regular $W_m$
#  
# 

g = pat.GreedyApprox(dictionary, Vn=pat.PWBasis(), verbose=True, remove=False)
g.construct_to_n(m)

mbg_rand = pat.MeasBasedGreedy(dictionary, Wm_rand.dot(u), Wm_rand, Vn=pat.PWBasis(), verbose=True, remove=False)
mbg_rand.construct_to_n(m)

mbg_reg = pat.MeasBasedGreedy(dictionary, Wm_reg.dot(u), Wm_reg, Vn=pat.PWBasis(), verbose=True, remove=False)
mbg_reg.construct_to_n(m)

mbgp_rand = pat.MeasBasedGreedyPerp(dictionary, Wm_rand.dot(u), Wm_rand, Vn=pat.PWBasis(), verbose=True, remove=False)
mbgp_rand.construct_to_n(m)

mbgp_reg = pat.MeasBasedGreedyPerp(dictionary, Wm_reg.dot(u), Wm_reg, Vn=pat.PWBasis(), verbose=True, remove=False)
mbgp_reg.construct_to_n(m)

#Vn_sin = pat.make_pw_sin_basis(div=fem_div)


# ### So we see that all greedy algorithms select the same 3 points, that span all the dictionary points, and certainly also spans all of $\mathcal{M}$
# 

greedys = [g, mbg_rand, mbg_reg,mbgp_rand, mbgp_reg]
g_labels = ['Plain', 'Meas., Wm random', 'Meas. Wm regular', 'Perp. Wm random', 'Perp. Wm regular']

for i, greedy in enumerate(greedys):

    ps = dict_ys[np.array(greedy.dict_sel, dtype=np.int32), :]
    print(g_labels[i])
    print(ps)


sns.set_palette('hls', len(greedys))
sns.set_style('whitegrid')

fig = plt.figure(figsize=(7,7))

for i, greedy in enumerate(greedys):
    labels = ['{0} point {1}'.format(g_labels[i], j) for j in range(greedy.n)] 
    
    ps = dict_ys[np.array(greedy.dict_sel, dtype=np.int32), :]
    
    plt.scatter(ps[:, 0], ps[:, 1], marker='o')

    for label, x, y in zip(labels, ps[:, 0], ps[:, 1]):
        plt.annotate(
            label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom')

plt.show()


# ### Can we find a closed form for the coefficients of this 4d solution? We look at the coefficients of the sub-domain solutions
# 

np.random.seed(2)

y = np.random.random((1,4))
y = np.array([[0.1, 0.3, 0.6, 0.9]])

u, a = make_soln(y, fem_div, a_bar=a_bar, c=c)

u0, a0 = make_soln(y.mean()*np.ones((1,4)), fem_div, a_bar=a_bar, c=c)
u1, a1 = make_soln(np.array([[1,1,1e16,1e16]]), fem_div, a_bar=0, c=1)
u2, a2 = make_soln(np.array([[1e16,1,1e16,1]]), fem_div, a_bar=0, c=1)
u3, a3 = make_soln(np.array([[1e16,1e16,1,1]]), fem_div, a_bar=0, c=1)
u4, a4 = make_soln(np.array([[1,1e16,1,1e16]]), fem_div, a_bar=0, c=1)

u5, a5 = make_soln(np.array([[1,1e16,1e16,1e16]]), fem_div, a_bar=0, c=1)
u6, a6 = make_soln(np.array([[1e16,1,1e16,1e16]]), fem_div, a_bar=0, c=1)
u7, a7 = make_soln(np.array([[1e16,1e16,1,1e16]]), fem_div, a_bar=0, c=1)
u8, a8 = make_soln(np.array([[1e16,1e16,1e16,1]]), fem_div, a_bar=0, c=1)

# The forgotten corner solutions...?
u9, a9   = make_soln(np.array([[1,1,1,1e16]]), fem_div, a_bar=0, c=1)
u10, a10 = make_soln(np.array([[1e16,1,1,1]]), fem_div, a_bar=0, c=1)
u11, a11 = make_soln(np.array([[1,1e16,1,1]]), fem_div, a_bar=0, c=1)
u12, a12 = make_soln(np.array([[1,1,1e16,1]]), fem_div, a_bar=0, c=1)

u = u[0]
us=[]
us.append(u0[0])
us.append(u1[0])
us.append(u2[0])
us.append(u3[0])
us.append(u4[0])
us.append(u5[0])
us.append(u6[0])
us.append(u7[0])
us.append(u8[0])
#us.append(u9[0])
#us.append(u10[0])
#us.append(u11[0])
#us.append(u12[0])


print(y)
print(a[0].values)

fig = plt.figure(figsize=(15, 20))
ax = fig.add_subplot(4, 4, 1, projection='3d')
u.plot(ax, title='$u$')
ax = fig.add_subplot(4, 4, 2, projection='3d')
a[0].plot(ax, title='$a$')
for i,v in enumerate(us):
    ax = fig.add_subplot(4, 4, i+3, projection='3d')
    v.plot(ax, title=r'$u_{{{0}}}$'.format(i))

plt.show()


# ### Can we represent the solution $u$ in terms of the 9 components $u_0$ to $u_8$?
# If so my theory is correct
# 

M = np.vstack([v.values.flatten() for v in us])
w = u.values.flatten()

C = M @ M.T
g = M @ w

#cf = np.linalg.lstsq(M.T, w)[0]
#print(cf)
cf = np.linalg.solve(C, g)
print("Coefficients:", cf)

#print(M @ M.T)
lambdas, V =  np.linalg.eig(M @ M.T)
print("eigenvalues: ", lambdas)

u_rec = pat.PWLinearSqDyadicH1(us[0].values * cf[0])
for i,v in enumerate(us[1:]):
    u_rec += v * cf[i+1]

print((u - u_rec).values.max())
print(y)

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
(u-u_rec).plot(ax, title='$u$')
#ax = fig.add_subplot(2, 2, 2, projection='3d')
#corner.plot(ax, title='')
#ax = fig.add_subplot(2, 2, 3, projection='3d')
#u.plot(ax, title='')
#ax = fig.add_subplot(2, 2, 4, projection='3d')
#u5.plot(ax, title='')
plt.show()


us_basis = pat.PWBasis(us)

lambdas, V =  np.linalg.eig(M @ M.T)
print("eigenvalues: ", lambdas)


fig = plt.figure(figsize=(15, 20))

for i, v in enumerate(V.T):
    ax = fig.add_subplot(4, 4, i+1, projection='3d')
    us_basis.reconstruct(v).plot(ax, title='$u$')


# # Investigating snapshot bases
# 
# A few tests here. First we investigate the sparsity claims about a random snapshot basis. If we have an understanding of the decay of coefficients for a projection, maybe this will point us in the right direction for greedy approaches to building the space, or whether there is any hope that some sort of compressed sensing type approach will work for this problem.
# 

import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')


# ### Snapshot basis
# 
# We consider the usual PDE problem 
# $$-\mathrm{div}(a(y) \nabla u) = f$$
# on the doman $D=[0,1]^2$ and take $a(y)$ to be the "checkerboard" random field. 
# 
# The space we operate in is $V = H_0^1([0,1]^2)$.
# 
# Our approximation space is the random snapshot basis, $V_n = \mathrm{span}\{ u_h(a(y_1)), \ldots, u_h(a(y_n)) \}$.
# 

fem_div = 7

a_bar = 1.0
c = 0.9
field_div = 2
side_n = 2**field_div

np.random.seed(3)
point_gen = pat.MonteCarlo(d=side_n*side_n, n=1, lims=[-1, 1])
a = pat.PWConstantSqDyadicL2(a_bar + c * point_gen.points[0,:].reshape([side_n, side_n]), div=field_div)
fem = pat.DyadicFEMSolver(div=fem_div, rand_field=a, f=1.0)
fem.solve()

u = fem.u

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
a.plot(ax, title='Random field $a$ on dyadic level 2 grid')
ax = fig.add_subplot(1, 2, 2, projection='3d')
fem.u.plot(ax, title='FEM solution $u_h(a(y))$ (labelled $\mathtt{u\_16}$)')
plt.show()


ns = [10, 20, 50, 100]
n = ns[-1]

Vn_sin = pat.make_pw_sin_basis(div=fem_div)
Vn_red, fields = pat.make_pw_reduced_basis(n, field_div=field_div, fem_div=fem_div)

cs_sin = []
cs_red = []

for i in range(len(ns)):
    Pu_red, c_red = Vn_red.subspace(slice(0,ns[i])).project(u, return_coeffs=True)
    cs_red.append(c_red)
    Pu_sin, c_sin = Vn_sin.subspace(slice(0,ns[i])).project(u, return_coeffs=True)
    cs_sin.append(c_sin)
    


fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 1, 1, title=r'Projection coefficients sorted')

for i in range(len(ns)):
    plt.semilogy(np.sort(np.abs(cs_red[i]))[::-1], label='red {0}'.format(len(cs_red[i])))
    plt.semilogy(np.sort(np.abs(cs_sin[i]))[::-1], '--', label='sin {0}'.format(len(cs_sin[i])))
plt.legend()
plt.show()


# $\def \dot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \adot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \cA {\mathcal{A}}$
# $\def \cD {\mathcal{D}}$
# $\def \cM {\mathcal{M}}$
# $\def \cN {\mathcal{N}}$
# $\def \cW {\mathcal{W}}$
# $\def \bc {\mathbf{c}}$
# $\def \bu {\mathbf{u}}$
# $\def \bv {\mathbf{v}}$
# $\def \bw {\mathbf{w}}$
# $\def \bG {\mathbf{G}}$
# $\def \bC {\mathbf{C}}$
# $\def \bD {\mathbf{D}}$
# $\def \bI {\mathbf{I}}$
# $\def \bP {\mathbf{P}}$
# $\def \bQ {\mathbf{Q}}$
# $\def \bR {\mathbf{R}}$
# $\def \bS {\mathbf{S}}$
# $\def \bT {\mathbf{T}}$
# $\def \bU {\mathbf{U}}$
# $\def \bV {\mathbf{V}}$
# $\def \bW {\mathbf{W}}$
# $\def \bPhi {\mathbf{\Phi}}$
# $\def \bPsi {\mathbf{\Psi}}$
# $\def \bGamma {\mathbf{\Gamma}}$
# $\def \bSigma {\mathbf{\Sigma}}$
# $\def \bTheta {\mathbf{\Theta}}$
# $\def \bOmega {\mathbf{\Omega}}$
# $\def \bbE {\mathbb{E}}$
# $\def \bbP {\mathbb{P}}$
# $\def \bbR {\mathbb{R}}$
# $\def \bbN {\mathbb{N}}$
# 
# ### When $\sigma_i = 0$ for $n< i \le K$, then what does our pseudo-inverse approach yield?
# 
# Again all tests are in $\bbR^K$, and we have two random orthonormal bases and $(\psi_1,\ldots,\psi_K)$ and $(\varphi_1,\ldots,\varphi_K)$, along with the singular values / PCA values of $(\sigma_1,\ldots,\sigma_n,0,\ldots)$, i.e. we assume the PCA comes up short with only $n$ dimensions (or that they are so small they should be truncated, which helps for stability of the final systems).
# 
# We assume further that the measurement space $W = \mathrm{span}(\psi_1,\ldots,\psi_m)$ and $W_\perp = \mathrm{span}(\psi_{m+1},\ldots,\psi_K)$, and we write $V = \mathrm{span}(\varphi_1,\ldots,\varphi_n)$.
# 
# We have the matrix $\bPhi = [\varphi_1 \ldots \varphi_K]$ and $\bPsi = [\psi_1\,\ldots\,\psi_K]$, where the basis vectors are the columns of the matrices. Finally we write $\bW = [\psi_1\,\ldots\,\psi_m]$, $\bW_\perp = [\psi_{m+1}\,\ldots\,\psi_K]$ and $\bV = [\varphi_{1}\,\ldots\,\varphi_n]$. For the cross-Grammian we have $\bG = \bPhi^T \bPsi$.
# 
# Now, $\bT = \bG^T \bD^{-1} \bG = \bPsi^T \bPhi \bSigma^{-2} \bPhi^T \bPsi$, the diagonal matrix $\bSigma = \mathrm{diag}(\sigma_1,\ldots,\sigma_n,0\ldots)$. Note also that, using the above notation $\bPsi = \begin{bmatrix} \bW & \bW_\perp \end{bmatrix}$, so in fact
# 
# $$ \bG = \bPhi^T \bPsi = \begin{bmatrix} \bPhi^T\bW & \bPhi^T\bW_\perp \end{bmatrix} $$
# 
# and then
# $$ \bS = \begin{bmatrix} \bW^T\bPhi \\ \bW_\perp^T\bPhi \end{bmatrix} \bSigma^{2} \begin{bmatrix} \bPhi^T \bW & \bPhi^T \bW_\perp \end{bmatrix} = \begin{bmatrix} \bW^T \bPhi \bSigma^{2} \bPhi^T \bW & \bW^T \bPhi \bSigma^{2} \bPhi^T \bW_\perp \\ \bW_\perp^T \bPhi \bSigma^{2} \bPhi^T \bW & \bW_\perp^T \bPhi \bSigma^{2} \bPhi^T \bW_\perp \end{bmatrix} $$
# 
# and similar for $\bT$ (except with $\bSigma^{-2}$).
# 
# which give us the expressions for $\bS_{1,1}$, $\bT_{1,1}$, $\bT_{1,2}$ etc.. The issue here is that $\bS_{1,1}$ or $\bT_{2,2}$ may no longer invertible as 
# 
# $$\bS_{1,1} = \bW^T \bPhi \bSigma^{2} \bPhi^T \bW =\bW^T \bV \bSigma_n^{2} \bV^T \bW$$
# 
# $$\bT_{2,2} = \bW_\perp^T \bPhi \bSigma^{-2} \bPhi^T \bW_\perp =\bW_\perp^T \bV \bSigma_n^{-2} \bV^T \bW_\perp$$
# 
# which are both of rank at most $n$. 
# 

import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Latex, display

import sys

get_ipython().magic('matplotlib inline')
np.random.seed(1)


K = 100  # The dimensionality of the ambient space (can be up to 2^16 for FEM solutions)
n = K    # The truncation dimension of the PCA / embedding dimension of the manifold 
m = 6    # The dimension off the measurement space

# First make two random orthonormal vector bases
Phi = sp.stats.ortho_group.rvs(dim=K) # The "PCA" space
Psi = sp.stats.ortho_group.rvs(dim=K) # The "measurement" space

sigma = np.sort(np.random.random(n))[::-1]
sigma[n:] = 0
Sigma = np.pad(np.diag(sigma), ((0,K-n),(0,K-n)), 'constant')
Sigma_inv = np.pad(np.diag(1.0/sigma), ((0,K-n),(0,K-n)), 'constant')
Sigma_n = np.diag(sigma)
Sigma_n_inv = np.diag(1.0/sigma)

V = Phi[:,:n]
W = Psi[:,:m]
W_p = Psi[:,m:]

T = Psi.T @ Phi @ Sigma_inv @ Sigma_inv @ Phi.T @ Psi
S = Psi.T @ Phi @ Sigma @ Sigma @ Phi.T @ Psi


# ### First test (which appears to pass)
# Checking that $\bS_{1,1} = \bW^T \bPhi \bSigma^{2} \bPhi^T \bW =\bW^T \bV \bSigma_n^{2} \bV^T \bW$
# 
# and $\bT_{2,2} = \bW_\perp^T \bPhi \bSigma^{-2} \bPhi^T \bW_\perp =\bW_\perp^T \bV \bSigma_n^{-2} \bV^T \bW_\perp$, etc...
# 

T21 = T[m:, :m]
T22 = T[m:, m:]
S11 = S[:m, :m]
S21 = S[m:, :m]
S22 = S[m:, m:]

T22_alt = W_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W_p
T21_alt = W_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W
S11_alt = W.T @ V @ Sigma_n @ Sigma_n @ V.T @ W
S22_alt = W_p.T @ V @ Sigma_n @ Sigma_n @ V.T @ W_p
S21_alt = W_p.T @ V @ Sigma_n @ Sigma_n @ V.T @ W

print('T_21 of shape {0}, rank {1}, condition {2}'.format(T21.shape, np.linalg.matrix_rank(T21), np.linalg.cond(T21)))
print('T_22 of shape {0}, rank {1}, condition {2}\n'.format(T22.shape, np.linalg.matrix_rank(T22), np.linalg.cond(T22)))
print('S_11 of shape {0}, rank {1}, condition {2}'.format(S11.shape, np.linalg.matrix_rank(S11), np.linalg.cond(S11)))
print('S_21 of shape {0}, rank {1}, condition {2}'.format(S21.shape, np.linalg.matrix_rank(S21), np.linalg.cond(S21)))
print('S_22 of shape {0}, rank {1}, condition {2}\n'.format(S22.shape, np.linalg.matrix_rank(S22), np.linalg.cond(S22)))

# Just to check
display(Latex(r'$\left\| \bT_{{2,1}} - \bW_\perp^T \bV \bSigma^{{-2}} \bV^T \bW \right\|_F =$ {0}'.format(np.linalg.norm(T21 - T21_alt))))
display(Latex(r'$\left\| \bS_{{1,1}} - \bW^T \bV \bSigma^{{2}} \bV^T \bW \right\|_F =$ {0}'.format(np.linalg.norm(S11 - S11_alt))))
display(Latex(r'$\left\| \bS_{{2,1}} - \bW_\perp^T \bV \bSigma^{{2}} \bV^T \bW \right\|_F =$ {0}'.format(np.linalg.norm(S21 - S21_alt))))


# Right away above we see several advantages of using $\bS_{1,1}$ to solve the system - the condition number is better in the case of full rank, and the system is only $m\times m$ in size.
# 
# ### Double checking SVD decompositions and psuedo-inverses
# 
# Because both $\bS_{1,1}$ and $\bT_{2,2}$ can be rank difficient if $n < m$ or $n < K-m$ respectively, so we need the pseudo-inverse, $\bS_{1,1}^\dagger$, to solve this system. Let us consider the SVD decomposition of 
# 
# $$(\bW^T \bV \bSigma_n) = \bP \bQ \bR \quad\text{giving us}\quad \bS_{1,1} = \bP \bQ \bQ^T \bP^T$$
# 
# As is usual, there is a stable inverse for $\bS_{1,1}$ in the span of the columns of $\bP$ up til at most $\min(n, m)$. Let us write that $\bP_{1:r}$ is the matrix made up of the first $r=\min(m,n)$ columns of $\bP$. Psuedo inverse would obv be of the form (assuming, e.g. that the rank of $\bS_{1,1}$ is $n$, and also writing $\bQ^{-2}$ for the appropriate diagonal matrix)
# 
# $$\bS^\dagger_{1,1} = \bP_{1:r} \bQ^{-2} (\bP_{1:r})^T$$
# 
# Similar calc applies for $\bT_{2,2}$, with the psuedo-inverse applying in the span of the SVD decomp of $(\bW^T \bV \bSigma_n^{-1})$, $\bP_\perp$ up til the $\min(K-m, n)$-th column.
# 

P, Q, RT = sp.linalg.svd(W.T @ V @ Sigma_n)
P_p, Q_p, R_pT = sp.linalg.svd(W_p.T @ V @ Sigma_n_inv)

rm = min(m,n)
rk = min(K-m, n)

T22_pinv = P_p[:,:rk] @ np.diag(1.0/(Q_p*Q_p)) @ P_p[:,:rk].T
S11_pinv = P[:,:rm] @ np.diag(1/(Q*Q)) @ P[:,:rm].T

display(Latex(r'$\left \| \bS_{{1,1}}^\dagger - \bP_{{1:n}} \bQ^{{-2}} (\bP_{{1:n}})^T \right\|_F =$  {0}'.format(np.linalg.norm(S11_pinv - np.linalg.pinv(S11)))))
display(Latex(r'$\left \| \bT_{{2,2}}^\dagger - \bP_{{\perp,1:n}} \bQ_\perp^{{-2}} (\bP_{{\perp,1:n}})^T \right\|_F =$  {0}'.format(np.linalg.norm(T22_pinv - np.linalg.pinv(T22)))))


# Given the SVD decompositions
# 
# $$ (\bW^T \bV \bSigma_n) = \bP \bQ \bR \quad\text{and}\quad(\bW_\perp^T \bV \bSigma_n) = \bP_\perp \bQ_\perp \bR_\perp $$
# 
# we find that (again with $r=\min(m,n)$)
# 
# \begin{align}
# \bS_{2,1} \bS_{1,1}^\dagger 
# &= \bP_\perp \bQ_\perp^{-1} \bR_\perp^T \bR \bQ \bP^T 
# \bP_{1:r} \bQ^{-2} (\bP_{1:r})^T \&= \bP_\perp \bQ_\perp^{-1} \bR_\perp^T \bR_{1:r} \bQ^{-1} (\bP_{1:r})^T \\end{align}
# 
# $$ \bw_\perp^* = \bS_{2,1} \bS_{1,1}^\dagger \bw = - \bT_{2,2}^\dagger \bT_{2,1} \bw $$
# 
# And once again there is a very similar calculation for $\bT_{2,2}^\dagger$.
# 

ra = min(K-m, n)
rm = min(m, n)

P, Q, RT = sp.linalg.svd(W.T @ V @ Sigma_n)
P_p, Q_p, R_pT = sp.linalg.svd(W_p.T @ V @ Sigma_n)
Ssolver_alt = P_p[:,:ra] @ np.diag(Q_p) @ R_pT[:ra] @ RT[:rm].T @ np.diag(1.0/Q) @ P[:,:rm].T
Ssolver = S21 @ np.linalg.pinv(S11)

print('(S21 * S11_inv) shape {0} condition {1}'.format(Ssolver.shape, np.linalg.cond(Ssolver)))
display(Latex(r'$\left\| \bS_{{2,1}} \bS_{{1,1}}^\dagger - (\bP_\perp)_{{1:n}} \bQ_\perp \bR_\perp^T \bR_W \bQ_W^{{-1}} (\bP_W)_{{1:n}}^T \right\|_F =${0}'.format(np.linalg.norm(Ssolver - Ssolver_alt))))

P, Q, RT = sp.linalg.svd(W.T @ V @ Sigma_n_inv)
P_p, Q_p, R_pT = sp.linalg.svd(W_p.T @ V @ Sigma_n_inv)
Tsolver_alt = P_p[:,:ra] @ np.diag(1.0/Q_p) @ R_pT[:ra] @ RT[:rm].T @ np.diag(Q) @ P[:,:rm].T
Tsolver = np.linalg.pinv(T22) @ T21

print('(T22_inv * T21) shape {0} condition {1}'.format(Tsolver.shape, np.linalg.cond(Tsolver)))
display(Latex(r'$\left\| \bT_{{2,2}}^\dagger \bT_{{2,1}} - (\bP_\perp)_{{1:n}} \bQ_\perp^{{-1}} \bR_\perp^T \bR_W \bQ_W (\bP_W)_{{1:n}}^T \right\|_F =${0}'.format(np.linalg.norm(Tsolver - Tsolver_alt))))

display(Latex(r'$\left\| \bS_{{2,1}} \bS_{{1,1}}^\dagger + \bT_{{2,2}}^\dagger \bT_{{2,1}} \right\|_F =$ {0}'.format(np.linalg.norm(Tsolver + Ssolver))))


print('That other thing Ive been semi hopeful for : {0}'.format(np.linalg.norm(Ssolver - Q.T @ P[:,:n].T)))


# ### So what are we actually getting to here? 
# 
# Well, the complexity of the SVD for really large $K$ is just stupendous. We want to consider maybe something like restrict $\bbR^K$ to $\bbR^{2m}$ spanned by $\psi_1,\ldots,\psi_{2m}$ to do this reconstruction, and we know that we'll have it within some amount...
# 
# In this case we simply have the "approximate" operators 
# 
# $$ \bT_{2,2}^{(\mathrm{app})} = (\bPsi_{m:2m})^T \bV \bSigma^{-2} \bV^T \bPsi_{m:2m} $$
# 
# and 
# 
# $$ \bT_{2,1}^{(\mathrm{app})} = \bPsi_{m:2m}^T \bV \bSigma^{-2} \bV^T \bPsi_{1:m} = \bPsi_{m:2m}^T \bV \bSigma^{-2} \bV^T \bW . $$ 
# 
# We see below we have much more manageable computations $\bT_{2,2}^{(\mathrm{app})}$ is now of shape $m \times m$.
# 

W2_p = Psi[:,m:2*m]

S11_approx = W.T @ V @ Sigma_n @ Sigma_n @ V.T @ W
S21_approx = W2_p.T @ V @ Sigma_n @ Sigma_n @ V.T @ W
print('S_11_approx of shape {0}, rank {1}, condition {2}'.format(S11_approx.shape, np.linalg.matrix_rank(S11_approx), np.linalg.cond(S11_approx)))
print('S_21_approx of shape {0}, rank {1}, condition {2}'.format(S21_approx.shape, np.linalg.matrix_rank(S21_approx), np.linalg.cond(S21_approx)))
Ssolver_approx = S21_approx @ np.linalg.pinv(S11_approx)

T22_approx = W2_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W2_p
T21_approx = W2_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W
print('T_21_approx of shape {0}, rank {1}, condition {2}'.format(T21_approx.shape, np.linalg.matrix_rank(T21_approx), np.linalg.cond(T21_approx)))
print('T_22_approx of shape {0}, rank {1}, condition {2}'.format(T22_approx.shape, np.linalg.matrix_rank(T22_approx), np.linalg.cond(T22_approx)))
Tsolver_approx = np.linalg.pinv(T22_approx) @ T21_approx

print('')
display(Latex(r'$\left\| \bS_{{2,1}} \bS_{{1,1}}^\dagger - \bS_{{2,1}}^{{(\mathrm{{app}})}} (\bS_{{1,1}}^{{(\mathrm{{app}})}})^\dagger \right\|_2 =$ {0}'.format(np.linalg.norm(W2_p @ Ssolver_approx - W_p @ Ssolver, ord=2))))
display(Latex(r'$\left\| \bT_{{2,2}}^\dagger \bT_{{2,1}} - (\bT_{{2,2}}^{{(\mathrm{{app}})}})^\dagger \bT_{{2,1}}^{{(\mathrm{{app}})}} \right\|_2 =$ {0}'.format(np.linalg.norm(W2_p @ Tsolver_approx - W_p @ Tsolver, ord=2))))


# ### Now we examine the approximation difference from $m+1$ to $K$
# 
# ...i.e. we take $\bW_\perp = \bPsi_{m:M}$ for $M = m+1, \ldots , K$
# 

Ssolver_acc = np.zeros(K-m)
Tsolver_acc = np.zeros(K-m)

for M in range(m+1,K):
    
    W2_p = Psi[:,m:M]
    S11_approx = W.T @ V @ Sigma_n @ Sigma_n @ V.T @ W
    S21_approx = W2_p.T @ V @ Sigma_n @ Sigma_n @ V.T @ W
    T22_approx = W2_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W2_p
    T21_approx = W2_p.T @ V @ Sigma_n_inv @ Sigma_n_inv @ V.T @ W

    Ssolver_approx = S21_approx @ np.linalg.pinv(S11_approx)
    Tsolver_approx = np.linalg.pinv(T22_approx) @ T21_approx

    Ssolver_acc[M-(m+1)] = np.linalg.norm(W2_p @ Ssolver_approx - W_p @ Ssolver, ord=2)
    Tsolver_acc[M-(m+1)] = np.linalg.norm(W2_p @ Tsolver_approx - W_p @ Tsolver, ord=2)

plt.figure(figsize=(10, 7))
plt.plot(range(m+1, K+1), Ssolver_acc, label=r'$S_{2,1} S_{1,1}^{-1}$')
plt.plot(range(m+1, K+1), Tsolver_acc, label=r'$T_{2,2}^{-1} T_{2,1}$')
plt.legend(loc=1)
plt.xlabel(r'Dim of $W + W_\perp$')
plt.ylabel(r'$||$ Solver - Approx Solver $||_2$')
plt.title(r'Meas space $W$ of dim $m=${0}, PCA space of dim $n=${1}'.format(m,n))
plt.show()





# $\def \dot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \adot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \cD {\mathcal{D}}$
# $\def \bc {\mathbf{c}}$
# $\def \bv {\mathbf{v}}$
# $\def \bG {\mathbf{G}}$
# 
# # Greedy algorithms with a 2d manifold - L-shaped domain division
# 
# Here we consider the solutions of the PDE $u_h(a(y))$ where $y\in\mathbb{R}^2$, $a(y) = y_1 \chi_{D_1}(x) + y_2 \chi_{D_2}(x)$, and $D_1 = [0,1/2) \times [0,1] \cup [1/2,1] \times [0,1/2)$ and $D_2 = [1/2, 1] \times [1/2,1]$, and $\chi_{D_1}$, $\chi_{D_2}$ are the indicator functions on $D_1$, $D_2$.
# 
# We're given our measurement space $W_m = \mathrm{span}\{w_1,\ldots,w_m\}$. We have a series of measurements $\langle w_i, u\rangle_V$, and we write $w := P_{W_m} u$, the projection of $u$ in $W_m$. We try random, even, and even sinusoidal measurements.
# 
# __Remarks:__
#  - __Note that $\dot{\cdot}{\cdot} = \dot{\cdot}{\cdot}_{V_h}$ here.__
#  - We can __assume that the $\omega_i$ are orthonormal__ as it is a fixed basis.
#  - We store $\dot{\omega_i}{v}$ for each $v\in \cD$, so can we di all projections and inner-products in $\mathbb{R}^m$? And furthermore without the need of a Gram matrix $\dot{\phi_i}{\phi_j}$ to do the projection or orthonormalisation of $\phi_i$?
#  - Finally, remember that _we can not know anything about $u$ other than $w :=P_{W_m} u$_, and we abuse notation slightly and also write $w=\dot{\omega_i}{u}$ for the vector in $\mathbb{R}^m$.
# 
# We have a dictionary $\cD$ of solutions $v(y)$, which we get from generating points in $y\in\mathbb{R}^{2}$ and then $v(y) = \frac{u_h(a(y))}{\| u_h(a(y)) \|_{V_h}}$ (i.e. we normalise all elements). 
# 
# __Algorithms 1-3, the "pure greedy", "measurement based OMP" and "measurement based PP" are used here.__
# 

import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')

def make_2d_param_soln(points, fem_div, a_bar=1.0, c=0.5, f=1.0, verbose=False):
    
    solns = []
    fields = []

    for p in points:
        field = pat.PWConstantSqDyadicL2(a_bar + c * np.array([[p[0], p[0]],[p[0], p[1]]]))
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        solns.append(fem_solver.u)
        
    return solns, fields


# ### Generate the solution $u$ that we want to approximate
# 

fem_div = 7

a_bar = 0.1
c = 2.0

np.random.seed(2)

y = np.array([[0.8, 0.1]])
print(y[0])
u, a = make_2d_param_soln(y, fem_div, a_bar=a_bar, c=c)
u = u[0]
a = a[0]

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
a.plot(ax, title='Example field $a(y)$ with $y\in\mathbb{R}^2$')
ax = fig.add_subplot(1, 2, 2, projection='3d')
u.plot(ax, title='FEM solution $u_h(a(y))$')
plt.show()


# ### Generate the basis $W_m$ of randomly placed local averages
# 

# local_width is the width of the measurement squares in terms of FEM mesh squares
width_div = 1
local_width = 2**width_div
spacing_div = 5

Wm_reg, Wloc_reg = pat.make_local_avg_grid_basis(width_div, spacing_div, fem_div, return_map=True)
Wm_reg = Wm_reg.orthonormalise()

m = Wm_reg.n
print('m =', m)

# We make the ambient spaces for Wm and Vn
np.random.seed(2)

Wm_rand, Wloc_rand = pat.make_pw_local_avg_random_basis(m=m, div=fem_div, width=local_width, return_map=True)
Wm_rand = Wm_rand.orthonormalise()

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 2, 1)
sns.heatmap(Wloc_rand.values, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
ax.set_title('Random measurement locations')
ax = fig.add_subplot(1, 2, 2)
sns.heatmap(Wloc_reg.values, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
ax.set_title('Regular measurement locations')
plt.plot()


# ### Generate the dictionary of snapshots
# 

dict_N = 50
dict_grid = np.linspace(0.0, 1.0, dict_N, endpoint=False)
y1s, y2s = np.meshgrid(dict_grid, dict_grid)

y1s = y1s.flatten()
y2s = y2s.flatten()

dict_ys = np.stack([y1s, y2s]).T

dictionary, dictionary_fields = make_2d_param_soln(dict_ys, fem_div, a_bar=a_bar, c=c)


greedy_algs = [pat.GreedyApprox(dictionary, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedOMP(dictionary, u, Wm_reg, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedPP(dictionary, u, Wm_reg, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedOMP(dictionary, u, Wm_rand, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedPP(dictionary, u, Wm_rand, Vn=pat.PWBasis(), verbose=True)]

greedy_algs_labels = ['Plain greedy', 
                      'Reg grid meas based OMP', 'Reg grid meas based PP', 
                      'Rand meas based OMP', 'Rand meas based PP',]

for g, l in zip(greedy_algs, greedy_algs_labels):
    print('Constructing ' + l)
    g.construct_to_n(m)


for i, greedy in enumerate(greedy_algs):
    ps = dict_ys[np.array(greedy.dict_sel, dtype=np.int32), :]
    print(greedy_algs_labels[i])
    print(ps)


sns.set_palette('hls', len(greedy_algs))
sns.set_style('whitegrid')

fig = plt.figure(figsize=(7,7))

for i, greedy in enumerate(greedy_algs):
    labels = ['{0} point {1}'.format(greedy_algs_labels[i], j) for j in range(greedy.n)] 
    
    ps = dict_ys[np.array(greedy.dict_sel, dtype=np.int32), :]
    
    plt.scatter(ps[:, 0], ps[:, 1], marker='o', label=greedy_algs_labels[i])

    for label, x, y in zip(labels, ps[:, 0], ps[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom')

plt.legend()
plt.show()


# ### What is the width of the "cone" of $V_n$?
# 
# We can calculate this using the eigenvalues of the grammian. Also, what what are the associated eigenvectors? Whoa. What does this all mean?
# 
# Evidently what I mean is for $v_1$ and $v_2$ that are _convex_ combinations (ermmmm...?)

fig = plt.figure(figsize=(15,12))
for i, v in enumerate(g.Vn.vecs):
    
    ax = fig.add_subplot(3, 3, i+1, projection='3d')
    v.plot(ax, title=r'$\phi_{{{0}}}$'.format(i+1))

plt.show()


lam, V = np.linalg.eigh(g.Vn.G)
print(lam[::-1])

fig = plt.figure(figsize=(15,12))
for i, v in enumerate(V.T[::-1]):
    
    vec = g.Vn.reconstruct(v)
    print(v)
    ax = fig.add_subplot(3, 3, i+1, projection='3d')
    vec.plot(ax, title='Eigenvector {0}'.format(i+1))

plt.show()


# ### Do all convex combinations of $V_n$ have a corresponding $y$?
# 




# $\def \dot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \adot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \cD {\mathcal{D}}$
# $\def \bc {\mathbf{c}}$
# $\def \bv {\mathbf{v}}$
# $\def \bG {\mathbf{G}}$
# 
# # Greedy algorithms with a 2d manifold
# 
# Here we consider the solutions of the PDE $u_h(a(y))$ where $y\in\mathbb{R}^2$, $a(y) = y_1 \chi_{D_1}(x) + y_2 \chi_{D_2}(x)$, and $D_1 = [0,1/2) \times [0,1]$ and $D_2 = [1/2, 1] \times [0,1]$, and $\chi_{D_1}$, $\chi_{D_2}$ are the indicator functions on $D_1$, $D_2$.
# 
# We're given our measurement space $W_m = \mathrm{span}\{w_1,\ldots,w_m\}$. We have a series of measurements $\langle w_i, u\rangle_V$, and we write $w := P_{W_m} u$, the projection of $u$ in $W_m$. We try random, even, and even sinusoidal measurements.
# 
# __Remarks:__
#  - __Note that $\dot{\cdot}{\cdot} = \dot{\cdot}{\cdot}_{V_h}$ here.__
#  - We can __assume that the $\omega_i$ are orthonormal__ as it is a fixed basis.
#  - We store $\dot{\omega_i}{v}$ for each $v\in \cD$, so can we di all projections and inner-products in $\mathbb{R}^m$? And furthermore without the need of a Gram matrix $\dot{\phi_i}{\phi_j}$ to do the projection or orthonormalisation of $\phi_i$?
#  - Finally, remember that _we can not know anything about $u$ other than $w :=P_{W_m} u$_, and we abuse notation slightly and also write $w=\dot{\omega_i}{u}$ for the vector in $\mathbb{R}^m$.
# 
# We have a dictionary $\cD$ of solutions $v(y)$, which we get from generating points in $y\in\mathbb{R}^{2}$ and then $v(y) = \frac{u_h(a(y))}{\| u_h(a(y)) \|_{V_h}}$ (i.e. we normalise all elements). 
# 
# __Algorithms 1-3, the "pure greedy", "measurement based OMP" and "measurement based PP" are used here.__
# 

import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')

def make_2d_param_soln(points, fem_div, a_bar=1.0, c=0.5, f=1.0, verbose=False):
    
    solns = []
    fields = []

    for p in points:
        field = pat.PWConstantSqDyadicL2(a_bar + c * np.repeat(p[:,np.newaxis], 2, axis=1).T)
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        solns.append(fem_solver.u)
        
    return solns, fields


# ### Generate the solution $u$ that we want to approximate
# 

fem_div = 7

a_bar = 0.1
c = 2.0

np.random.seed(2)

y = np.array([[0.8, 0.1]])
print(y[0])
u, a = make_2d_param_soln(y, fem_div, a_bar=a_bar, c=c)
u = u[0]
a = a[0]

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
a.plot(ax, title='Example field $a(y)$ with $y\in\mathbb{R}^2$')
ax = fig.add_subplot(1, 2, 2, projection='3d')
u.plot(ax, title='FEM solution $u_h(a(y))$')
plt.show()


# ### Generate the basis $W_m$ of randomly placed local averages
# 

# local_width is the width of the measurement squares in terms of FEM mesh squares
width_div = 1
local_width = 2**width_div
spacing_div = 4

Wm_reg, Wloc_reg = pat.make_local_avg_grid_basis(width_div, spacing_div, fem_div, return_map=True)
Wm_reg = Wm_reg.orthonormalise()

m = Wm_reg.n
print('m =', m)

# We make the ambient spaces for Wm and Vn
np.random.seed(2)

Wm_rand, Wloc_rand = pat.make_pw_local_avg_random_basis(m=m, div=fem_div, width=local_width, return_map=True)
Wm_rand = Wm_rand.orthonormalise()

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 2, 1)
sns.heatmap(Wloc_rand.values, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
ax.set_title('Random measurement locations')
ax = fig.add_subplot(1, 2, 2)
sns.heatmap(Wloc_reg.values, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
ax.set_title('Regular measurement locations')
plt.plot()


# ### Generate the dictionary of snapshots
# 
# Note that the $y$ that form the dictionary are on the regular ```dict_n``` $\times$ ```dict_N``` grid in $[0,1]^2$.
# 

dict_N = 20
dict_grid = np.linspace(0.0, 1.0, dict_N, endpoint=False)
y1s, y2s = np.meshgrid(dict_grid, dict_grid)

y1s = y1s.flatten()
y2s = y2s.flatten()

dict_ys = np.stack([y1s, y2s]).T

dictionary, dictionary_fields = make_2d_param_soln(dict_ys, fem_div, a_bar=a_bar, c=c)


greedy_algs = [pat.GreedyApprox(dictionary, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedOMP(dictionary, u, Wm_reg, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedPP(dictionary, u, Wm_reg, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedOMP(dictionary, u, Wm_rand, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedPP(dictionary, u, Wm_rand, Vn=pat.PWBasis(), verbose=True)]

greedy_algs_labels = ['Plain greedy', 
                      'Reg grid meas based OMP', 'Reg grid meas based PP', 
                      'Rand meas based OMP', 'Rand meas based PP',]

for g, l in zip(greedy_algs, greedy_algs_labels):
    print('Constructing ' + l)
    g.construct_to_n(m)


# ### So we see that all greedy algorithms select the same 3 points, that span all the dictionary points, and certainly also spans all of $\mathcal{M}$
# 

for i, greedy in enumerate(greedy_algs):
    ps = dict_ys[np.array(greedy.dict_sel, dtype=np.int32), :]
    print(greedy_algs_labels[i])
    print(ps)


sns.set_palette('hls', len(greedy_algs))
sns.set_style('whitegrid')

fig = plt.figure(figsize=(7,7))

for i, greedy in enumerate(greedy_algs):
    labels = ['{0} point {1}'.format(greedy_algs_labels[i], j) for j in range(greedy.n)] 
    
    ps = dict_ys[np.array(greedy.dict_sel, dtype=np.int32), :]
    
    plt.scatter(ps[:, 0], ps[:, 1], marker='o')

    for label, x, y in zip(labels, ps[:, 0], ps[:, 1]):
        plt.annotate(
            label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom')

plt.show()


# ### Wait so we really have 3 linearly independent solutions and that's it?
# 

Vn = g.Vn

fig = plt.figure(figsize=(15, 4))
for i,v in enumerate(Vn.vecs):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    v.plot(ax, title=r'$v_{{{0}}}$'.format(i+1))
plt.show()

v3_p = Vn.subspace(slice(0,2)).project(Vn.vecs[-1])
v2_perp = Vn.vecs[1] - Vn.subspace(slice(0,1)).project(Vn.vecs[1])
v3_perp = Vn.vecs[2] - Vn.subspace(slice(0,1)).project(Vn.vecs[2])

v2_orth = Vn.vecs[1] - Vn.subspace_mask(np.array([1,0,1],dtype=np.bool)).project(Vn.vecs[1])
v3_orth = Vn.vecs[2] - Vn.subspace(slice(0,2)).project(Vn.vecs[2])


fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(1, 3, 1, projection='3d')
v3_p.plot(ax, title=r'Projections of $v_3$ on $\mathrm{span}\{v_1,v_2\}$')
ax = fig.add_subplot(1, 3, 2, projection='3d')
v2_perp.plot(ax, title=r'$v_2 - P_{v_1} v_2$')
ax = fig.add_subplot(1, 3, 3, projection='3d')
v3_perp.plot(ax, title=r'$v_3 - P_{v_1} v_3$')
plt.show()

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1, projection='3d')
(v2_perp + v3_perp).plot(ax, title=r'$v_2 - P_{v_1} v_2 + v_3 - P_{v_1} v_3$')
plt.show()

fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(1, 3, 1, projection='3d')
v2_orth.plot(ax, title=r'$v_2 - P_{v_1,v_3} v_2$')
ax = fig.add_subplot(1, 3, 2, projection='3d')
v3_orth.plot(ax, title=r'$v_3 - P_{v_1,v_2} v_3$')
plt.show()

print('max(v2_perp + v3_perp): ', (v2_perp - v3_perp).values.max())
print('dot:', v2_perp.dot(v3_perp))



# ### Can we find a closed form for the coefficients of this 2d solution, when expressed as a sum of the sub-domain solutions?
# 

np.random.seed(2)

y = np.array([[0.91241841, 0.123141241]])

y1 = np.array([[y[0,0], 1e10]])
y2 = np.array([[1e10, y[0,1]]])
y1 = np.array([[1, 1e10]])
y2 = np.array([[1e10, 1]])

yw = np.array([[y.mean(),y.mean()]])

u, a = make_2d_param_soln(y, fem_div, a_bar=a_bar, c=c)
uw, aw = make_2d_param_soln(yw, fem_div, a_bar=a_bar, c=c)
u0, a0 = make_2d_param_soln(np.array([[1,1]]), fem_div, a_bar=0, c=1)
u1, a1 = make_2d_param_soln(y1, fem_div, a_bar=0, c=1)
u2, a2 = make_2d_param_soln(y2, fem_div, a_bar=0, c=1)

u = u[0]
u0 = u0[0]
uw = uw[0]
u1=u1[0]
u2=u2[0]

M = np.array([[uw.values[12,55], uw.values[61, 23]], [u1.values[12,55], u1.values[61, 23]]])
w = np.array([u.values[12,55], u.values[61, 23]])

wp2 = np.linalg.solve(M.T, w)
print(wp2)
u_proc_2 = u - wp2[0]*uw - wp2[1]*u1

M = np.array([[uw.values[12,68], uw.values[61, 100]], [u2.values[12,68], u2.values[61, 100]]])
w = np.array([u.values[12,68], u.values[61, 100]])
wp1 = np.linalg.solve(M.T, w)
print(wp1)
u_proc_1 = u - wp1[0]*uw - wp1[1]*u2

ym = a_bar + y.mean() * c
yd = c *0.5 * (y[0,0] - y[0,1])
print('y=', y)
print('ym=', ym)
print('yd=', yd)
print('yd/ym=', yd/ym)
print((ym/yd) * wp2[1] * (a_bar + c*y[0,0]))
print((ym/yd) * wp1[1] * (a_bar + c*y[0,1]))

zd = yd/ym
print('zd / ym(1+zd) =', zd / (ym*(1+zd)))
print('zd / ym(1-zd) =', zd / (ym*(1-zd)))

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(2, 2, 1, projection='3d')
u.plot(ax, title='$u$')
ax = fig.add_subplot(2, 2, 2, projection='3d')
u_proc_2.plot(ax, title='')
ax = fig.add_subplot(2, 2, 3, projection='3d')
u_proc_1.plot(ax, title='')
ax = fig.add_subplot(2, 2, 4, projection='3d')
#(u - uw - (yd/ym) * (u2 - u1)).plot(ax, title='')
(u - uw - (yd/ym) * (u2/(a_bar + c*y[0,1]) - u1/(a_bar + c*y[0,0]))).plot(ax, title='')
plt.show()


# ### Lets try applying those operators $A_0$, $A_1$ and $A_2$ and see what properties we observe
# 

print('y =', y)

u, a = make_2d_param_soln(y, fem_div, a_bar=a_bar, c=c)
u0, a0 = make_2d_param_soln(np.array([[1,1]]), fem_div, a_bar=0, c=1)
u1, a1 = make_2d_param_soln(y1, fem_div, a_bar=0, c=1)
u2, a2 = make_2d_param_soln(y2, fem_div, a_bar=0, c=1)
u = u[0]
a = a[0]
u0=u0[0]
a0=a0[0]
u1=u1[0]
u2=u2[0]

fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field=a0, f=1)

import scipy.linalg
import scipy.sparse

n_side = 2**fem_div - 1
h = 1.0 / (n_side + 1)

a = np.copy(a0.interpolate(fem_div).values)
a[:,2**(fem_div-1):] = 0
diag = 2.0 * (a[:-1, :-1] + a[:-1,1:] + a[1:,:-1] + a[1:, 1:]).flatten()
lr_diag = -(a[1:, 1:] + a[:-1, 1:]).flatten()
lr_diag[n_side-1::n_side] = 0 # These corresponds to edges on left or right extreme
lr_diag = lr_diag[:-1]
ud_diag = -(a[1:-1, 1:] + a[1:-1, :-1]).flatten()
A0_1 = scipy.sparse.diags([diag, lr_diag, lr_diag, ud_diag, ud_diag], [0, -1, 1, -n_side, n_side]).tocsr()

a = np.copy(a0.interpolate(fem_div).values)
a[:,:2**(fem_div-1)] = 0
diag = 2.0 * (a[:-1, :-1] + a[:-1,1:] + a[1:,:-1] + a[1:, 1:]).flatten()
lr_diag = -(a[1:, 1:] + a[:-1, 1:]).flatten()
lr_diag[n_side-1::n_side] = 0 # These corresponds to edges on left or right extreme
lr_diag = lr_diag[:-1]
ud_diag = -(a[1:-1, 1:] + a[1:-1, :-1]).flatten()
A0_2 = scipy.sparse.diags([diag, lr_diag, lr_diag, ud_diag, ud_diag], [0, -1, 1, -n_side, n_side]).tocsr()

u0_1 = pat.PWLinearSqDyadicH1(u0.values)
u0_1.values[:, 2**(fem_div-1):] = 0

u0_2 = pat.PWLinearSqDyadicH1(u0.values)
u0_2.values[:, :2**(fem_div-1)] = 0

print(u0.values[1:-1,1:-1].flatten().shape)
print(A0_1.shape)

f = fem_solver.A @ u0.values[1:-1,1:-1].flatten()
f0_1 = A0_1 @ u0.values[1:-1,1:-1].flatten()
f0_2 = A0_2 @ u0.values[1:-1,1:-1].flatten()

n_side = 2**fem_div - 1
f = f.reshape([n_side, n_side])
f0_1 = f0_1.reshape([n_side, n_side])
f0_2 = f0_2.reshape([n_side, n_side])

ff = pat.PWLinearSqDyadicH1(np.pad(f, ((1,1),(1,1)), 'constant'))
f0_1f = pat.PWLinearSqDyadicH1(np.pad(f0_1, ((1,1),(1,1)), 'constant'))
f0_2f = pat.PWLinearSqDyadicH1(np.pad(f0_2, ((1,1),(1,1)), 'constant'))

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(2, 3, 1, projection='3d')
u0.plot(ax, title=r'$u_0$')
ax = fig.add_subplot(2, 3, 2, projection='3d')
u0_1.plot(ax, title=r'$u_0 \chi_{D_1} = u_0 |_{D_1}$')
ax = fig.add_subplot(2, 3, 3, projection='3d')
u0_2.plot(ax, title=r'$u_0 \chi_{D_2} = u_0 |_{D_2}$')
ax = fig.add_subplot(2, 3, 4, projection='3d')
ff.plot(ax, title=r'$A_0 u_0 = f$')
ax = fig.add_subplot(2, 3, 5, projection='3d')
f0_1f.plot(ax, title=r'$A_0 (u_0 \chi_{D_1})$')
ax = fig.add_subplot(2, 3, 6, projection='3d')
f0_2f.plot(ax, title=r'$A_0 (u_0 \chi_{D_2})$')
plt.show()


# So we clearly see that $A_0 (\chi_{D_i} u_0) \neq \chi_{D_i} f$, which makes sense. On the other hand I suspect we still have that in the weak sense $A_0 (\chi_{D_i} u_0) = A_i u_i$
# 

print((f0_1f + f0_2f).values.sum())
print(ff.values.sum())


n_side = 2**fem_div - 1

# Make u's
u, a = make_2d_param_soln(y, fem_div, a_bar=a_bar, c=c)
u0, a0 = make_2d_param_soln(np.array([[1,1]]), fem_div, a_bar=0, c=1)
u1, a1 = make_2d_param_soln(y1, fem_div, a_bar=0, c=1)
u2, a2 = make_2d_param_soln(y2, fem_div, a_bar=0, c=1)
u = u[0]
u0=u0[0]
u1=u1[0]
u2=u2[0]

# Lets make A_0, A_1 and A_2
fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = pat.PWConstantSqDyadicL2(np.ones((2,2))), f = 1)
A0 = fem_solver.A

vals = np.array([[1.0, 0.0]])
a1 = pat.PWConstantSqDyadicL2(np.repeat(vals, 2, axis=1).reshape((2,2)).T)
fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = a1, f = 1)
A1 = fem_solver.A

vals = np.array([[0.0, 1.0]])
a2 = pat.PWConstantSqDyadicL2(np.repeat(vals, 2, axis=1).reshape((2,2)).T)
fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = a2, f = 1)
A2 = fem_solver.A

def apply_operator(A, u):
    return pat.PWLinearSqDyadicH1(np.pad((A @ u.values[1:-1,1:-1].flatten()).reshape([n_side, n_side]), ((1,1),(1,1)), 'constant'))

f0 = apply_operator(A0, u0)
f1 = apply_operator(A1, u0)
f2 = apply_operator(A2, u0)

fig = plt.figure(figsize=(15, 10))
#ax = fig.add_subplot(2, 3, 1, projection='3d')
#f0.plot(ax, title=r'$A_0 u_0$')
ax = fig.add_subplot(2, 3, 1, projection='3d')
f1.plot(ax, title=r'$A_1 u_0$')
ax = fig.add_subplot(2, 3, 2, projection='3d')
f2.plot(ax, title=r'$A_2 u_0$')
ax = fig.add_subplot(2, 3, 3, projection='3d')
(f1-f2).plot(ax, title=r'$A_1u_0 - A_2u_0$')
ax = fig.add_subplot(2, 3, 4, projection='3d')
apply_operator(A1, u1).plot(ax, title=r'$A_1 u_1$')
ax = fig.add_subplot(2, 3, 5, projection='3d')
apply_operator(A2, u2).plot(ax, title='')
ax = fig.add_subplot(2, 3, 6, projection='3d')
(apply_operator(A1, u1) - apply_operator(A2, u2)).plot(ax, title=r'$A_1u_1 - A_2u_2$')
plt.show()

print(apply_operator(A2, u0).values.sum())
print(apply_operator(A2, u2).values.sum())
print(apply_operator(A1, u0).values.sum())
print(apply_operator(A1, u1).values.sum())


# ### Lets move the boundary from the center as symmetry may be reducing the dimensionality
# 

def make_sliver_soln(points, fem_div, a_bar=1.0, c=0.5, f=1.0, verbose=False):
    
    solns = []
    fields = []

    for p in points:
        vals = p[0] * np.ones((4,4))
        vals[:,:3] = p[1]
        field = pat.PWConstantSqDyadicL2(a_bar + c * vals)
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        solns.append(fem_solver.u)
        
    return solns, fields


fem_div = 7

a_bar = 0.1
c = 2.0

y = np.array([[0.91241841, 0.123141241]])

u, a = make_sliver_soln(y, fem_div, a_bar=a_bar, c=c)
u = u[0]
a = a[0]

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
a.plot(ax, title='Example field $a(y)$ with $y\in\mathbb{R}^2$')
ax = fig.add_subplot(1, 2, 2, projection='3d')
u.plot(ax, title='FEM solution $u_h(a(y))$')
plt.show()


y1 = np.array([[1, 1e10]])
y2 = np.array([[1e10, 1]])

yw = np.array([[y.mean(),y.mean()]])

u, a = make_sliver_soln(y, fem_div, a_bar=a_bar, c=c)
uw, aw = make_sliver_soln(yw, fem_div, a_bar=a_bar, c=c)
u0, a0 = make_sliver_soln(np.array([[1,1]]), fem_div, a_bar=0, c=1)
u1, a1 = make_sliver_soln(y1, fem_div, a_bar=0, c=1)
u2, a2 = make_sliver_soln(y2, fem_div, a_bar=0, c=1)

u = u[0]
u0=u0[0]
uw=uw[0]
u1=u1[0]
u2=u2[0]

ym = a_bar + y.mean() * c
yd = c *0.5 * (y[0,0] - y[0,1])
print('y=', y)
print('ym=', ym)
print('yd=', yd)
print('yd/ym=', yd/ym)
print((ym/yd) * wp2[1] * (a_bar + c*y[0,0]))
print((ym/yd) * wp1[1] * (a_bar + c*y[0,1]))

zd = yd/ym
print('zd / ym(1+zd) =', zd / (ym*(1+zd)))
print('zd / ym(1-zd) =', zd / (ym*(1-zd)))

us = [u0, u1, u2]
M = np.vstack([v.values.flatten() for v in us])
w = u.values.flatten()

C = M @ M.T
g = M @ w

cf = np.linalg.solve(C, g)
print("Calculated coefficients:", cf)

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(2, 2, 1, projection='3d')
u.plot(ax, title='$u$')
ax = fig.add_subplot(2, 2, 2, projection='3d')
u1.plot(ax, title='')
ax = fig.add_subplot(2, 2, 3, projection='3d')
u2.plot(ax, title='')
ax = fig.add_subplot(2, 2, 4, projection='3d')
#(u - uw - (yd/ym) * (u2/(a_bar + c*y[0,1]) - u1/(a_bar + c*y[0,0]))).plot(ax, title='')
(u - cf[0] * u0 - cf[1] * u1 - cf[2] * u2).plot(ax, title='')
plt.show()


# We rewrite the field as
# $$
# a(y) = \bar{a} + c (y_1\chi_{D_1} + y_2\chi_{D_2}) 
# = y_m \left(1+ y_d\left(\chi_{D_1} - \chi_{D_2} \right)\right)
# $$
# where $y_m = \bar{a} + c\frac{y_1+y_2}{2}$ and $y_d = \frac{\frac{y_1-y_2}{2}}{y_m}$. We then define the operators $A_0$, $A_1$ and $A_2$ as in my notes.
# 
# The equation previously written as $(\bar{a}A_0 + cy_1A_1 + cy_2A_2)u$ now becomes
# $$
# y_m \left(A_0 u + y_d\left(A_1 u - A_2 u \right)\right)
# $$
# 
# What does this get us? We try on full $u$ and the domain decomposed $u$'s, i.e. solutions to $A_i$ on $H_0^1(D_i)$
# 
# ### First for the sliver... and we clearly see that $A_1 u_0 \neq A_1 u_1$, proving the point that this is exactly where symmetry is needed in my argument
# 

n_side = 2**fem_div - 1

# Lets make A_0, A_1 and A_2
fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = pat.PWConstantSqDyadicL2(np.ones((2,2))), f = 1)
A0 = fem_solver.A

vals = np.ones((4,4))
vals[:,:3] = 0.0
a1 = pat.PWConstantSqDyadicL2(vals)
fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = a1, f = 1)
A1 = fem_solver.A

vals = np.zeros((4,4))
vals[:,:3] = 1.0
a2 = pat.PWConstantSqDyadicL2(vals)
fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = a2, f = 1)
A2 = fem_solver.A

def apply_operator(A, u):
    return pat.PWLinearSqDyadicH1(np.pad((A @ u.values[1:-1,1:-1].flatten()).reshape([n_side, n_side]), ((1,1),(1,1)), 'constant'))

f0 = apply_operator(A0, u0)
f1 = apply_operator(A1, u0)
f2 = apply_operator(A2, u0)

fig = plt.figure(figsize=(15, 10))
#ax = fig.add_subplot(2, 3, 1, projection='3d')
#f0.plot(ax, title=r'$A_0 u_0$')
ax = fig.add_subplot(2, 3, 1, projection='3d')
f1.plot(ax, title=r'$A_1 u_0$')
ax = fig.add_subplot(2, 3, 2, projection='3d')
f2.plot(ax, title=r'$A_2 u_0$')
ax = fig.add_subplot(2, 3, 3, projection='3d')
(f1-f2).plot(ax, title=r'$A_1u_0 - A_2u_0$')
ax = fig.add_subplot(2, 3, 4, projection='3d')
apply_operator(A1, u1).plot(ax, title=r'$A_1 u_1$')
ax = fig.add_subplot(2, 3, 5, projection='3d')
apply_operator(A2, u2).plot(ax, title='')
ax = fig.add_subplot(2, 3, 6, projection='3d')
((apply_operator(A1, u1) - apply_operator(A2, u2))).plot(ax, title=r'$A_1u_1 - A_2u_2$')
plt.show()

print(apply_operator(A2, u0).values.sum())
print(apply_operator(A2, u2).values.sum())
print(apply_operator(A1, u0).values.sum())
print(apply_operator(A1, u1).values.sum())





# # Piecewise linear functions on triangulations
# 
# We test the two OMP algorithm on functions in two dimensions, $[0,1]^2$, that is we take $V = H_0^1([0,1]^2)$. We are approximating solutions $u(a(y))$ of the diffusion
# 
# We consider a dyadic level-$N$ uniform grid with spacing $\Delta x_{(N)} = 2^{-N}$. The solution is computed on the regular triangulation on this grid, evidently with $h=2^{-N}$. The field $a$ is a pw-constant function on a typically coarser grid, with spacing say $2^{-N_a}$, where $N_a \le N$. 
# 
# The dictionary $\mathcal{D}$ for the OMP algorithms consists of the representers of local integration, where the kernel for the local integration are hat-functions. The hat functions we consider are on the squares of size $2^{-N_\mathrm{hat}}$, again with $N_{\mathrm{hat}} \le N$.
# 
# That is, if $\mathrm{Hat}_{i,j}$ is the pw linear hat function on the square $[i 2^{-N_{\mathrm{hat}}}, (i+1) 2^{-N_{\mathrm{hat}}}) \times [j 2^{-N_{\mathrm{hat}}}, (j+1) 2^{-N_{\mathrm{hat}}})$, then $\mathcal{D}^{\mathrm{hat}}$ is the collection of functions $\omega_{i,j}$ such that
# $$
# \langle f, \omega_{i,j}\rangle_{H_0^1} = \int_{[0,1]^2} f \, \mathrm{Hat}_{i,j} \, \mathrm{d}x
# $$
# 
# The measurement space $W_m$ is constructed such as to minimised $\beta(V_n, W_m)$, where $V_n$ is the provided approximation space. In this case $V_n$ is the space of random the solutions $u(a(y))$
# 

import numpy as np
import scipy as sp
import math
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')


# ### Now we are going to set up a full dictionary of local integration points
# 

fem_div = 7
field_div = 2

n = 50

np.random.seed(3)

Vn_sin = pat.make_pw_sin_basis(div=fem_div)
Vn_red, fields = pat.make_pw_reduced_basis(n, field_div=field_div, fem_div=fem_div)
Vn_red_o = Vn_red.orthonormalise()


# Lets plot our measurment locations
fig = plt.figure(figsize=(15, 6))

disp = 6
for i in range(disp):
    ax1 = fig.add_subplot(2, disp, i+1, projection='3d')
    Vn_sin.vecs[i].plot(ax1, title=r'sin $\omega_{{{0}}}$'.format(i+1))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set(xlabel='', ylabel='')
    ax2 = fig.add_subplot(2, disp, i+1+disp, projection='3d')
    Vn_red_o.vecs[i].plot(ax2, title=r'reduced $\omega_{{{0}}}$'.format(i+1))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.set(xlabel='', ylabel='')
plt.show()

fig = plt.figure(figsize=(15,10))
disp = 12
for i in range(disp):
    ax1 = fig.add_subplot(3, disp/3, i+1, projection='3d')
    Vn_red_o.vecs[i].plot(ax1, title=r'$\phi_{{{0}}}$'.format(i+1))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set(xlabel='', ylabel='')
plt.savefig('Vn_RedBasisOrtho.pdf')
plt.show()


# ### Now examine the convergence of projections against a "snapshot" reduced basis and a sinusoidal basis
# 
# The idea being that we are examining the balance of both $\mu(V_n, W_m)$ and $\varepsilon$ with the two choices of approximation space
# 

num_sol = 10
sols, sol_fields = pat.make_pw_reduced_basis(num_sol, field_div=field_div, fem_div=fem_div)
soln_col = sols.vecs

dist_sin = np.zeros((num_sol, n))
dist_red = np.zeros((num_sol, n))

for i, v in enumerate(soln_col):
    for j in range(1,n):
        P_v_sin = Vn_sin.subspace(slice(0,j)).project(v)
        P_v_red = Vn_red.subspace(slice(0,j)).project(v)
        
        dist_sin[i, j] = (v - P_v_sin).norm()
        dist_red[i, j] = (v - P_v_red).norm()


# ### Now we compare the OMP algorithm against both...  For now we do it here with n=20
# 

width = 2

n = 20
m=150

print('Construct dictionary of local averages...')
D = pat.make_pw_hat_rep_dict(fem_div, width=width)

print('Worst-case greedy basis construction...')

wcbc = pat.WorstCaseOMP(D, Vn_sin.subspace(slice(0,20)), Wm=pat.PWBasis(), verbose=True)
Wm_wc_sin = wcbc.construct_to_m(m)
Wm_wc_sin_o = Wm_wc_sin.orthonormalise()

wcbc = pat.WorstCaseOMP(D, Vn_red_o.subspace(slice(0,20)), Wm=pat.PWBasis(), verbose=True)
Wm_wc_red = wcbc.construct_to_m(m)
Wm_wc_red_o = Wm_wc_red.orthonormalise()

bs_wc_sin = np.zeros(m)
bs_wc_red = np.zeros(m)

# For efficiency it makes sense to compute the basis pair and the associated
# cross-gramian only once, then sub sample it as we grow m...
BP_wc_sin_l = pat.BasisPair(Wm_wc_sin_o, Vn_sin.subspace(slice(0,20)))
BP_wc_red_l = pat.BasisPair(Wm_wc_red_o, Vn_red_o.subspace(slice(0,20)))

for i in range(n, m):
    BP_wc_sin =  BP_wc_sin_l.subspace(Wm_indices=slice(0,i))
    BP_wc_red =  BP_wc_red_l.subspace(Wm_indices=slice(0,i))

    bs_wc_sin[i] = BP_wc_sin.beta()
    bs_wc_red[i] = BP_wc_red.beta()


sns.set_style('whitegrid')
line_style = ['-', '--', ':', '-', '-.']

sns.set_palette("hls", 8)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for sinusoid and reduced bases with $n=20$')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')
    
plt.plot(range(m), bs_wc_sin, label=r'Sinusoid basis')
plt.plot(range(m), bs_wc_red, label=r'Reduced basis')
ax.set(xlim=[1,m], ylim=[0,1], xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$')
ax.legend(loc=4)
plt.savefig('SinVsRedBeta.pdf')
plt.show()  

n = 50
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, title=r'Projection errors for various solutions of $u_h(a(y))$ and average')

cp = sns.color_palette("hls", 8)
#plt.semilogy(range(1,n), dist_sin[0, 1:], ':', color=cp[0], linewidth=1, label=r'Sinusois basis: proj error single vec')
#for i, v in enumerate(soln_col[1:]):
#    plt.semilogy(range(1,n), dist_sin[i, 1:], ':', color=cp[0], linewidth=1)
plt.semilogy(range(1,n), dist_sin[:,1:].mean(axis=0), color=cp[0], label=r'Sinusoid basis: average projection error')
    
#plt.semilogy(range(1,n), dist_red[0, 1:], ':', color=cp[1], linewidth=1, label=r'Reduced basis: proj error single vec')
#for i, v in enumerate(soln_col[1:]):
#    plt.semilogy(range(1,n), dist_red[i, 1:], ':', color=cp[1], linewidth=1)
plt.semilogy(range(1,n), dist_red[:,1:].mean(axis=0), color=cp[1], label=r'Reduced basis: average projection error')
ax.set(xlim=[1,n-1], xlabel=r'$n$', ylabel=r'$||v - P_{V_n} v ||$')
ax.legend(loc=1)
plt.savefig('SinVsRedProjErr.pdf')
plt.show()  





# $\def \dot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \adot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \cD {\mathcal{D}}$
# $\def \cM {\mathcal{M}}$
# $\def \bc {\mathbf{c}}$
# $\def \bv {\mathbf{v}}$
# $\def \bG {\mathbf{G}}$
# 
# # How close does domain decomposition come to describing the manifold of solutions $\cM$?
# 
# In ```03_greedy_2d_manifold``` we consider the solutions of the PDE $u_h(a(y))$ where $y\in\mathbb{R}^2$, $a(y) = y_1 \chi_{D_1}(x) + y_2 \chi_{D_2}(x)$, and $D_1 = [0,1/2) \times [0,1]$ and $D_2 = [1/2, 1] \times [0,1]$, and $\chi_{D_1}$, $\chi_{D_2}$ are the indicator functions on $D_1$, $D_2$.
# 
# We saw that the solution manifold $\cM = \{u(a(y)) : y\in U=[0,1]^2\}$ was contained in a 3 dimensional space spanned by the subdomain solutions on $D$, $D_1$ and $D_2$.
# 
# ### Showing that $\cM$ is spanned by 3 solutions
# 
# The demonstration is a reasonably straightforward solution inspired by the results of the greedy algorithm in the  ```03_greedy_2d_manifold``` worksheet. We have 
# $$
# a(y) = \bar{a} + c(y_1 \chi_{D_1} + y_2 \chi_{D_2})
# $$
# 
# We define the operators $A_0$, $A_1$ and $A_2$ in the inner product
# $$
# \dot{A_0 u}{v}_V = \int_D \nabla u \cdot \nabla v \, \mathrm{d} x
# $$
# and
# $$
# \dot{A_1 u}{v}_V = \int_{D} \chi_{D_1} \nabla u \cdot \nabla v \, \mathrm{d} x = \int_{D_1} \nabla u \cdot \nabla v \, \mathrm{d} x
# $$
# and similar for $A_2$. Recall that $V=H_0^1(D)$. Now let us assume that both $a$ is constant. We can re-write the PDE problem in its weak form as
# 
# $$
# \left ( \bar{a}A_0 + c y_1 A_1 + c y_2 A_2 \right) u = f
# $$
# 
# The essence of the is to see that we can pull out the "scale" of the field out the front, and we are left with one more parameter which is the amount of difference in the field between $D_1$ and $D_2$, that is,
# $$
# a(y_1, y_2) = a(y_m, y_d) = y_m \left( 1 + y_d (\chi_{D_1} - \chi_{D_2}) \right)
# $$
# where we have defined 
# $$
# y_m = \bar{a} + c\frac{y_1 + y_2}{2} \quad \text{and} \quad y_d = c \frac{y_1-y_2}{2} y_m^{-1}
# $$
# 
# Now, we write $u_0\in H_0^1(D)$, $u_1\in H_0^1(D_1)$ and $u_2 \in H_0^1(D_2)$ for the solutions of 
# $$
# A_0 u_0 = f \quad A_1 u_1 = \chi_{D_1} f \quad A_2 u_2 = \chi_{D_2} f
# $$
# 
# __Then the solution is given by__
# 
# $$
# \large
# u = \frac{1}{y_m} \left( u_0 - y_d \left( \frac{u_1}{1+y_d} - \frac{u_2}{1-y_d}\right)\right)
# $$
# 
# This can be shown by re-writing the PDE in the weak form as
# 
# $$
# y_m \left(A_0 + y_d (A_1 - A_2) \right) u = f
# $$
# 
# Substituting the solution above in to this form we obtain
# 
# $$
# \left(A_0 + y_d (A_1 - A_2) \right) \left( u_0 - y_d \left( \frac{u_1}{1+y_d} - \frac{u_2}{1-y_d}\right)\right)
# = A_0 u_0 - y_d (A_1 - A_2) u_0 - y_d \left( \frac{A_0 u_1}{1+y_d} - \frac{A_0 u_2}{1-y_d}\right) - y_d^2 \left( \frac{A_1 u_1}{1+y_d} + \frac{A_2 u_2}{1-y_d}\right)
# $$
# 
# We have used the self evident fact that $A_1 u_2 = A_2 u_1 = 0$. Now, it is straightforward to show that $A_0 u_1  = A_1 u_1$ and similar for $u_2$. This thus simplifies to 
# 
# $$
# A_0 u_0 + y_d (A_1 - A_2) u_0 - y_d \left(A_1 u_1 - A_2 u_2 \right)
# $$
# 
# Now we consider the symmetry in the $x_1$ coordinate. We know that $(A_1 u_0) (x_1, x_2) = (A_2 u_0) (1 - x_1, x_2)$ because of the $x_1$-axis symmetry of the problem. This implies that we must have 
# $$
# (A_1 - A_2) u = (\chi_{D_1} - \chi_{D_2} )f,
# $$
# hence that $(A_1 - A_2) u_0 = \left(A_1 u_1 - A_2 u_2 \right)$ and hence that we find our solution produces $A_0 u_0 = f$ which satisfies the PDE.
# 
# ### We see that a similar approach _almost_ works for the non-symmetric cases and for higher-dimensional fields
# 
# We see that for $D_1$ and $D_2$ that are non-symmetric, $(A_1 - A_2) u_0 \neq \left(A_1 u_1 - A_2 u_2 \right)$, but the difference is small.
# 
# Can we perhaps quantify the convergence of symmetric domain decompositions in some way? Are symmetric domain decompositions perhaps a good method of decomposing a problem, in general?

import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')

def make_soln(points, fem_div, a_bar=1.0, c=0.5, f=1.0, verbose=False):
    
    solns = []
    fields = []

    for p in points:
        field = pat.PWConstantSqDyadicL2(a_bar + c * p.reshape((2,2)))
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        solns.append(fem_solver.u)
        
    return solns, fields


# ### Generate the solution $u$ that we want to approximate
# 

fem_div = 7

a_bar = 0.1
c = 2.0

np.random.seed(2)

y = np.random.random((1,4))

u, a = make_soln(y, fem_div, a_bar=a_bar, c=c)
u = u[0]
a = a[0]

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
a.plot(ax, title='Example field $a(y)$ with $y\in\mathbb{R}^2$')
ax = fig.add_subplot(1, 2, 2, projection='3d')
u.plot(ax, title='FEM solution $u_h(a(y))$')
plt.show()


# $\def \dot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \adot #1#2{\left\langle #1, #2 \right\rangle}$
# $\def \cD {\mathcal{D}}$
# $\def \cW {\mathcal{W}}$
# $\def \cM {\mathcal{M}}$
# $\def \bc {\mathbf{c}}$
# $\def \bv {\mathbf{v}}$
# $\def \bG {\mathbf{G}}$
# $\def \bc {\mathbf{c}}$
# $\def \bx {\mathbf{x}}$
# $\def \by {\mathbf{y}}$
# $\def \bu {\mathbf{u}}$
# $\def \bv {\mathbf{v}}$
# $\def \bw {\mathbf{w}}$
# $\def \bG {\mathbf{G}}$
# $\def \bC {\mathbf{C}}$
# $\def \bD {\mathbf{D}}$
# $\def \bI {\mathbf{I}}$
# $\def \bP {\mathbf{P}}$
# $\def \bQ {\mathbf{Q}}$
# $\def \bR {\mathbf{R}}$
# $\def \bS {\mathbf{S}}$
# $\def \bT {\mathbf{T}}$
# $\def \bU {\mathbf{U}}$
# $\def \bV {\mathbf{V}}$
# $\def \bW {\mathbf{W}}$
# $\def \bPhi {\mathbf{\Phi}}$
# $\def \bPsi {\mathbf{\Psi}}$
# $\def \bGamma {\mathbf{\Gamma}}$
# $\def \bSigma {\mathbf{\Sigma}}$
# $\def \bTheta {\mathbf{\Theta}}$
# $\def \bOmega {\mathbf{\Omega}}$
# $\def \bbE {\mathbb{E}}$
# $\def \bbP {\mathbb{P}}$
# $\def \bbR {\mathbb{R}}$
# $\def \bbN {\mathbb{N}}$
# 
# # Why do the greedy results fail?
# 
# Note that all the calculations have been offloaded to a heavy weight script: ```scripts/greedy_Vn/07_greedy_Vn_average_cases.py```. Here we simply import the results from this script. For more information on the greedy algorithms, either see the write up or the earlier notebook ```02_greedy_Vn.ipynb```
# 
# ### Here we examine the "Pure Greedy" algorithm that does not adapt to any measurement
# 
# Given the dictionary $\cD$ we chose
# $$
# \phi_1 = {\arg\max}_{v\in\cD} \| v \|
# $$
# and given $V_{n}=\mathrm{span}\{\phi_1, \ldots, \phi_n\}$, our next choice is
# $$
# \phi_{n+1} = {\arg\max}_{v\in\cD} \| v - P_{V_n} v \|
# $$
# 
# ** I really think this is something interesting to try and nail for Banff. Interesting work in progress, and really it's only me that has the ability or the interest to investigate this **
# 
# For example - look at the selected individuals from the greedy, and look at their existence on the fringe of what most dict elements already have, and how there is not "much in common" with 99% of random realisations, hence contravening the accepted logic
# 
# Also consider what might be interesting in terms of Kolmogorov n-Width results vs PCA / best estimator / least squares distance
# 
# ### Thought after MoRePaS poster session:
# 
# Basically although we choose $\varphi_n$ to be the furthest from $V_{n-1}$, i.e. the residual $\| \varphi_n - P_{V_{n-1}} \varphi_n\|$ is maximised, it just simply isn't the case that this choice really represents 
# 
# $$
# \mathbb{E}_\nu ( \| v - P_{V_{n-1}} v \|_V ) = 
# \int_U \| u(a) - P_{V_{n-1}} u(a) \|_V \, \mathrm{d} \mu(a)
# \int_V \| v - P_{V_{n-1}} v \|_V \, \mathrm{d} \nu(v)
# $$
# 
# where $\nu(v) = \mu(u^{-1}(v))$ is the push forward measure, assuming $u$ is continuous and bijective.
# 
# ### Testing the greedy algorithm
# 
# So - perhaps in a sense what we want is to see which SVD components are within the chosen greedy measurements, and how far "down" the singular vectors we have gone. What does this mean? We can give the likelihood of the greedily chosen vectors in some sense from the SVD of that map.
# 
# Also compare the convergence of the residual to the convergence of the singular values. This may yeild something interesting. To do this though, we need to have an orthogonal representation in either basis, multivariate Legendre for the $Y$ space, and some surmised ortho-normal basis on $V$.
# 
# Well, we could do that, or just trust the convergence of the PCA basis in $V$ that we get from lots and lots of snapshots $u(y_i)$.
# 
# A heirarchical hat basis that is then orthonormalised in $V$ could be a good contender, making sure to apply the Gram-Schmidt process down the heirarchy of scales. One thing to note though - 
# 
# ### Some notes from the MoRePaS conference
# 
# People really are banging on about Kolmogorov $N$-widths as the real be-all end-all of measuring the appropriateness of linear approximation methods. Is this accurate? Stephen Rave seems to think that it's really important, before going on to non-linear approx because of this in a particular case. THink about this.
# 
# See Daniel Kressner's talk on low-rank tensors in reduced basis methods. Really interesting. Also 
# 
# Also Hackbusch/Kuhn 2009
# 

import numpy as np
import scipy as sp
import pandas as pd
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')

results_file = '../../scripts/greedy_Vn/results/07_greedy_Vn_stats.npy'
stats = np.load(results_file)


# ### Run this little snippet to see a plot of the basis vectors from the plain greedy basis
# 
# ...to get a sense of how crazy they are.
# 

g = pat.PWBasis(file_name = "../../scripts/greedy_Vn/results/PlainGreedy_Basis.npz")
r = pat.PWBasis(file_name = "../../scripts/greedy_Vn/results/Reduced_Basis.npz")
p = pat.PWBasis(file_name = "../../scripts/greedy_Vn/results/PCA_Basis.npz")

fig = plt.figure(figsize=(15, 15))
for i, v in enumerate(g.vecs[:16]):
    ax = fig.add_subplot(4, 4, i+1, projection='3d')
    v.plot(ax)
fig.savefig('GreedyPlots.pdf')
plt.show()


fig = plt.figure(figsize=(15, 15))
for i, v in enumerate(r.vecs[:16]):
    ax = fig.add_subplot(4, 4, i+1, projection='3d')
    v.plot(ax)
fig.savefig('ReducedBasisPlots.pdf')
plt.show()


generic_Vns_labels = ['Sinusoid basis', 'Reduced basis', 'Plain greedy', 'PCA']
adapted_Vns_labels = ['Meas based OMP', 'Meas based PP']
Wm_labels = ['Reg grid', 'Random']

sns.set_palette('hls', len(generic_Vns_labels) + len(adapted_Vns_labels))
cp = sns.color_palette()
sns.set_style('whitegrid')

lss = ['-', '--']
lws = [2,1]

axs = []
fig = plt.figure(figsize=(15, 8))
axs.append(fig.add_subplot(1, 2, 1, title='Projection error $\| u_h - P_{V_n} u_h \|_{H_0^1}$, Wm: Reg grid, with 100% CI'))
axs[-1].set(yscale="log", xlabel='$n$')
axs.append(fig.add_subplot(1, 2, 2, title=r'inf-sup condition $\beta(W_m, V_n)$'))
axs[-1].set(yscale="log", xlabel='$n$')

i = 0
Wm_label = Wm_labels[0]
for j, Vn_label in enumerate(generic_Vns_labels):

    Vn_n = np.where(np.isclose((~np.isclose(stats[0, i, j, :, :], 0.0)).sum(axis=0), stats.shape[3]))[-1][-1]

    label = 'Wm: ' + Wm_label + ' Vn: ' + Vn_label

    plt.sca(axs[0])
    sns.tsplot(stats[1, i, j, :, 2:Vn_n], range(2, Vn_n), ls=lss[i], lw=lws[i],color=cp[j], ci=[100])
    plt.plot(range(2, Vn_n), stats[1, i, j, :, 2:Vn_n].mean(axis=0), label=label, ls=lss[i], lw=lws[i], color=cp[j])
    plt.legend(loc=3)

    plt.sca(axs[1])
    plt.plot(range(2, Vn_n), stats[2, i, j, 0, 2:Vn_n], lss[i], label=label, lw=lws[i], color=cp[j])
    plt.legend(loc=3)

for j_i, Vn_label in enumerate(adapted_Vns_labels):
    j = j_i + len(generic_Vns_labels)

    label = 'Wm: ' + Wm_label + ' Vn: ' + Vn_label

    Vn_n = np.where(np.isclose((~np.isclose(stats[0, i, j, :, :], 0.0)).sum(axis=0), stats.shape[3]))[-1][-1]

    plt.sca(axs[0])
    sns.tsplot(stats[1, i, j, :, 2:Vn_n], range(2, Vn_n), ls=lss[i], lw=lws[i], color=cp[j], ci=[100])
    plt.plot(range(2, Vn_n), stats[1, i, j, :, 2:Vn_n].mean(axis=0), label=label, ls=lss[i], lw=lws[i], color=cp[j])        
    plt.legend(loc=3)

    plt.sca(axs[1])
    plt.plot(range(2, Vn_n), stats[2, i, j, 0, 2:Vn_n], lss[i], label=label, lw=lws[i], color=cp[j])
    plt.legend(loc=3)

plt.show()


# Damn, lost some notes. No matter, we have $u\in \cM$ and we want to find
# $$
# \bbP_v \left ( \langle v, C^{-1} v \rangle \ge \langle u, C^{-1} u \rangle \right)
# $$
# where our approx is
# $$
# \langle v, C v \rangle = \frac 1 N \sum_{i=1}^N \left|\langle u_i, v \rangle_V \right|^2
# $$
# and as we have $C = U U^* = \Phi \bSigma^2 \Phi^*$ where $(\Phi^* v)_i = \langle \varphi_i, v \rangle_V$. Now for any $v\in \mathrm{span}(\varphi_1,\ldots)$, the matched normal probability distribution is going to be (now truncating to $N$ singular vectors because otherwise it makes no sense)
# 
# $$
# p(v) = (2\pi)^{-N/2} \prod_{i=1}^N \sigma_i^{-1} \exp \left(-\frac{\left | \langle v, C^{-1} v \rangle_V \right |^2 }{2}\right) = (2\pi)^{-N/2} \prod_{i=1}^N \sigma_i^{-1} \exp \left(- \sum_{i\ge 1} \frac{ \left | \langle \varphi_i, v \rangle_V  \right |^2}{2\sigma_i^2}\right)
# $$
# 
# So - in the end what it means is that $\frac{\langle \varphi_i, v \rangle}{\sigma_i^2}$ should be roughly flat, and that the likelihood, for example, of finding anything less likely than $\phi_i^G$, the greedy basis element, will be
# 
# $$
# \bbP_v \left ( \langle v, C^{-1} v \rangle \ge \langle \varphi_i^G C^{-1} \varphi_i^G \rangle \right)
# = \prod_{i=1}^N \left(1 -  \mathrm{erf} \left(\frac{ \left | \langle \varphi_i, v \rangle_V \right|^2 }{2\sigma_i^2}\right)\right)
# $$
# 

p.dot(g.vecs[0])


# Plot the worst-cases to see the diff between $L_\infty$ and the the average case
# 

def make_soln(points, fem_div, field_div, a_bar=1.0, c=0.5, f=1.0, verbose=False):
    
    solns = []
    fields = []

    for p in points:
        field = pat.PWConstantSqDyadicL2(a_bar + c * p.reshape((2**field_div,2**field_div)))
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        solns.append(fem_solver.u)
        
    return solns, fields

fem_div = 7

a_bar = 1.0
c = 0.9
field_div = 2
side_n = 2**field_div

def make_PCA(N = 1e3):

    np.random.seed(1)
    dict_basis_small, dict_fields = pat.make_pw_reduced_basis(N, field_div, fem_div, a_bar=a_bar, c=c, f=1.0, verbose=False)
    dict_basis_small.make_grammian()

    cent = dict_basis_small.reconstruct(np.ones(N) / N)

    import copy

    cent_vecs = copy.deepcopy(dict_basis_small.vecs)
    for i in range(len(cent_vecs)):
        cent_vecs[i] = cent_vecs[i] - cent

    dict_basis_small_cent = pat.PWBasis(cent_vecs)
    dict_basis_small_cent.make_grammian()
    
    lam, V = sp.linalg.eigh(dict_basis_small_cent.G)
    lams = np.sqrt(lam[np.abs(lam) > 1e-10][::-1])
    n = len(lams)

    PCA_vecs = []
    for i, v in enumerate(np.flip(V.T, axis=0)[:n]):
        vec = dict_basis_small_cent.reconstruct(v)
        PCA_vecs.append(vec / lams[i])

    return pat.PWBasis(PCA_vecs), lams


Vn_PCA_small_dict, small_lams = make_PCA(int(1e2))
Vn_PCA_mid_dict, mid_lams = make_PCA(int(5e2))
Vn_PCA_big_dict, big_lams = make_PCA(int(1e3))


# Now plot the rep of g.vecs[0] in terms of the PCA basis...

plt.semilogy(small_lams)
plt.semilogy(mid_lams)
plt.semilogy(big_lams)
plt.show()

Vn_PCA = Vn_PCA_big_dict
lams = big_lams
n = Vn_PCA.n

plt.semilogy(np.abs(Vn_PCA.dot(g.vecs[0]/g.vecs[0].norm())))
plt.semilogy(np.abs(Vn_PCA.dot(r.vecs[0]/r.vecs[0].norm())))
#plt.semilogy(np.abs(Vn_PCA.dot(g.vecs[1]/g.vecs[1].norm())))
#plt.semilogy(np.abs(Vn_PCA.dot(g.vecs[2]/g.vecs[2].norm())))
plt.semilogy(lams)
plt.show()

g_comp = Vn_PCA.dot(g.vecs[0]/g.vecs[0].norm())
r_comp = Vn_PCA.dot(r.vecs[0]/r.vecs[0].norm())

# Look at this
print(np.linalg.norm(g_comp[:3*n//4]))
print(np.linalg.norm(r_comp[:3*n//4]))
print(np.linalg.norm(g_comp[3*n//4:]))
print(np.linalg.norm(r_comp[3*n//4:]))


# ### Making a heirarchical hat basis for $V$
# 
# __Idea:__ try looking at the condition number of different enumerations of a hat basis in $H_0^1$. Try
#  - linear fill (usual)
#  - alternating fill
#  - diamon fill
# 

h_lev = 2 # The level the hat basis goes to

first_hat = pat.PWLinearSqDyadicH1(div = 1)
first_hat.values[1,1] = 1
first_hat = first_hat / first_hat.norm()

hat_basis = pat.PWBasis(vecs=[first_hat.interpolate(h_lev) / first_hat.norm()])

# Linear fill:
for l in range(1,h_lev):
    for i in range(2**(l+1)-1):
        for j in range(2**(l+1)-1):
            
            h = pat.PWLinearSqDyadicH1(div = l+1)
            h.values[i + 1, j + 1] = 1
            hat_basis.add_vector(h.interpolate(h_lev) / h.norm())
            
hat_basis.make_grammian()

print(np.linalg.cond(hat_basis.G))
print(hat_basis.G.shape)
print(np.linalg.matrix_rank(hat_basis.G))
print((hat_basis.vecs[0].values.shape[0]-2)**2)


for vec in hat_basis.vecs:
    print(vec.values)








# # Piecewise linear functions on triangulations
# 
# We test the two OMP algorithm on functions in two dimensions, $[0,1]^2$, that is we take $V = H_0^1([0,1]^2)$. We are approximating solutions $u(a(y))$ of the diffusion
# 
# We consider a dyadic level-$N$ uniform grid with spacing $\Delta x_{(N)} = 2^{-N}$. The solution is computed on the regular triangulation on this grid, evidently with $h=2^{-N}$. The field $a$ is a pw-constant function on a typically coarser grid, with spacing say $2^{-N_a}$, where $N_a \le N$. 
# 
# The dictionary $\mathcal{D}$ for the OMP algorithms consists of the representers of local integration, where the kernel for the local integration are hat-functions. The hat functions we consider are on the squares of size $2^{-N_\mathrm{hat}}$, again with $N_{\mathrm{hat}} \le N$.
# 
# That is, if $\mathrm{Hat}_{i,j}$ is the pw linear hat function on the square $[i 2^{-N_{\mathrm{hat}}}, (i+1) 2^{-N_{\mathrm{hat}}}) \times [j 2^{-N_{\mathrm{hat}}}, (j+1) 2^{-N_{\mathrm{hat}}})$, then $\mathcal{D}^{\mathrm{hat}}$ is the collection of functions $\omega_{i,j}$ such that
# $$
# \langle f, \omega_{i,j}\rangle_{H_0^1} = \int_{[0,1]^2} f \, \mathrm{Hat}_{i,j} \, \mathrm{d}x
# $$
# 
# The measurement space $W_m$ is constructed such as to minimised $\beta(V_n, W_m)$, where $V_n$ is the provided approximation space. In this case $V_n$ is the space of random the solutions $u(a(y))$
# 

import numpy as np
import scipy as sp
import math
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')


# ### Now we are going to set up a full dictionary of local integration points
# 
# And look at projection errors as we decrease the width and the spacing. The code below is run on a cluster from the ```pyApproxTools/scripts/02_2d_PW_omp.py``` script. The results are processed and displayed in the next cell.
# 
# ```
# np.random.seed(3)
# 
# fem_div = 7
# field_div = 2
# 
# n = 20
# m = 200
# 
# try:
#     width = int(sys.argv[1])
# except IndexError:
#     print("Usage: " + sys.argv[0] + " width")
#     sys.exit(1)
# 
# # Create Vn - an orthonormalised reduced basis
# Vn, fields = pat.make_pw_reduced_basis(n, field_div=field_div, fem_div=fem_div)
# Vn = Vn.orthonormalise()
# 
# Wms_c = []
# Wms_wc = []
# 
# bs_c = np.zeros(m) 
# bs_wc = np.zeros(m) 
# 
# print('Construct dictionary of local averages...')
# D = pat.make_pw_hat_rep_dict(fem_div, width=width)
# 
# print('Greedy basis construction...')
# cbc = pat.CollectiveOMP(D, Vn, Wm=pat.PWBasis(), verbose=True)
# Wm_c = cbc.construct_to_m(m)
# Wm_c_o = Wm_c.orthonormalise()
# Wms_c.append(Wm_c)
# Wm_c_o.save('Wm_c_{0}'.format(width))
# 
# wcbc = pat.WorstCaseOMP(D, Vn, Wm=pat.PWBasis(), verbose=True)
# Wm_wc = wcbc.construct_to_m(m)
# Wm_wc_o = Wm_wc.orthonormalise()
# Wms_wc.append(Wm_wc)
# Wm_wc_o.save('Wm_wc_{0}'.format(width))
# 
# # For efficiency it makes sense to compute the basis pair and the associated
# # cross-gramian only once, then sub sample it as we grow m...
# BP_c_l = pat.BasisPair(Wm_c_o, Vn)
# BP_wc_l = pat.BasisPair(Wm_wc_o, Vn)
# 
# for i in range(n, m):
#     BP_c =  BP_c_l.subspace(Wm_indices=slice(0,i))
#     BP_wc =  BP_wc_l.subspace(Wm_indices=slice(0,i))
# 
#     bs_c[i] = BP_c.beta()
#     bs_wc[i] = BP_wc.beta()
# 
# np.save('bs_c_{0}'.format(width), bs_c)
# np.save('bs_wc_{0}'.format(width), bs_wc)
# ```
# 

fem_div = 7
field_div = 2

n = 20

widths = [2**i for i in range(fem_div-3)][::-1]

# Import data
bs_cs = []
bs_wcs = []


Wms_c = []
Wms_wc = []
for width in widths:
    Wms_c.append(pat.PWBasis(file_name='../../scripts/omp/Wm_c_{0}.npz'.format(width)))
    Wms_wc.append(pat.PWBasis(file_name='../../scripts/omp/Wm_wc_{0}.npz'.format(width)))
    
    bs_cs.append(np.load('../../scripts/omp/bs_c_{0}.npy'.format(width)))
    bs_wcs.append(np.load('../../scripts/omp/bs_wc_{0}.npy'.format(width)))
m = bs_cs[0].shape[0]


# Lets plot our measurment locations

fig = plt.figure(figsize=(15, 8))

for i, width in enumerate(widths):
    meas_c = Wms_c[i].vecs[0]
    for vec in Wms_c[i].vecs[1:]:
        meas_c += vec

    meas_wc = Wms_wc[i].vecs[0]
    for vec in Wms_wc[i].vecs[1:]:
        meas_wc += vec
    print('max ' + str(width) + ' Coll: ' + str(meas_c.values.max()) + ' WC: ' + str(meas_wc.values.max()))
    ax1 = fig.add_subplot(2, len(widths), i+1, projection='3d')
    meas_c.plot(ax1, title='Collective OMP, width {0}'.format(width))
    ax2 = fig.add_subplot(2, len(widths), i+1+len(widths), projection='3d')
    meas_wc.plot(ax2, title='Worst-case OMP, width {0}'.format(width))


sns.set_style('whitegrid')
line_style = ['-', '--', ':', '-', '-.']
pals = [ 'Blues_r', 'Reds_r', 'Greens_r', 'Purples_r']

bl = (51/255, 133/255, 255/255)
re = (255/255, 102/255, 102/255)

axs = []

fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for varying local avg widths')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')
    
sns.set_palette(pals[1])
cp = sns.color_palette()
for i, width in enumerate(widths[1:]):    
    plt.plot(range(m), bs_wcs[i], line_style[i], label=r'Worst-Case OMP $W_m$ for $\varepsilon={{{0}}} \times 2^{{{1}}}$'.format(width, -fem_div), color=re)#cp[i])

sns.set_palette(pals[0])
cp = sns.color_palette()
for i, width in enumerate(widths[1:]):
    plt.plot(range(m), bs_cs[i], line_style[i], label=r'Collective OMP $W_m$ for $\varepsilon={{{0}}} \times 2^{{{1}}}$'.format(width, -fem_div), color=bl)#cp[i])

ax.set(xlabel='m', ylabel=r'$\beta(V_n, W_m)$', xlim=[0,200], ylim=[0,1])#r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=4)
plt.savefig('2dCOMPvsWCOMPLocAvg.pdf')
plt.show()


v = Wm_wc.vecs[0]
for w in Wm_wc.vecs[1:]:
    v += w

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
v.plot(ax1)
plt.show()

fig = plt.figure(figsize=(8, 8))
ax2 = fig.add_subplot(1, 1, 1, projection='3d')
Vn.vecs[2].plot(ax2)
plt.show()





# # Worst-case OMP tests
# 

import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')


# ### Worst-case OMP algorithm
# 
# We construct a measurement basis $W_m$ based on a given approximation basis $V_n$. Our ambient space $V$ is $H_0^1([0,1])$. Here $V_n$ is the sinusoids, normalised in $H_0^1([0,1])$, so $V_n = \mathrm{span}\{\phi_1,\ldots,\phi_n\}$, where $\phi_k = \frac{\sqrt{2}}{\pi k} \sin(k \pi x)$.
# 
# The measurements are assumed to be point evaluations, which have representer in $H_0^1$ of
# $$
# \omega_{x_0}(x) = \frac{1}{\sqrt{x_0 (1-x_0)}}
# \begin{cases}
# x (1 - x_0) & \text{for } x \le x_0 \(1 - x) x_0 & \text{for } x > x_0
# \end{cases}
# $$
# 
# This implementation of the algorithm finds the vector $v\in V_n$ that is furthest possible from $W_m$, that is at each step we look for 
# $$
# v_k = argmax_{v\in V_n,\, \|v\|=1} \| v - P_{W_{k-1}} v \|
# $$
# and then find the dictionary element most closely aligned with this vector
# $$
# \omega_k 
# = \mathrm{argmax}_{\omega\in\mathcal{D}} |\left\langle \omega, v_k - P_{W_{k-1}} v_k \right\rangle| 
# $$
# It is because of the selection of $v_k$ that we call this the _worst-case_ approach. We find $v_k$ through the SVD decomposition of the cross-gramian of $W_{k-1}$ and $V_n$. Although this is expensive, it works out quite well as the slowest part of the algorithm is actually the dot product search through $\mathcal{D}$, as the SVD code is written in 
# 
# ### Lets look at $\beta(V_n, W_m)$ for our worst-case OMP basis and a random basis for comparison
# Note that this calculation is done for a small dictionary that only has $N=10^3$ elements, to save time, however as we saw in the collective-OMP, the size of the dictionary doesn't have a huge impact in this example.
# 

N = 1e3
dictionary = pat.make_unif_dictionary(N)

ns = [20,40]
np.random.seed(3)
#n = 20
m = 200
bs_comp = np.zeros((len(ns), m))
bs_wcomp = np.zeros((len(ns), m))
bs_rand = np.zeros((len(ns), m))

Vn = pat.make_sin_basis(ns[-1])
Wms_comp = []
Wms_wcomp = []
Wms_rand = []

for j, n in enumerate(ns):

    #gbc = pat.CollectiveOMP(dictionary, Vn.subspace(slice(0,n)), verbose=True)
    #Wm_comp = gbc.construct_to_m(m)
    #Wms_comp.append(Wm_comp)
    #Wm_comp_o = Wm_comp.orthonormalise()
    
    wcgbc = pat.WorstCaseOMP(dictionary, Vn.subspace(slice(0,n)), verbose=True)
    Wm_wcomp = wcgbc.construct_to_m(m)
    Wms_wcomp.append(Wm_wcomp)
    Wm_wcomp_o = Wm_wcomp.orthonormalise()

    Wm_rand = pat.make_random_delta_basis(m)
    Wms_rand.append(Wm_rand)
    Wm_rand_o = Wm_rand.orthonormalise()

    #BP_comp_l = pat.BasisPair(Wm_comp_o, Vn.subspace(slice(0,n)))
    BP_wcomp_l = pat.BasisPair(Wm_wcomp_o, Vn.subspace(slice(0,n)))    
    BP_rand_l = pat.BasisPair(Wm_rand_o, Vn.subspace(slice(0,n)))
    for i in range(n, m):
        #BP_comp = BP_comp_l.subspace(Wm_indices=slice(0,i))
        #bs_comp[j, i] = BP_comp.beta()

        BP_wcomp =  BP_wcomp_l.subspace(Wm_indices=slice(0,i))
        bs_wcomp[j, i] = BP_wcomp.beta()
        
        BP_rand = BP_rand_l.subspace(Wm_indices=slice(0,i))
        bs_rand[j, i] = BP_rand.beta()


sns.set_palette("deep")
cp = sns.color_palette()

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title='beta(Vn, Wm) against m for various n')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')

for i, n in enumerate(ns):
    plt.plot(range(m), bs_wcomp[i, :], label='worst-case omp Wm for n={0}'.format(n), color=cp[i])#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))    
    plt.plot(range(m), bs_comp[i, :], '--', label='collective omp Wm for n={0}'.format(n), color=cp[i])#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))
    plt.plot(range(m), bs_rand[i, :], ':', label='random Wm for n={0}'.format(n), color=cp[i], lw=1)#r'Random $W_m$, $n={{{0}}}$'.format(n))

ax.set(xlabel='m', ylabel='beta(Vn, Wm)')#r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=4)
plt.show()


bs_unif_int = np.zeros((len(ns), m))
Vn = pat.make_sin_basis(ns[-1])

Wms_unif_int = []

for j, n in enumerate(ns):
    for i in range(n, m):
        Wm_unif_int = pat.Basis([pat.FuncVector(params=[[x]],coeffs=[[1.0]],funcs=['H1UIDelta']) for x in np.linspace(0.0, 1.0, i, endpoint=False)+0.5/i])
        Wm_unif_int_o = Wm_unif_int.orthonormalise()

        BP_ui = pat.BasisPair(Wm_unif_int_o, Vn.subspace(slice(0,n)))
        bs_unif_int[j, i] = BP_ui.beta()


sns.set_palette("deep")
sns.set_style("whitegrid")
cp = sns.color_palette()

bl = (51/255, 133/255, 255/255)
re = (255/255, 102/255, 102/255)

axs = []
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n=20$, $40$')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')

#for i, n in enumerate(ns):
i=0
n=ns[i]
plt.plot(range(n+1,m), bs_unif_int[i, n+1:], '--', label=r'Uniformly spaced / optimal $W_m$ for $n={{{0}}}$'.format(n), color=cp[1])
plt.plot(range(m), bs_wcomp[i, :], '--', label=r'Worst Case OMP $W_m$ for $n={{{0}}}$'.format(n), color=re)#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))    
plt.plot(range(m), bs_comp[i, :], '--', label=r'Collective OMP $W_m$ for $n={{{0}}}$'.format(n), color=bl)#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))
plt.plot(range(m), bs_rand[i, :], '--', label=r'Random $W_m$ for $n={{{0}}}$'.format(n), color=cp[3], lw=1)#r'Random $W_m$, $n={{{0}}}$'.format(n))
i=1
n=ns[i]
plt.plot(range(n+1,m), bs_unif_int[i, n+1:], label=r'Uniformly spaced $W_m$ / optimal for $n={{{0}}}$'.format(n), color=cp[1])
plt.plot(range(m), bs_wcomp[i, :], label=r'Worst Case OMP $W_m$ for $n={{{0}}}$'.format(n), color=re)#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))    
plt.plot(range(m), bs_comp[i, :], label=r'Collective OMP $W_m$ for $n={{{0}}}$'.format(n), color=bl)#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))
plt.plot(range(m), bs_rand[i, :], label=r'Random $W_m$ for $n={{{0}}}$'.format(n), color=cp[3], lw=1)#r'Random $W_m$, $n={{{0}}}$'.format(n))

ax.set(xlabel='m', ylabel=r'$\beta(V_n, W_m)$', xlim=[0,200], ylim=[0,1])#r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=4)
plt.savefig('WCOMPvsCOMPvsUnif.pdf')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n=20$')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')

#for i, n in enumerate(ns):
i=0
n=ns[i]
plt.plot(range(n+1,m), bs_unif_int[i, n+1:], label=r'Uniformly spaced $W_m$ (optimal)'.format(n), color=cp[1])
plt.plot(range(m), bs_wcomp[i, :], label=r'Worst-case OMP $W_m$'.format(n), color=re)#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))    
plt.plot(range(m), bs_comp[i, :], label=r'Collective OMP $W_m$'.format(n), color=bl)#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))
plt.plot(range(m), bs_rand[i, :], label=r'Random $W_m$'.format(n), color=cp[4])#r'Random $W_m$, $n={{{0}}}$'.format(n))

ax.set(xlabel='m', ylabel=r'$\beta(V_n, W_m)$', xlim=[0,200], ylim=[0,1.05])#r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=4)
plt.savefig('WCOMPvsCOMPvsUnif_20.pdf')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n=40$')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')

i=1
n=ns[i]
plt.plot(range(n+1,m), bs_unif_int[i, n+1:], label=r'Uniformly spaced $W_m$ (optimal)'.format(n), color=cp[1])
plt.plot(range(m), bs_wcomp[i, :], label=r'Worst-case OMP $W_m$'.format(n), color=re)#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))    
plt.plot(range(m), bs_comp[i, :], label=r'Collective OMP $W_m$'.format(n), color=bl)#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))
plt.plot(range(m), bs_rand[i, :], label=r'Random $W_m$'.format(n), color=cp[4])#r'Random $W_m$, $n={{{0}}}$'.format(n))

ax.set(xlabel='m', ylabel=r'$\beta(V_n, W_m)$', xlim=[0,200], ylim=[0,1.05])#r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=4)
plt.savefig('WCOMPvsCOMPvsUnif_40.pdf')
plt.show()


sns.set_palette("deep")
sns.set_style("whitegrid")
cp = sns.color_palette()

bl = (51/255, 133/255, 255/255)
re = (255/255, 102/255, 102/255)


# Plot the evaluation points in the Wm_rand basis 
# (note that the basis is infact orthonormalised so this isn't *quite* an accurate picture)
Wm_points = [vec.params_array(0)[0] for vec in Wms_wcomp[0].vecs]
n = ns[0]

axs = []
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$ for WC-OMP basis, with eval points'.format(n))
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$ and point locations')
plt.plot(range(n,n+40), bs_wcomp[0,20:60], color=re, label=r'$\beta(V_n, W_m)$ for WC-OMP $W_m$')

plt.plot(n * np.ones(n-1), Wm_points[:n-1], 'o', color=cp[5], markersize=4, label='eval point')
plt.plot(n, Wm_points[n-1], 'o', color=cp[2], markersize=6, label='New eval point')
for m_plot in range(n, n+40-1):
    plt.plot((m_plot+1) * np.ones(m_plot), Wm_points[:m_plot], 'o', color=cp[5], markersize=4)
    plt.plot(m_plot+1, Wm_points[m_plot], 'o', color=cp[2], markersize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('PointEvalLoc_WCOMP_20.pdf')
plt.show()


# Plot the evaluation points in the Wm_rand basis 
# (note that the basis is infact orthonormalised so this isn't *quite* an accurate picture)
Wm_points = [vec.params_array(0)[0] for vec in Wms_rand[0].vecs]

axs = []
fig = plt.figure(figsize=(13, 8))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$ for random basis, with eval points'.format(n))
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$ and point locations')
plt.plot(range(n,n+40), bs_rand[0,20:60], color=cp[4], label=r'$\beta(V_n, W_m)$ for random $W_m$')

plt.plot(n * np.ones(n-1), Wm_points[:n-1], 'o', color=cp[5], markersize=4, label='eval point')
plt.plot(n, Wm_points[n-1], 'o', color=cp[2], markersize=6, label='New eval point')
for m_plot in range(n, n+40-1):
    plt.plot((m_plot+1) * np.ones(m_plot), Wm_points[:m_plot], 'o', color=cp[5], markersize=4)
    plt.plot(m_plot+1, Wm_points[m_plot], 'o', color=cp[2], markersize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('PointEvalLoc_rand_20.pdf')
plt.show()


# ### Fix a minimum $\beta^*$ and vary $n$, plot resulting $m^*$
# 
# This result is actually found using the script ```scripts/01_m_star_tests.py``` and the accompanying ```make_01_scripts.py``` which produces the batch job for cluster computers. Here we just import the results which are found in csv files imported from the computing cluster.
# 
# Note that this, at present, is for the value $\beta^* = 0.5$.
# 

data_dir = '../../scripts/omp/'
comp_file = data_dir + 'comp_sin_m_star.csv'
wcomp_file = data_dir + 'wcomp_sin_m_star.csv'

ms_comp = np.loadtxt(comp_file)
ms_wcomp = np.loadtxt(wcomp_file)

sns.set_palette("deep")
sns.set_style("whitegrid")
cp = sns.color_palette()

bl = (51/255, 133/255, 255/255)
re = (255/255, 102/255, 102/255)

axs = []
fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(1, 1, 1, title=r'$\tilde{m}(n)$ for $\beta^*=0.5$, against $n$')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')

plt.plot(ms_wcomp[:,0], ms_wcomp[:,1], label=r'$\tilde{m}(n)$ Worst-case OMP', color=re)
plt.plot(ms_comp[:,0], ms_comp[:,1], label=r'$\tilde{m}(n)$ Collective OMP', color=bl)

ax.set(xlabel='$n$', ylabel=r'$\tilde{m}$', xlim=[20,195])#r'$m$', ylabel=r'$\beta(V_n, W_m)$')
ax.xaxis.set_ticks(np.arange(25, 200, 25))
plt.legend(loc=4)
plt.savefig('m_star_COMP_vs_WCOMP.pdf')
plt.show()


# ### Fix a minimum $\beta^*$ and find resulting $m^*$ with an incremental $V_n$
# 
# That is, for a given $n$, say we have found $m^*(n)$ and we have corresponding $W_{m^*(n)}$, then what is the $m^*(n+1)$ if we *keep* $W_{m^*(n)}$ and use OMP to find the next few measurements to satisfy $\beta(V_{n+1}, W_{m^*(n_1)}) > \beta$?
# 
# This result is actually found using the script ```scripts/03_m_n_incremental.py```. Here we just import the results which are found in csv files imported from the computing cluster.
# 
# Again we have the value $\beta^* = 0.5$.
# 

data_dir = '../../scripts/omp/'
comp_file = data_dir + 'comp_sin_n_incr_m_star.csv'
wcomp_file = data_dir + 'wcomp_sin_n_incr_m_star.csv'

ms_nincr_comp = np.loadtxt(comp_file)
ms_nincr_wcomp = np.loadtxt(wcomp_file)

sns.set_palette("deep")
sns.set_style("whitegrid")
cp = sns.color_palette()

bl = (51/255, 133/255, 255/255)
re = (255/255, 102/255, 102/255)

bl_d = (0.75*51/255, 0.75*133/255, 0.75*255/255)
re_d = (0.75*255/255, 0.75*102/255, 0.75*102/255)

axs = []
fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(1, 1, 1, title=r'$\tilde{m}(n)$ for $\beta^*=0.5$ with $n$-incrementally constructed $W_m$, against $n$')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')

plt.plot(ms_nincr_wcomp[:,0], ms_nincr_wcomp[:,1], '--', label=r'$\tilde{m}(n)$, $n$-incremental $W_m$, Worst-case OMP', color=re_d, lw=1)
plt.plot(ms_nincr_comp[:,0], ms_nincr_comp[:,1], '--', label=r'$\tilde{m}(n)$, $n$-incremental $W_m$,  Collective OMP', color=bl_d, lw=1)

plt.plot(ms_wcomp[:,0], ms_wcomp[:,1], '-', label=r'$\tilde{m}(n)$, Worst-case OMP', color=re)
plt.plot(ms_comp[:,0], ms_comp[:,1], '-', label=r'$\tilde{m}(n)$, Collective OMP', color=bl)

ax.set(xlabel='$n$', ylabel=r'$\tilde{m}$', xlim=[20,195])#r'$m$', ylabel=r'$\beta(V_n, W_m)$')
ax.xaxis.set_ticks(np.arange(25, 200, 25))
plt.legend(loc=4)
plt.savefig('n_incr_COMP_vs_WCOMP.pdf')
plt.show()






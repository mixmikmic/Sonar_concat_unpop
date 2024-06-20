import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Chicago data
# 
# Now with no Gaussian decay
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import sepp.sepp_grid_space
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
_pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*_pred.mesh_data(), _pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Train
# 

trainer = sepp.sepp_grid_space.Trainer2(grid, 20)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


predictor = trainer.to_predictor(model)
predictor.data = trainer.data
pred1 = predictor.predict(datetime.datetime(2017,1,1))
pred2 = predictor.predict(datetime.datetime(2016,9,1))
pred3 = predictor.predict(datetime.datetime(2016,10,15))


fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16,10))

for ax in axes.flat:
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)

bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)

for ax, pred in zip(axes.flat, [bpred, pred1, pred2, pred3]):
    m = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)
fig.tight_layout()


# # How it changes with $r_0$
# 

r0_range = np.exp(np.linspace(0, np.log(200), 100))
models = {}
for r0 in r0_range:
    trainer = sepp.sepp_grid_space.Trainer2(grid, r0)
    trainer.data = points
    models[r0] = trainer.train(datetime.datetime(2017,1,1), iterations=50)


fig, axes = plt.subplots(ncols=2, figsize=(16,3))

axes[0].plot(r0_range, [models[r].theta for r in r0_range], color="black")
axes[0].set(title="theta")
axes[1].plot(r0_range, [1/models[r].omega for r in r0_range], color="black")
axes[1].set(title="1 / omega")

fig.tight_layout()
fig.savefig("../varying_r0_no_g.pdf")


# # Fix everything
# 

trainer = sepp.sepp_grid_space.Trainer2(grid, 300)
trainer.data = points


T, data = trainer.make_data()
model = trainer.initial_model(T, data)
model


for _ in range(50):
    model = sepp.sepp_grid_space.Model2(model.mu, model.T, model.grid, model.theta, 1/20, model.r0)
    opt = trainer._optimiser(model, data)
    model = opt.iterate()


model


bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
fig, ax = plt.subplots(figsize=(8,5))
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
cb = fig.colorbar(m, ax=ax)


# Look at $L = \sum_j \log(b_j + \theta t_j) - n\theta$ so
# $$ \frac{\delta}{\delta \theta} L =
# \sum_j \frac{t_j}{b_j + \theta t_j} - n
# = \sum_j \frac{1}{(t_j/b_j)^{-1} + \theta} - n $$
# 
# So I think that we get a very negative number means that $\theta=0$ is the optimal.
# 

backs = model.background(data)


trigs = [0]
for j in range(1, data.shape[-1]):
    deltas = data[:,j][:,None] - data[:,:j]
    trigs.append( np.sum(model.trigger(None, deltas)) )
trigs = np.asarray(trigs) / model.theta


np.sum(trigs / backs) - len(backs)


model.omega * np.exp(-model.omega) / (np.pi * model.r0**2)


# Force $\theta=0.4$ says...
# 

T, data = trainer.make_data()
model = trainer.initial_model(T, data)
for _ in range(150):
    model = sepp.sepp_grid_space.Model2(model.mu, model.T, model.grid, 0.4, 1/20, model.r0)
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


backs = model.background(data)
trigs = [0]
for j in range(1, data.shape[-1]):
    deltas = data[:,j][:,None] - data[:,:j]
    trigs.append( np.sum(model.trigger(None, deltas)) )
trigs = np.asarray(trigs) / model.theta

np.sum(trigs / backs) - len(backs)


bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
fig, ax = plt.subplots(figsize=(8,5))
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
cb = fig.colorbar(m, ax=ax)


w = trigs / backs
w = w[w>0]
def f(theta):
    return np.sum(1/(1/w + theta)) - len(trigs)
f(0), f(0.01), f(0.1), f(0.2)


theta = np.linspace(0.05, 0.2, 200)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(theta, [f(t) for t in theta])
ax.plot(theta, [0]*len(theta))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Grid based; Histogram estimators
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import sepp.grid_nonparam


# # With real data
# 

import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


# The "South" side works okay...
# 

northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# # Fit the model
# 

trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=1.5)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)


model


pred = trainer.prediction_from_background(model)


fig, axes = plt.subplots(ncols=2, figsize=(16,6))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=ax)
ax.set_title("Estimated background rate")

ax = axes[1]
x = np.arange(10) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
#ax.scatter(x, model.alpha[:len(x)] * model.theta)
ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel")
ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
    model.bandwidth, color="None", edgecolor="black")
#ax.bar(x + (x[1] - x[0]) / 2, model.trigger(None, x), model.bandwidth, color="None", edgecolor="black")
None


np.max(model.mu), np.min(model.mu)


bandwidths = [0.05, 0.15, 0.3, 1]
models = {}
for b in bandwidths:
    trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=b)
    trainer.data = points
    models[b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)
    print(b, models[b])


fig, axes = plt.subplots(ncols=4, figsize=(16,4))

for ax, (b, model), s in zip(axes, models.items(), [600,200,100,30]):
    x = np.arange(s) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
    ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel, h={} days".format(b))
    ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
           model.bandwidth, color="None", edgecolor="black")
fig.tight_layout()
fig.savefig("../north_trigger.pdf")


bandwidths = [0.05, 0.15, 0.3, 1]
models = {}
for b in bandwidths:
    trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=b)
    trainer.data = points
    models[b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=True)
    print(b, models[b])


fig, axes = plt.subplots(ncols=4, figsize=(16,4))

for ax, (b, model), s in zip(axes, models.items(), [600,200,100,30]):
    x = np.arange(s) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
    ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel, h={} days".format(b))
    ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
           model.bandwidth, color="None", edgecolor="black")
fig.tight_layout()


# # Other regions of chicago
# 

sides = ["Far North", "Northwest", "North", "West", "Central",
    "South", "Southwest", "Far Southwest", "Far Southeast"]


def load(side):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points


bandwidths = [0.05, 0.15, 0.3, 1]

models = {}
for side in sides:
    grid, points = load(side)
    models[side] = {}
    for b in bandwidths:
        trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=b)
        trainer.data = points
        try:
            models[side][b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)
        except ValueError as ex:
            #print("Failed because {} for {}/{}".format(ex, side, b))
            print("Failed: {}/{}".format(side, b))
            models[side][b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=True)


fig, axes = plt.subplots(ncols=4, nrows=len(sides), figsize=(16,20))

for side, axe in zip(sides, axes):
    for ax, bw, s in zip(axe, models[side], [900,300,150,50]):
        model = models[side][bw]
        x = np.arange(s) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
        ax.set(xlabel="Days", ylabel="Rate", title="{}, h={} days".format(side, bw))
        ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
               model.bandwidth, color="None", edgecolor="black")

fig.tight_layout()











import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Fixed trigger
# 
# Use an exponential decay in time, and Gaussian in space, and see what background we can fit to the Chicago data.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_fixed
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.naive


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# # Train
# 

tk = sepp.sepp_fixed.ExpTimeKernel(0.2)
sk = sepp.sepp_fixed.GaussianSpaceKernel(50)
trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


fig, axes = plt.subplots(ncols=2, figsize=(16,5))

for ax in axes:
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)

ax = axes[0]
pred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
m = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
cb = fig.colorbar(m, ax=ax)
ax.set_title("SEPP prediction background")

naive = open_cp.naive.CountingGridKernel(grid.xsize, grid.ysize, grid.region())
naive.data = points
pred = naive.predict().renormalise()
ax = axes[1]
m = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
cb = fig.colorbar(m, ax=ax)
ax.set_title("Naive background rate")

None


# ### Speed of convergence
# 
# Our of interest (for our prediction work) we look at how fast the algorithm converges.  10 to 20 iterations is more than enough.
# 

tk = sepp.sepp_fixed.ExpTimeKernel(0.2)
sk = sepp.sepp_fixed.GaussianSpaceKernel(50)
trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.initial_model(T, data)
for _ in range(100):
    opt = trainer._optimiser(model, data)
    old_model = model
    model = opt.iterate()
    print(model, np.mean((model.mu - old_model.mu)**2), (model.theta - old_model.theta)**2)


# ## Dependence on parameters
# 
# Quickly look how $\theta$, the overall trigger rate, varies with $\omega$, the time kernel decay factor.
# 

omegas = np.linspace(0.02, 1, 20)
sigmas = [50] # [10, 25, 50, 100, 250]
models = dict()
for omega in omegas:
    for s in sigmas:
        tk = sepp.sepp_fixed.ExpTimeKernel(omega)
        sk = sepp.sepp_fixed.GaussianSpaceKernel(s)
        trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
        trainer.data = points
        models[(omega,s)] = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig, ax = plt.subplots(figsize=(8,6))

ax.plot([1/o for o in omegas], [models[(o,50)].theta for o in omegas])
ax.set(xlabel="$\omega^{-1}$")
#ax.plot(omegas, [models[(o,50)].theta for o in omegas])
#ax.set(xlabel="$\omega$")
ax.set(ylabel="$\theta$")


omegas_inv = np.linspace(1, 250, 150)
sigmas = [10, 25, 50, 100, 250]
models = dict()
for omega_inv in omegas_inv:
    for s in sigmas:
        tk = sepp.sepp_fixed.ExpTimeKernel(1 / omega_inv)
        sk = sepp.sepp_fixed.GaussianSpaceKernel(s)
        trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
        trainer.data = points
        models[(omega_inv,s)] = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig, ax = plt.subplots(figsize=(12,6))

for s in sigmas:
    ax.plot(omegas_inv, [models[(o,s)].theta for o in omegas_inv])
ax.legend(["$\sigma={}$".format(s) for s in sigmas])
ax.set(xlabel="$\omega^{-1}$")
ax.set(ylabel="$\\theta$")


all_models = []
for sigma, maxoi in zip([10, 25, 50, 100, 250], [500,200,50,20,6]):
    omegas_inv = np.linspace(1, maxoi, 50)
    models = []
    for omega_inv in omegas_inv:
        tk = sepp.sepp_fixed.ExpTimeKernel(1 / omega_inv)
        sk = sepp.sepp_fixed.GaussianSpaceKernel(sigma)
        trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
        trainer.data = points
        models.append( trainer.train(datetime.datetime(2017,1,1), iterations=20) )
    all_models.append((sigma, omegas_inv, models))


fig, axes = plt.subplots(ncols=5, figsize=(18,5))

for ax, (s, ois, models) in zip(axes, all_models):
    ax.plot(ois, [m.theta for m in models], color="black")
    ax.set(xlabel="$\omega^{-1}$")
    #ax.set(ylabel="$\\theta$")
    ax.set(title="$\sigma={}$".format(s))
fig.tight_layout()


fig.savefig("../fixed_grid_1.pdf")





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger.  Now with nearest neighbour estimators.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections
import open_cp.predictors
import scipy.stats
import sepp.kernels


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(trainer, data, model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    t_marginal = sepp.kernels.compute_t_marginal(model.trigger_kernel)
    xy_marginal = sepp.kernels.compute_space_marginal(model.trigger_kernel)
    
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Training
# 
# Nearest neighbour
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15, cutoff=1500)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=1500, time_size=20, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Training
# 
# Nearest neighbour, with no cutoff
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=1500, time_size=20, space_floor=np.exp(-20))
fig.savefig("../no_grid_nnkde_1.pdf")


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))
fig.savefig("../no_grid_nnkde_1a.pdf")





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger, but with the trigger split between space and time.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections, os
import open_cp.predictors
import sepp.kernels
import scipy.stats


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
_pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*_pred.mesh_data(), _pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * model.trigger_time_kernel(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], model.trigger_space_kernel, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Fixed bandwidth
# 

tk_time_prov = sepp.kernels.FixedBandwidthKernelProvider(1)
tk_space_prov = sepp.kernels.FixedBandwidthKernelProvider(20, cutoff=1000)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


fig = plot(model, space_size=900, time_size=40, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=900, time_size=40, space_floor=np.exp(-20))


fig.savefig("../no_grid_kde_split_1.pdf")


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


fig.savefig("../no_grid_kde_split_1a.pdf")


# ## Stochastic EM
# 

tk_time_prov = sepp.kernels.FixedBandwidthKernelProvider(1)
tk_space_prov = sepp.kernels.FixedBandwidthKernelProvider(20, cutoff=1000)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.Optimiser1SEMFactory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=25)
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=1000, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=1000, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Grid based SEPP method(s)
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


# # With real data
# 

import open_cp.sources.chicago
import open_cp.geometry
import opencrimedata.chicago
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


def add_random_noise(points):
    return points
    #ts = points.timestamps + np.random.random(size=points.timestamps.shape) * 60 * 1000 * np.timedelta64(1,"ms")
    #ts = np.sort(ts)
    #return points.from_coords(ts, points.xcoords, points.ycoords)


trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1, timeunit=datetime.timedelta(days=1))
trainer.data = add_random_noise(points)
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)


model


pred = trainer.prediction_from_background(model)


fig, ax = plt.subplots(figsize=(10,6))

ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=ax)
ax.set_title("Estimated background rate")
None


np.max(model.mu), np.min(model.mu)


24 * 60 / model.omega


# Unfortunately, the predicted parameters are not very "realistic", because $\theta$ is small, and $\omega$ indicates a very fast repeat time.
# 

trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1, timeunit=datetime.timedelta(days=1))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)


model


24 * 60 / model.omega


# # Other regions of chicago
# 
# Notice how small $\theta$ gets!
# 

sides = ["Far North", "Northwest", "North", "West", "Central",
    "South", "Southwest", "Far Southwest", "Far Southeast"]


def load(side):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= np.datetime64("2010-01-01")) & (all_points.timestamps < np.datetime64("2011-01-01"))
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points

def train(grid, points):
    trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1)
    trainer.data = add_random_noise(points)
    model = trainer.train(datetime.datetime(2011,1,1), iterations=50)
    return model


for side in sides:
    model = train(*load(side))
    print(side, model.theta, 1/model.omega, np.max(model.mu))


# # Vary the `cutoff`
# 
# Computationally expensive.
# 

grid, points = load("South")


trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1)
trainer.data = add_random_noise(points)
model = trainer.train(datetime.datetime(2011,1,1), iterations=50)
model


model = trainer.train(datetime.datetime(2011,1,1), iterations=100)
model


model = trainer.train(datetime.datetime(2011,1,1), iterations=200)
model


cutoff = [0.1, 0.2, 0.5, 1, 1.5, 2]
lookup = {}
for c in cutoff:
    trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=c)
    trainer.data = add_random_noise(points)
    model = trainer.train(datetime.datetime(2011,1,1), iterations=100)
    lookup[c] = model


lookup


pred = trainer.prediction_from_background(lookup[0.1])
pred.mask_with(grid)
pred = pred.renormalise()

pred1 = trainer.prediction_from_background(lookup[0.5])
pred1.mask_with(grid)
pred1 = pred1.renormalise()

np.max(np.abs(pred.intensity_matrix - pred1.intensity_matrix))











import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, os
import open_cp.predictors
import open_cp.kernels
import open_cp.seppexp
import open_cp.naive
import open_cp.evaluation
import open_cp.logger
open_cp.logger.log_to_true_stdout()


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


def load(side, start, end):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= start) & (all_points.timestamps < end)
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points, geo

grid, points, geo = load("South", np.datetime64("2010-01-01"), np.datetime64("2010-09-01"))


trainer = open_cp.seppexp.SEPPTrainer(grid=grid)
trainer.data = points
predictor = trainer.train(cutoff_time=np.datetime64("2011-09-01"), iterations=50)


back = predictor.background_prediction()
predictor.data = points
pred = predictor.predict(np.datetime64("2011-09-01T12:00"))
predictor.theta, predictor.omega * 60 * 24


npredictor = open_cp.naive.CountingGridKernel(grid.xsize, region=grid.region())
npredictor.data = points
naive = npredictor.predict()
naive.mask_with(grid)
naive = naive.renormalise()


fig, axes = plt.subplots(ncols=3, figsize=(16,6))

for ax in axes:
    ax.add_patch(descartes.PolygonPatch(geo, fc="none"))
    ax.set_aspect(1)

mappable = axes[0].pcolor(*back.mesh_data(), back.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[0])
axes[0].set_title("Estimated background rate")

mappable = axes[1].pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[1])
axes[1].set_title("Full prediction")

mat = pred.intensity_matrix - back.intensity_matrix
mappable = axes[2].pcolor(*pred.mesh_data(), mat, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[2])
axes[2].set_title("Difference")

fig.tight_layout()


fig, axes = plt.subplots(ncols=3, figsize=(16,6))

for ax in axes:
    ax.add_patch(descartes.PolygonPatch(geo, fc="none"))
    ax.set_aspect(1)

nback = back.renormalise()
mappable = axes[0].pcolor(*nback.mesh_data(), nback.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[0])
axes[0].set_title("Normalised background rate")

mappable = axes[1].pcolor(*naive.mesh_data(), naive.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[1])
axes[1].set_title("'Naive' prediction")

mat = naive.intensity_matrix - nback.intensity_matrix
mappable = axes[2].pcolor(*pred.mesh_data(), mat, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[2])
axes[2].set_title("Difference")

fig.tight_layout()


# # Compare hit rates
# 
# Cannot be distinguished.
# 

class SeppExpProvider(open_cp.evaluation.StandardPredictionProvider):
    def give_prediction(self, grid, points, time):
        trainer = open_cp.seppexp.SEPPTrainer(grid=grid)
        trainer.data = points
        predictor = trainer.train(cutoff_time=time, iterations=50)
        return predictor.background_prediction()


grid, points, geo = load("South", np.datetime64("2010-01-01"), np.datetime64("2011-01-01"))


provider = open_cp.evaluation.NaiveProvider(points, grid)
evaluator = open_cp.evaluation.HitRateEvaluator(provider)
evaluator.data = points
time_range = evaluator.time_range(datetime.datetime(2010,9,1),
        datetime.datetime(2010,12,31), datetime.timedelta(days=1))
result = evaluator.run(time_range, range(1,100))


provider = SeppExpProvider(points, grid)
evaluator = open_cp.evaluation.HitRateEvaluator(provider)
evaluator.data = points
time_range = evaluator.time_range(datetime.datetime(2010,9,1),
        datetime.datetime(2010,12,31), datetime.timedelta(days=1))
result1 = evaluator.run(time_range, range(1,100))


import pandas as pd


frame = pd.DataFrame(result.rates).T.describe().T
frame.head()


frame1 = pd.DataFrame(result1.rates).T.describe().T
frame1.head()


fig, ax = plt.subplots(figsize=(14,8))

ax.plot(frame["mean"]*100, label="naive")
ax.plot(frame1["mean"]*100, label="sepp")


# # With cutoff
# 

import sepp.sepp_grid

class SeppExpProvider(open_cp.evaluation.StandardPredictionProvider):
    def give_prediction(self, grid, points, time):
        trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=0.5)
        trainer.data = points # Add noise??
        model = trainer.train(time, iterations=50)
        return trainer.prediction_from_background(model)


grid, points, geo = load("South", np.datetime64("2010-01-01"), np.datetime64("2011-01-01"))


provider = open_cp.evaluation.NaiveProvider(points, grid)
evaluator = open_cp.evaluation.HitRateEvaluator(provider)
evaluator.data = points
time_range = evaluator.time_range(datetime.datetime(2010,9,1),
        datetime.datetime(2010,12,31), datetime.timedelta(days=1))
result = evaluator.run(time_range, range(1,100))


provider = SeppExpProvider(points, grid)
evaluator = open_cp.evaluation.HitRateEvaluator(provider)
evaluator.data = points
time_range = evaluator.time_range(datetime.datetime(2010,9,1),
        datetime.datetime(2010,12,31), datetime.timedelta(days=1))
result1 = evaluator.run(time_range, range(1,100))


import pandas as pd
frame = pd.DataFrame(result.rates).T.describe().T
frame1 = pd.DataFrame(result1.rates).T.describe().T


fig, ax = plt.subplots(figsize=(14,8))

ax.plot(frame["mean"]*100, label="naive")
ax.plot(frame1["mean"]*100, label="sepp")





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger, but with the trigger split between space and time.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections, os
import open_cp.predictors
import sepp.kernels
import scipy.stats


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * model.trigger_time_kernel(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], model.trigger_space_kernel, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Nearest neighbour, variable bandwidth
# 

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(20)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=5)
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# This gets very, very slow: each iteration taking a couple of hours...
# 

# ## Stochastic EM
# 

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(20)
opt_fac = sepp.sepp_full.Optimiser1SEMFactory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac) 
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=10)
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


backgrounds, trigger_deltas = trainer.sample_to_points(model, datetime.datetime(2017,1,1))


backgrounds.shape, trigger_deltas.shape


# # Nearest neighbour
# 
# Now with k=15 for spatial components
# 

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=5)
model


fig = plot(model, space_size=1000, time_size=25, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Stochastic EM
# 

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
opt_fac = sepp.sepp_full.Optimiser1SEMFactory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac) 
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=15)
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# HERE!


import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("Southwest")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
print(points.number_data_points)
points = open_cp.geometry.intersect_timed_points(points, northside)
print(points.number_data_points)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)


cells = trainer.to_cells(datetime.datetime(2017,1,1))


cells.shape


cells, model = trainer.initial_model(datetime.datetime(2017,1,1))
model


for _ in range(50):
    opt = sepp.sepp_grid.ExpDecayOptFast(model, cells)
    model = opt.optimised_model()
    print(model)


# ### Add a random pertubation to the timestamps
# 
# So long as we remember to re-order, this works fine.
# 

cells, model = trainer.initial_model(datetime.datetime(2017,1,1))
new_cells = []
for cell in cells.flat:
    if len(cell) > 0:
        cell += np.random.random(size=len(cell)) * 0.1
        cell.sort()
        assert all(x<0 for x in cell)
    new_cells.append(cell)
cells = np.asarray(new_cells).reshape(cells.shape)


for _ in range(50):
    opt = sepp.sepp_grid.ExpDecayOptFast(model, cells)
    model = opt.optimised_model()
    print(model)


# ## Check the cause of the repeated timestamp...
# 

import impute.chicago
import shapely.geometry


def gen():
    with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
        yield from impute.chicago.load_only_with_point(file)
next(gen())


proj = impute.chicago.projector()
rows = []
for row in gen():
    in_time_range = row.datetime >= datetime.datetime(2016,1,1) and row.datetime < datetime.datetime(2017,1,1)
    if in_time_range and row.crime_type=="BURGLARY":
        point = shapely.geometry.Point(*proj(*row.point))
        if northside.intersects(point):
            rows.append(row)
rows.sort(key = lambda row : row.datetime)
len(rows)


points.number_data_points


cells = np.empty((grid.yextent, grid.xextent), dtype=np.object)
for x in range(grid.xextent):
    for y in range(grid.yextent):
        cells[y, x] = list()
for row in rows:
    x, y = grid.grid_coord(*proj(*row.point))
    cells[y,x].append(row)


[x for x in cells.flat if len(x)>10]


# So we see here two exact repeats (with the same coordinates, not just the same grid cells).
# 
# An examination of the raw CSV files shows these have different "Case Number"s as well.
# 
# It is hard to know, but given that times are probably estimates and/or rounded, and that address is to the block only, and that we believe the location is somewhat random, we probably feel that these are "near repeats" not exact repeats.
# 




import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Grid based SEPP method(s)
# 
# Using KDE based trigger function.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import sepp.grid_nonparam


import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# # Fit the model
# 

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(1))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


def plot(model, trigger_limit=20):
    fig, axes = plt.subplots(ncols=2, figsize=(16,6))

    ax = axes[0]
    pred = trainer.prediction_from_background(model)
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
    fig.colorbar(mappable, ax=ax)
    ax.set_title("Estimated background rate")

    ax = axes[1]
    x = np.linspace(0, trigger_limit, 200)
    x = x * (trainer.time_unit / np.timedelta64(1,"D"))
    ax.plot(x, model.trigger_func(x) * model.theta)
    ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel")
    
    return fig


fig = plot(model)


np.max(model.mu), np.min(model.mu), np.sum(model.mu)


# ## Investigate different initial conditions
# 
# It doesn't seem to really matter!  Which is good!
# 

cells, modeli = trainer.initial_model(datetime.datetime(2017,1,1))


def func(t):
    return np.exp(-t / 100)


model = sepp.grid_nonparam.KDEModel(modeli.mu, modeli.T, modeli.theta, func)
model


for _ in range(50):
    opt = trainer.provider.make_opt(model, cells)
    model = opt.optimised_model()


fig = plot(model)


# # Vary the bandwidth
# 

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.05))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


fig = plot(model, 15)


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.1))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


fig = plot(model, 15)


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.2))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


fig = plot(model, 15)


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.5))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
plot(model)
model


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(2))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
plot(model)
model


# # Use a variable bandwidth KDE
# 

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(5))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
_ = plot(model, 15)
model


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(15))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
_ = plot(model)
model


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(30))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
_ = plot(model)
model


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(50))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
_ = plot(model)
model


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(100))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
_ = plot(model)
model


# # Summary plots for article
# 

bandwidths = [0.05, 0.1, 1, 2]

models = {}
for bw in bandwidths:
    trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(bw))
    trainer.data = points
    models[bw] = trainer.train(datetime.datetime(2017,1,1), iterations=50)


fig, axes = plt.subplots(ncols=4, figsize=(16,4))

for ax, bw in zip(axes, models):
    model = models[bw]

    x = np.linspace(0, 20, 200)
    x = x * (trainer.time_unit / np.timedelta64(1,"D"))
    ax.plot(x, model.trigger_func(x) * model.theta, color="black")
    ax.set(xlabel="Days", ylabel="Rate", title="Bandwidth of {} days".format(bw))
    
fig.tight_layout()
fig.savefig("../grid_kde_by_bandwidth.pdf")


nearestns = [5, 10, 20, 50]

models = {}
for k in nearestns:
    trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(k))
    trainer.data = points
    models[k] = trainer.train(datetime.datetime(2017,1,1), iterations=50)


fig, axes = plt.subplots(ncols=4, figsize=(16,4))

for ax, bw in zip(axes, models):
    model = models[bw]

    x = np.linspace(0, 20, 200)
    x = x * (trainer.time_unit / np.timedelta64(1,"D"))
    ax.plot(x, model.trigger_func(x) * model.theta, color="black")
    ax.set(xlabel="Days", ylabel="Rate", title="Nearest neighbours: {}".format(bw))
    
fig.tight_layout()
fig.savefig("../grid_kde_by_nn.pdf")


# # Other regions of chicago
# 
# Using other years of data is interesting...
# 

sides = ["Far North", "Northwest", "North", "West", "Central",
    "South", "Southwest", "Far Southwest", "Far Southeast"]


def load(side):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points


bandwidths = [0.05, 0.1, 1, 2]

models = {}
for side in sides:
    grid, points = load(side)

    models[side] = {}
    for bw in bandwidths:
        trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(bw))
        trainer.data = points
        models[side][bw] = trainer.train(datetime.datetime(2017,1,1), iterations=50)


import pickle
with open("kde_temp.pic", "wb") as f:
    pickle.dump(models, f)


fig, axes = plt.subplots(ncols=4, nrows=len(sides), figsize=(16,20))

for side, axe in zip(sides, axes):
    for ax, bw in zip(axe, models[side]):
        model = models[side][bw]

        x = np.linspace(0, 50, 200)
        x = x * (trainer.time_unit / np.timedelta64(1,"D"))
        ax.plot(x, model.trigger_func(x) * model.theta, color="black")
        ax.set(xlabel="Days", ylabel="Rate", title="{} / {} days".format(side, bw))
    
fig.tight_layout()








import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Grid based SEPP method(s)
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


# # With real data
# 

import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels


#datadir = os.path.join("..", "..", "..", "..", "..", "Data")
datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


def add_random_noise(points):
    ts = points.timestamps + np.random.random(size=points.timestamps.shape) * 60 * 1000 * np.timedelta64(1,"ms")
    ts = np.sort(ts)
    return points.from_coords(ts, points.xcoords, points.ycoords)


trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1, timeunit=datetime.timedelta(days=1))
trainer.data = add_random_noise(points)
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)


model


pred = trainer.prediction_from_background(model)


fig, ax = plt.subplots(figsize=(10,6))

ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=ax)
ax.set_title("Estimated background rate")
None


np.max(model.mu), np.min(model.mu)


24 * 60 / model.omega


# Unfortunately, the predicted parameters are not very "realistic".  The triggering kernel is
# $$ g(t) = \theta \omega e^{-\omega t} $$
# with time measured in "days".  We estimate $1 / \omega \approx 419$ _minutes_, and $\theta$ is small.   This means that events need to be very near in time before the triggering kernel adds much.
# 

trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1, timeunit=datetime.timedelta(days=1))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)


model


24 * 60 / model.omega


# # Other regions of chicago
# 
# Notice how small $\theta$ gets!
# 

sides = ["Far North", "Northwest", "North", "West", "Central",
    "South", "Southwest", "Far Southwest", "Far Southeast"]


def load(side):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= np.datetime64("2010-01-01")) & (all_points.timestamps < np.datetime64("2011-01-01"))
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points

def train(grid, points):
    trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1)
    trainer.data = add_random_noise(points)
    model = trainer.train(datetime.datetime(2011,1,1), iterations=50)
    return model


for side in sides:
    model = train(*load(side))
    print(side, model.theta, 1/model.omega, np.max(model.mu))


# # Vary the `cutoff`
# 
# Computationally expensive.
# 

grid, points = load("South")


trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1)
trainer.data = add_random_noise(points)
model = trainer.train(datetime.datetime(2011,1,1), iterations=50)
model


model = trainer.train(datetime.datetime(2011,1,1), iterations=100)
model


model = trainer.train(datetime.datetime(2011,1,1), iterations=200)
model


cutoff = [0.1, 0.2, 0.5, 1, 1.5, 2]
lookup = {}
for c in cutoff:
    trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=c)
    trainer.data = add_random_noise(points)
    model = trainer.train(datetime.datetime(2011,1,1), iterations=100)
    lookup[c] = model


lookup


pred = trainer.prediction_from_background(lookup[0.1])
pred.mask_with(grid)
pred = pred.renormalise()

pred1 = trainer.prediction_from_background(lookup[0.5])
pred1.mask_with(grid)
pred1 = pred1.renormalise()

np.max(np.abs(pred.intensity_matrix - pred1.intensity_matrix))








import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Chicago data
# 
# Now with no Gaussian decay, and a histogram in time
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import sepp.sepp_grid_space
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels


#datadir = os.path.join("..", "..", "..", "..", "..", "Data")
datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Train
# 

def plot(model, histlen):
    fig, axes = plt.subplots(ncols=2, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    ax = axes[1]
    x = np.arange(histlen) * model.bandwidth
    ax.bar(x + model.bandwidth/2, model.alpha_array[:len(x)] * model.theta / model.bandwidth,
           model.bandwidth, color="none", edgecolor="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    None


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 30)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=0.5)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 40)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=0.1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 40)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=0.8)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 40)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=1.2)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 35)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=1.3)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 35)


# ## Different $r_0$ value
# 
# This becomes computationally expensive...
# 

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=100, bandwidth=1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 10)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=100, bandwidth=0.1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 30)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=250, bandwidth=1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 6)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=250, bandwidth=0.1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 60)





get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# # Normal in 2D
# 
# We work in polar coordinates, so it's important to remember to integrate properly.  If our density depends on the radius $r$ then to be a probability density we need
# $$ 2 \pi \int_0^\infty r f(r) \ dr = 1 $$
# 

fig, ax = plt.subplots(figsize=(12,6))

x = np.linspace(0,4,200)
y = np.exp(-x*x/2)

ax.plot(x, y)
ax.plot(x, x*y)
None


np.sum(x*y) * (x[1]-x[0])


# ## Exponential
# 
# (Here and below I omit the $2\pi$ factor.  Don't forget this!)
# 
# $$ f(r) = \omega^2 e^{-\omega r} $$
# so $f = F'$ where $F(r) = - \omega e^{-\omega r}$ and hence
# $$ \int_0^r x f(x) \ dx = \big[ xF(x) \big]_0^r - \int_0^r F(x) \ dx 
# = - r \omega e^{-\omega r} + \int_0^r \omega e^{-\omega x} \ dx 
# = - r \omega e^{-\omega r} + 1 - e^{-\omega r}
# = 1 - (1+\omega r) e^{-\omega r}
# $$
# 

fig, ax = plt.subplots(ncols=2, figsize=(16,6))

x = np.linspace(0,10,1000)
omega = 1
y = omega * omega * np.exp(-x * omega)

ax[0].plot(x, x*y)
ax[1].plot(x, 1 - (1 + omega * x) * np.exp(-omega * x))
None


np.sum(x*y) * (x[1]-x[0])


# ## Cauchy like
# 
# Not sure this one has a name,
# 
# $$ f(r) = \frac{2e}{(1+r^2)^{1+e}} $$
# 
# for $e>0$.  Then
# $$ \int_0^r x f(x) \ dx = \int_0^r 2ex (1+x^2)^{-1-e} \ dx
# = - \big[ (1+x^2)^{-e} \big]_0^r
# = 1 - (1+r^2)^{-e}
# $$
# 

fig, ax = plt.subplots(ncols=2, figsize=(16,6))

x = np.linspace(0,10,1000)
e = 1
y = (e+e) / ((1+x*x)**(1+e))

ax[0].plot(x, x*y)
ax[1].plot(x, (1-(1+x*x)**(-e)))
None


np.sum(x*y) * (x[1]-x[0])








import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger.  Now with nearest neighbour estimators.
# 
# Here we use a fixed bandwidth KDE for the background.  It sort of works, but nothing very interesting.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections
import open_cp.predictors
import scipy.stats
import sepp.kernels


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(trainer, data, model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    t_marginal = sepp.kernels.compute_t_marginal(model.trigger_kernel)
    xy_marginal = sepp.kernels.compute_space_marginal(model.trigger_kernel)
    
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Training
# 
# Mix of nearest neighbour and fixed bandwidth
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15, cutoff=1500)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=1500, time_size=20, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(trainer, data, model, space_size=1000, time_size=10, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ### Larger background bandwidth
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15, cutoff=1500)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(250)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=1500, time_size=20, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections
import open_cp.predictors
import scipy.stats
import sepp.kernels


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(trainer, data, model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    # Quickly compute marginals...
    opt = trainer._optimiser(model, data)
    x, w = opt.data_for_trigger_opt()
    prov = opt_fac._trigger_provider
    ker = open_cp.kernels.GaussianBase(x)
    ker.covariance_matrix = np.diag(prov._scale)
    ker.bandwidth = prov._h
    ker.weights = w
    xy_marginal = open_cp.kernels.marginalise_gaussian_kernel(ker, axis=0)
    tx_marginal = open_cp.kernels.marginalise_gaussian_kernel(ker, axis=2)
    t_marginal = open_cp.kernels.marginalise_gaussian_kernel(tx_marginal, axis=1)
    
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Fixed bandwidth
# 

tk_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(1, scale=[1,20,20])
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(20)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Stochastic EM
# 

tk_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(20, scale=[0.01,1,1])
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(20)
opt_fac = sepp.sepp_full.OptimiserSEMFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import opencrimedata.chicago
import descartes
import pickle, lzma, datetime, collections
import open_cp.predictors
import scipy.stats
import sepp.kernels


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
_pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*_pred.mesh_data(), _pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    xy_marginal = sepp.kernels.compute_space_marginal(model.trigger_kernel)
    t_marginal = sepp.kernels.compute_t_marginal(model.trigger_kernel)
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Fixed bandwidth
# 

tk_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(1, scale=[1,20,20])
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)
model


fig = plot(model, space_size=600, time_size=35, space_floor=np.exp(-20))


#fig.savefig("../no_grid_kde_1.pdf")


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


#fig.savefig("../no_grid_kde_1a.pdf")


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=750, time_size=50, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Stochastic EM
# 

tk_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(1, scale=[1,20,20])
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.OptimiserSEMFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=20)


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger, but with the trigger split between space and time.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections, os
import open_cp.predictors
import sepp.kernels
import scipy.stats
import opencrimedata.chicago


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

from plotting_split import *


# ## Nearest neighbour, variable bandwidth
# 

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(20)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=5)
model


fig = plot(model, space_size=1000, time_size=20, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# This gets very, very slow: each iteration taking a couple of hours...
# 

# ## Stochastic EM
# 

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(20)
opt_fac = sepp.sepp_full.Optimiser1SEMFactory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac) 
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=15)
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


backgrounds, trigger_deltas = trainer.sample_to_points(model, datetime.datetime(2017,1,1))
backgrounds.shape, trigger_deltas.shape


# ## Nearest neighbour
# 
# Now with k=15 for spatial components
# 

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=5)
model


fig = plot(model, space_size=800, time_size=15, space_floor=np.exp(-20), geo=northside, grid=grid)


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(10):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()


fig = plot(model, space_size=1300, time_size=15, space_floor=np.exp(-20), geo=northside, grid=grid)


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ### Stochastic EM
# 

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
opt_fac = sepp.sepp_full.Optimiser1SEMFactory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac) 
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=15)
model


fig = plot(model, space_size=1300, time_size=15, space_floor=np.exp(-20), geo=northside, grid=grid)


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger.  Now with nearest neighbour estimators.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections
import open_cp.predictors
import scipy.stats
import sepp.kernels
import opencrimedata.chicago


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(trainer, data, model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    t_marginal = sepp.kernels.compute_t_marginal(model.trigger_kernel)
    xy_marginal = sepp.kernels.compute_space_marginal(model.trigger_kernel)
    
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Stochastic EM
# 
# Both nearest neighbour.  Tends to blow up after too many iterations.  We also tried with different initial conditions, but this doesn't help.
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
opt_fac = sepp.sepp_full.OptimiserSEMFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=10)


fig = plot(trainer, data, model, space_size=1500, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Different initial conditions
# 
# Rosser et al. use an initial p matrix of
# $$ p_{i,j} = \exp\big(-\alpha(t_j-t_i)\big)
# \exp\Big( \frac{-(x_j-x_i)^2-(y_j-y_i)^2}{2\beta^2} \Big). $$
# They use $\alpha=0.1$ (units of day${}^{-1}$) and $\beta=50$ (units of meter).  Thus the initial "background" diagonal entries are all $1$.
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
opt_fac = sepp.sepp_full.OptimiserSEMFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
original_model = trainer.initial_model(T, data)


original_model


def initial_background(pts):
    assert len(pts.shape) == 2
    assert pts.shape[0] == 2
    return np.zeros(pts.shape[1]) + 1

def initial_trigger(pts):
    times = pts[0]
    distsq = pts[1]**2 + pts[2]**2
    value = np.exp(-0.1 * times) * np.exp(-distsq/(2*50*50)) 
    #if pts.shape[1] > 0:
    #    raise Exception(pts, value)
    return value

model = sepp.sepp_full.Model(original_model.T, original_model.mu, initial_background, 1, initial_trigger)


opt = trainer._optimiser(model, data)
opt.p


for _ in range(5):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()


fig = plot(trainer, data, model, space_size=1500, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(10):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()


fig = plot(trainer, data, model, space_size=1500, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# HERE!!!!





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
#import open_cp.evaluation
import descartes
import pickle, lzma, datetime, collections
import open_cp.predictors
#import open_cp.kernels
import scipy.stats
#import scipy.integrate
import sepp.kernels


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
_pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*_pred.mesh_data(), _pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    xy_marginal = sepp.kernels.compute_space_marginal(model.trigger_kernel)
    t_marginal = sepp.kernels.compute_t_marginal(model.trigger_kernel)
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Fixed bandwidth
# 

tk_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(1, scale=[1,20,20])
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)
model


fig = plot(model, space_size=600, time_size=35, space_floor=np.exp(-20))


fig.savefig("../no_grid_kde_1.pdf")


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


fig.savefig("../no_grid_kde_1a.pdf")


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=750, time_size=50, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Stochastic EM
# 

tk_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(1, scale=[1,20,20])
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.OptimiserSEMFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=20)


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger, but with the trigger split between space and time.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections, os
import open_cp.predictors
import sepp.kernels
import scipy.stats


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * model.trigger_time_kernel(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], model.trigger_space_kernel, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Fixed bandwidth
# 

tk_time_prov = sepp.kernels.FixedBandwidthKernelProvider(1)
tk_space_prov = sepp.kernels.FixedBandwidthKernelProvider(30, cutoff=750)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(30)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Stochastic EM
# 

tk_time_prov = sepp.kernels.FixedBandwidthKernelProvider(1)
tk_space_prov = sepp.kernels.FixedBandwidthKernelProvider(30, cutoff=750)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(30)
opt_fac = sepp.sepp_full.Optimiser1SEMFactory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=25)
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=1000, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=1000, time_size=30, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger.  Now with nearest neighbour estimators.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections
import open_cp.predictors
import scipy.stats
import sepp.kernels


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(trainer, data, model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    t_marginal = sepp.kernels.compute_t_marginal(model.trigger_kernel)
    xy_marginal = sepp.kernels.compute_space_marginal(model.trigger_kernel)
    
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Stochastic EM
# 
# Both nearest neighbour.  Tends to blow up after too many iterations.  We also tried with different initial conditions, but this doesn't help.
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
opt_fac = sepp.sepp_full.OptimiserSEMFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=10)


fig = plot(trainer, data, model, space_size=1500, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Different initial conditions
# 
# Rosser et al. use an initial p matrix of
# $$ p_{i,j} = \exp\big(-\alpha(t_j-t_i)\big)
# \exp\Big( \frac{-(x_j-x_i)^2-(y_j-y_i)^2}{2\beta^2} \Big). $$
# They use $\alpha=0.1$ (units of day${}^{-1}$) and $\beta=50$ (units of meter).  Thus the initial "background" diagonal entries are all $1$.
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
opt_fac = sepp.sepp_full.OptimiserSEMFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
original_model = trainer.initial_model(T, data)


original_model


def initial_background(pts):
    assert len(pts.shape) == 2
    assert pts.shape[0] == 2
    return np.zeros(pts.shape[1]) + 1

def initial_trigger(pts):
    times = pts[0]
    distsq = pts[1]**2 + pts[2]**2
    value = np.exp(-0.1 * times) * np.exp(-distsq/(2*50*50)) 
    #if pts.shape[1] > 0:
    #    raise Exception(pts, value)
    return value

model = sepp.sepp_full.Model(original_model.T, original_model.mu, initial_background, 1, initial_trigger)


opt = trainer._optimiser(model, data)
opt.p


for _ in range(5):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()


fig = plot(trainer, data, model, space_size=1500, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(10):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()


fig = plot(trainer, data, model, space_size=1500, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Grid based SEPP method(s)
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


# # With real data
# 

import opencrimedata.chicago
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels


#datadir = os.path.join("/media", "disk", "Data")
datadir = os.path.join("..", "..", "..", "..", "..", "Data")
with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


open_cp.sources.chicago.set_data_directory(datadir)
northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Continue with analysis
# 

#def add_random_noise(points):
#    ts = points.timestamps + np.random.random(size=points.timestamps.shape) * 60 * 1000 * np.timedelta64(1,"ms")
#    ts = np.sort(ts)
#    return points.from_coords(ts, points.xcoords, points.ycoords)

def add_random_noise(points):
    return points


trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
trainer.data = add_random_noise(points)
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)


model


pred = trainer.prediction_from_background(model)


fig, ax = plt.subplots(figsize=(10,6))

ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=ax)
ax.set_title("Estimated background rate")
None


np.max(model.mu), np.min(model.mu)


# Unfortunately, the predicted parameters are not very "realistic".  The triggering kernel is
# $$ g(t) = \theta \omega e^{-\omega t} $$
# with time measured in "days".  We estimate $1 / \omega \approx 225$ _minutes_, and $\theta$ comparable to the background rate.   This means that events need to be very near in time before the triggering kernel adds much.
# 

trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
trainer.data = add_random_noise(points)
model = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)


model


24 * 60 / model.omega


# ## With old code
# 
# Check that we get the same result with the main `open_cp` code.  (For a while, this had poorly chosen initial conditions, which lead to convergence problems.  I've now fixed this.)
# 

import open_cp.seppexp


trainer = open_cp.seppexp.SEPPTrainer(grid=grid)
trainer.data = add_random_noise(points)
predictor = trainer.train(iterations=50)


predictor.theta, predictor.omega * 60 * 24


predictor = trainer.train(iterations=50, use_corrected=True)


predictor.theta, predictor.omega * 60 * 24


# ## Allowing repeats
# 
# We'll also bin the data to the nearest day (moving to noon each day).
# 
# Notice that $\theta$ is now tiny!  Indeed, using either edge correction or not, if we run the algorithm for longer, then $\theta$ tends to 0.
# 

pts = points.bin_timestamps(np.datetime64("2017-01-01T12:00"), np.timedelta64(1, "D"))


trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
trainer.data = pts
model = trainer.train(datetime.datetime(2017,1,1), iterations=200, use_fast=False)
model


1 / model.omega


# Binning to 12 hours seems to work better, but $\omega^{-1}$ is still a bit small.
# 
# Changing the offset to 6am gets back the old $\theta \approx 0$ behaviour.  After worrying a bit this was the algorithm, but I think it shows some unexpected dependence on the distribution of timestamps in the input data.
# - If we bin to the nearest 12 hours, with an offset of midnight, it "works"
# - Changing the offset to 6am, and we get a much smaller $\theta$
# - If we subtract 6 hours from the original timestamps, and then bin with the offset of 6am, it goes back to "working" (as it should!)
# 

trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
trainer.data = points.bin_timestamps(np.datetime64("2017-01-01T12:00"), np.timedelta64(12, "h"))
model = trainer.train(datetime.datetime(2017,1,2, 0,0), iterations=200, use_fast=False)
model


trainer.data.time_range


cells, _ = trainer.make_points(datetime.datetime(2017,1,2, 0,0))
min([np.min(x) for x in cells.flat if len(x)>0]), max([np.max(x) for x in cells.flat if len(x)>0])


trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
trainer.data = points.bin_timestamps(np.datetime64("2017-01-01T06:00"), np.timedelta64(12, "h"))
model = trainer.train(datetime.datetime(2017,1,1), iterations=200, use_fast=False)
model


trainer.data.time_range


cells, _ = trainer.make_points(datetime.datetime(2017,1,1))
min([np.min(x) for x in cells.flat if len(x)>0]), max([np.max(x) for x in cells.flat if len(x)>0])


ts = points.timestamps - np.timedelta64(6, "h")
pts = open_cp.data.TimedPoints(ts, points.coords)

trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
trainer.data = pts.bin_timestamps(np.datetime64("2017-01-01T06:00"), np.timedelta64(12, "h"))
model = trainer.train(datetime.datetime(2017,1,1), iterations=200, use_fast=False)
model


trainer.data.time_range


cells, _ = trainer.make_points(datetime.datetime(2017,1,1))
min([np.min(x) for x in cells.flat if len(x)>0]), max([np.max(x) for x in cells.flat if len(x)>0])


# ## More systematically explore binning offset
# 

for hour in range(24):
    trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
    trainer.data = points.bin_timestamps(np.datetime64("2017-01-01T00:00")
                        + np.timedelta64(hour, "h"), np.timedelta64(1, "D"))
    model = trainer.train(datetime.datetime(2017,1,2, 0,0), iterations=50, use_fast=False)
    print(hour, model)


by_hour = {}
for hour in range(24):
    trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
    trainer.data = points.bin_timestamps(np.datetime64("2017-01-01T00:00")
                        + np.timedelta64(hour, "h"), np.timedelta64(12, "h"))
    model = trainer.train(datetime.datetime(2017,1,2, 0,0), iterations=100, use_fast=False)
    print(hour, model)
    by_hour[hour] = model


fig, ax = plt.subplots(figsize=(16, 5))
x = list(by_hour.keys())
x.sort()
y = [by_hour[t].theta for t in x]
ax.plot(x,y)


# # Other regions of chicago
# 

sides = ["Far North", "Northwest", "North", "West", "Central",
    "South", "Southwest", "Far Southwest", "Far Southeast"]


def load(side):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= np.datetime64("2010-01-01")) & (all_points.timestamps < np.datetime64("2011-01-01"))
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points

def train(grid, points):
    trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
    trainer.data = add_random_noise(points)
    model = trainer.train(datetime.datetime(2011,1,1), iterations=50)
    return model


for side in sides:
    model = train(*load(side))
    print(side, model.theta, 24*60/model.omega, np.max(model.mu))


def train(grid, points):
    trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
    trainer.data = points.bin_timestamps(np.datetime64("2017-01-01T12:00"), np.timedelta64(12, "h"))
    model = trainer.train(datetime.datetime(2011,1,1), iterations=50, use_fast=False)
    return model


for side in sides:
    model = train(*load(side))
    print(side, model.theta, 24*60/model.omega, np.max(model.mu))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Grid based SEPP method(s)
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import sepp.grid_nonparam


# # With real data
# 

import open_cp.sources.chicago
import open_cp.geometry
import opencrimedata.chicago
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("South")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Train
# 

trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=1.5)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=True)


model


pred = trainer.prediction_from_background(model)


fig, axes = plt.subplots(ncols=2, figsize=(16,6))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=ax)
ax.set_title("Estimated background rate")

ax = axes[1]
x = np.arange(10) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
#ax.scatter(x, model.alpha[:len(x)] * model.theta)
ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel")
ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
    model.bandwidth, color="None", edgecolor="black")
#ax.bar(x + (x[1] - x[0]) / 2, model.trigger(None, x), model.bandwidth, color="None", edgecolor="black")
None


np.max(model.mu), np.min(model.mu)


# ## Varying bandwidth
# 

bandwidths = [0.05, 0.15, 0.3, 1]
models = {}
for b in bandwidths:
    trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=b)
    trainer.data = points
    models[b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=True)
    print(b, models[b])


fig, axes = plt.subplots(ncols=4, figsize=(16,4))

for ax, (b, model), s in zip(axes, models.items(), [600,200,100,30]):
    x = np.arange(s) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
    ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel, h={} days".format(b))
    ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
           model.bandwidth, color="None", edgecolor="black")
fig.tight_layout()
#fig.savefig("../south_trigger.pdf")


bandwidths = [0.05, 0.15, 0.3, 1]
models = {}
for b in bandwidths:
    trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=b)
    trainer.data = points
    models[b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=True)
    print(b, models[b])


fig, axes = plt.subplots(ncols=4, figsize=(16,4))

for ax, (b, model), s in zip(axes, models.items(), [600,200,100,30]):
    x = np.arange(s) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
    ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel, h={} days".format(b))
    ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
           model.bandwidth, color="None", edgecolor="black")
fig.tight_layout()


# # Other regions of chicago
# 

sides = ["Far North", "Northwest", "North", "West", "Central",
    "South", "Southwest", "Far Southwest", "Far Southeast"]


def load(side):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points


bandwidths = [0.05, 0.15, 0.3, 1]

models = {}
for side in sides:
    grid, points = load(side)
    models[side] = {}
    for b in bandwidths:
        trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=b)
        trainer.data = points
        #try:
        #    models[side][b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)
        #except ValueError as ex:
            #print("Failed because {} for {}/{}".format(ex, side, b))
        #    print("Failed: {}/{}".format(side, b))
        models[side][b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=True)


fig, axes = plt.subplots(ncols=4, nrows=len(sides), figsize=(16,20))

for side, axe in zip(sides, axes):
    for ax, bw, s in zip(axe, models[side], [900,300,150,50]):
        model = models[side][bw]
        x = np.arange(s) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
        ax.set(xlabel="Days", ylabel="Rate", title="{}, h={} days".format(side, bw))
        ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
               model.bandwidth, color="None", edgecolor="black")

fig.tight_layout()











import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections
import open_cp.predictors
import scipy.stats
import sepp.kernels
import opencrimedata.chicago


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(trainer, data, model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    # Quickly compute marginals...
    opt = trainer._optimiser(model, data)
    x, w = opt.data_for_trigger_opt()
    prov = opt_fac._trigger_provider
    ker = open_cp.kernels.GaussianBase(x)
    ker.covariance_matrix = np.diag(prov._scale)
    ker.bandwidth = prov._h
    ker.weights = w
    xy_marginal = open_cp.kernels.marginalise_gaussian_kernel(ker, axis=0)
    tx_marginal = open_cp.kernels.marginalise_gaussian_kernel(ker, axis=2)
    t_marginal = open_cp.kernels.marginalise_gaussian_kernel(tx_marginal, axis=1)
    
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Fixed bandwidth
# 

tk_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(1, scale=[1,20,20])
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(20)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Stochastic EM
# 

tk_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(20, scale=[0.01,1,1])
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(20)
opt_fac = sepp.sepp_full.OptimiserSEMFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Grid based SEPP method(s)
# 
# Using KDE based trigger function.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import sepp.grid_nonparam


import open_cp.sources.chicago
import open_cp.geometry
import opencrimedata.chicago
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("South")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Train
# 

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(1))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


def plot(model, trigger_limit=20):
    fig, axes = plt.subplots(ncols=2, figsize=(16,6))

    ax = axes[0]
    pred = trainer.prediction_from_background(model)
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
    fig.colorbar(mappable, ax=ax)
    ax.set_title("Estimated background rate")

    ax = axes[1]
    x = np.linspace(0, trigger_limit, 200)
    x = x * (trainer.time_unit / np.timedelta64(1,"D"))
    ax.plot(x, model.trigger_func(x) * model.theta)
    ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel")
    
    return fig


fig = plot(model)


np.max(model.mu), np.min(model.mu)


# ## Investigate different initial conditions
# 
# It doesn't seem to really matter!  Which is good!
# 

cells, modeli = trainer.initial_model(datetime.datetime(2017,1,1))


def func(t):
    return np.exp(-t / 100)


model = sepp.grid_nonparam.KDEModel(modeli.mu, modeli.T, modeli.theta, func)
model


for _ in range(50):
    opt = trainer.provider.make_opt(model, cells)
    model = opt.optimised_model()


fig = plot(model)


# # Vary the bandwidth
# 

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.05))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


fig = plot(model, 15)


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.1))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


fig = plot(model, 15)


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.2))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


fig = plot(model, 15)


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.5))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
plot(model)
model


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(2))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
plot(model)
model


# # Use a variable bandwidth KDE
# 

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(5))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
_ = plot(model, 15)
model


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(15))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
_ = plot(model)
model


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(30))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
_ = plot(model)
model


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(50))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
_ = plot(model)
model


trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(100))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
_ = plot(model)
model


# # Summary plots for article
# 

bandwidths = [0.05, 0.1, 1, 2]

models = {}
for bw in bandwidths:
    trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(bw))
    trainer.data = points
    models[bw] = trainer.train(datetime.datetime(2017,1,1), iterations=50)


fig, axes = plt.subplots(ncols=4, figsize=(16,4))

for ax, bw in zip(axes, models):
    model = models[bw]

    x = np.linspace(0, 20, 200)
    x = x * (trainer.time_unit / np.timedelta64(1,"D"))
    ax.plot(x, model.trigger_func(x) * model.theta, color="black")
    ax.set(xlabel="Days", ylabel="Rate", title="Bandwidth of {} days".format(bw))
    
fig.tight_layout()
#fig.savefig("../grid_kde_by_bandwidth.pdf")


nearestns = [5, 10, 20, 50]

models = {}
for k in nearestns:
    trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(k))
    trainer.data = points
    models[k] = trainer.train(datetime.datetime(2017,1,1), iterations=50)


fig, axes = plt.subplots(ncols=4, figsize=(16,4))

for ax, bw in zip(axes, models):
    model = models[bw]

    x = np.linspace(0, 20, 200)
    x = x * (trainer.time_unit / np.timedelta64(1,"D"))
    ax.plot(x, model.trigger_func(x) * model.theta, color="black")
    ax.set(xlabel="Days", ylabel="Rate", title="Nearest neighbours: {}".format(bw))
    
fig.tight_layout()
#fig.savefig("../grid_kde_by_nn.pdf")


# # Other regions of chicago
# 
# Using other years of data is interesting...
# 

sides = ["Far North", "Northwest", "North", "West", "Central",
    "South", "Southwest", "Far Southwest", "Far Southeast"]


def load(side):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points


bandwidths = [0.05, 0.1, 1, 2]

models = {}
for side in sides:
    grid, points = load(side)

    models[side] = {}
    for bw in bandwidths:
        trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(bw))
        trainer.data = points
        models[side][bw] = trainer.train(datetime.datetime(2017,1,1), iterations=50)


fig, axes = plt.subplots(ncols=4, nrows=len(sides), figsize=(16,20))

for side, axe in zip(sides, axes):
    for ax, bw in zip(axe, models[side]):
        model = models[side][bw]

        x = np.linspace(0, 50, 200)
        x = x * (trainer.time_unit / np.timedelta64(1,"D"))
        ax.plot(x, model.trigger_func(x) * model.theta, color="black")
        ax.set(xlabel="Days", ylabel="Rate", title="{} / {} days".format(side, bw))
    
fig.tight_layout()








import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger.  Now with nearest neighbour estimators.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections
import open_cp.predictors
import scipy.stats
import sepp.kernels
import opencrimedata.chicago


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(trainer, data, model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    t_marginal = sepp.kernels.compute_t_marginal(model.trigger_kernel)
    xy_marginal = sepp.kernels.compute_space_marginal(model.trigger_kernel)
    
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Training
# 
# Nearest neighbour
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15, cutoff=1500)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=1500, time_size=20, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(trainer, data, model, space_size=1500, time_size=20, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Training
# 
# Nearest neighbour, with no cutoff (very slow...)
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=1500, time_size=20, space_floor=np.exp(-20))
#fig.savefig("../no_grid_nnkde_1.pdf")


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))
#fig.savefig("../no_grid_nnkde_1a.pdf")





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger.  Now with nearest neighbour estimators.
# 
# Here we use a fixed bandwidth KDE for the background.  It sort of works, but nothing very interesting.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections
import open_cp.predictors
import scipy.stats
import sepp.kernels
import opencrimedata.chicago


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(trainer, data, model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    t_marginal = sepp.kernels.compute_t_marginal(model.trigger_kernel)
    xy_marginal = sepp.kernels.compute_space_marginal(model.trigger_kernel)
    
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Training
# 
# Mix of nearest neighbour and fixed bandwidth
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15, cutoff=1500)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=1500, time_size=20, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(trainer, data, model, space_size=1000, time_size=10, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ### Larger background bandwidth
# 

tk_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(k=15, cutoff=1500)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(250)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)


fig = plot(trainer, data, model, space_size=1500, time_size=20, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# HERE!!!





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Fixed trigger
# 
# Use an exponential decay in time, and Gaussian in space, and see what background we can fit to the Chicago data.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_fixed
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.naive
import opencrimedata.chicago


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# # Train
# 

tk = sepp.sepp_fixed.ExpTimeKernel(0.2)
sk = sepp.sepp_fixed.GaussianSpaceKernel(50)
trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


fig, axes = plt.subplots(ncols=2, figsize=(16,5))

for ax in axes:
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)

ax = axes[0]
pred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
m = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
cb = fig.colorbar(m, ax=ax)
ax.set_title("SEPP prediction background")

naive = open_cp.naive.CountingGridKernel(grid.xsize, grid.ysize, grid.region())
naive.data = points
pred = naive.predict().renormalise()
ax = axes[1]
m = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
cb = fig.colorbar(m, ax=ax)
ax.set_title("Naive background rate")

None


# ## Variance in $\theta$
# 

all_models = []
for sigma, maxoi in zip([10, 25, 50, 100, 250], [200,30,20,20,6]):
    omegas_inv = np.linspace(1, maxoi, 50)
    models = []
    for omega_inv in omegas_inv:
        tk = sepp.sepp_fixed.ExpTimeKernel(1 / omega_inv)
        sk = sepp.sepp_fixed.GaussianSpaceKernel(sigma)
        trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
        trainer.data = points
        models.append( trainer.train(datetime.datetime(2017,1,1), iterations=20) )
    all_models.append((sigma, omegas_inv, models))


fig, axes = plt.subplots(ncols=5, figsize=(18,5))

for ax, (s, ois, models) in zip(axes, all_models):
    ax.plot(ois, [m.theta for m in models], color="black")
    ax.set(xlabel="$\omega^{-1}$")
    #ax.set(ylabel="$\\theta$")
    ax.set(title="$\sigma={}$".format(s))
fig.tight_layout()


fig.savefig("../fixed_grid_mod_1.pdf")





get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os, lzma, csv, bz2
import tilemapbase
import numpy as np

#datadir = os.path.join("/media", "disk", "Data")
datadir = os.path.join("..", "..", "..", "..", "Data")


# # The input data
# 
# https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
# 
# We use the `opencrimedata` package we have developed.
# 

import opencrimedata.chicago as chicago
import opencrimedata
print(opencrimedata.__version__)


filename = os.path.join(datadir, "chicago_all.csv.xz")
def gen():
    with lzma.open(filename, "rt", encoding="UTF8") as f:
        yield from chicago.load_only_with_point(f)
        
next(gen())


coords_wm = np.asarray([tilemapbase.project(*row.point) for row in gen() if row.crime_type=="BURGLARY"])


def gen_new():
    fn = os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz")
    with lzma.open(fn, "rt", encoding="UTF8") as f:
        yield from chicago.load_only_with_point(f)
        
coords_new_wm = np.asarray([tilemapbase.project(*row.point) for row in gen_new() if row.crime_type=="BURGLARY"])


fig, axes = plt.subplots(ncols=2, figsize=(17,8))

ex = tilemapbase.Extent.from_centre(0.25662, 0.3722, xsize=0.00005)
plotter = tilemapbase.Plotter(ex, tilemapbase.tiles.OSM, width=800)
for ax in axes:
    plotter.plot(ax)
axes[0].scatter(*coords_wm.T, marker="x", color="black", alpha=0.5)
axes[1].scatter(*coords_new_wm.T, marker="x", color="black", alpha=0.5)
None


fig.savefig("Chicago_overview.png", dpi=150)


# ## Chicago regions
# 
# We also use some code from [`open_cp`](https://github.com/QuantCrimAtLeeds/PredictCode) and the geographical data available at
# 
# https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6
# 
# Download in geojson format.
# 

import open_cp.sources.chicago as chicago
chicago.set_data_directory(datadir)


chicago.get_side("South")





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger, but with the trigger split between space and time.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections, os
import open_cp.predictors
import sepp.kernels
import scipy.stats
import opencrimedata.chicago


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * model.trigger_time_kernel(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], model.trigger_space_kernel, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Fixed bandwidth
# 

tk_time_prov = sepp.kernels.FixedBandwidthKernelProvider(1)
tk_space_prov = sepp.kernels.FixedBandwidthKernelProvider(30, cutoff=750)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(30)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)
model


fig = plot(model, space_size=250, time_size=10, space_floor=np.exp(-20))


backgrounds, trigger_deltas = trainer.sample_to_points(model, datetime.datetime(2017,1,1))


backgrounds.shape, trigger_deltas.shape


# ## Stochastic EM
# 

tk_time_prov = sepp.kernels.FixedBandwidthKernelProvider(1)
tk_space_prov = sepp.kernels.FixedBandwidthKernelProvider(30, cutoff=750)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(30)
opt_fac = sepp.sepp_full.Optimiser1SEMFactory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=25)
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=1000, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=1000, time_size=30, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Chicago data
# 
# Now with no Gaussian decay, and a histogram in time
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import sepp.sepp_grid_space
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels
import opencrimedata.chicago


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Train
# 

def plot(model, histlen):
    fig, axes = plt.subplots(ncols=2, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    ax = axes[1]
    x = np.arange(histlen) * model.bandwidth
    ax.bar(x + model.bandwidth/2, model.alpha_array[:len(x)] * model.theta / model.bandwidth,
           model.bandwidth, color="none", edgecolor="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    None


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=1, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 30)


# With this new data, we find that often the "edge corrected" algorithm is numerically unstable....
# 

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=0.5, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 40)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=0.1, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 40)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=0.8, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 40)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=1.2, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 35)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=1.3, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 35)


# ## Different $r_0$ value
# 
# This becomes computationally expensive...
# 

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=100, bandwidth=1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 10)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=100, bandwidth=0.1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 30)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=250, bandwidth=1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 6)


trainer = sepp.sepp_grid_space.Trainer3(grid, r0=250, bandwidth=0.1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


plot(model, 60)








import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # Chicago data
# 
# Now with no Gaussian decay
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import sepp.sepp_grid_space
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")


import open_cp.sources.chicago
import open_cp.geometry
import opencrimedata.chicago
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
_pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*_pred.mesh_data(), _pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Train
# 

trainer = sepp.sepp_grid_space.Trainer2(grid, 20)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


predictor = trainer.to_predictor(model)
predictor.data = trainer.data
pred1 = predictor.predict(datetime.datetime(2017,1,1))
pred2 = predictor.predict(datetime.datetime(2016,9,1))
pred3 = predictor.predict(datetime.datetime(2016,10,15))


fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16,10))

for ax in axes.flat:
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)

bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)

for ax, pred in zip(axes.flat, [bpred, pred1, pred2, pred3]):
    m = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)
fig.tight_layout()


# # How it changes with $r_0$
# 
# Can't start with $r_0$ being too small, or edge converges to "no triggering" which the base library raises as an exception (_maybe_ it shouldn't do this...  but it does, for now).
# 

r0_range = np.exp(np.linspace(1, np.log(100), 100))
models = {}
for r0 in r0_range:
    trainer = sepp.sepp_grid_space.Trainer2(grid, r0)
    trainer.data = points
    models[r0] = trainer.train(datetime.datetime(2017,1,1), iterations=50)


fig, axes = plt.subplots(ncols=2, figsize=(16,3))

axes[0].plot(r0_range, [models[r].theta for r in r0_range], color="black")
axes[0].set(title="theta")
axes[1].plot(r0_range, [1/models[r].omega for r in r0_range], color="black")
axes[1].set(title="1 / omega")

fig.tight_layout()
#fig.savefig("../varying_r0_no_g.pdf")








import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


# # With real data
# 
# Using the model with KDE estimates for background and trigger, but with the trigger split between space and time.
# 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections, os
import open_cp.predictors
import sepp.kernels
import scipy.stats
import opencrimedata.chicago


datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")


northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)


mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)


fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
_pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*_pred.mesh_data(), _pred.intensity_matrix, cmap="Greys", rasterized=True)
None


# ## Plotting functions
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * model.trigger_time_kernel(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], model.trigger_space_kernel, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# ## Fixed bandwidth
# 
# Doing: Re-run to make a better plot.  Probably won't put in the paper.
# 

tk_time_prov = sepp.kernels.FixedBandwidthKernelProvider(1)
tk_space_prov = sepp.kernels.FixedBandwidthKernelProvider(20, cutoff=1000)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model


fig = plot(model, space_size=900, time_size=20, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=900, time_size=20, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# ## Stochastic EM
# 

tk_time_prov = sepp.kernels.FixedBandwidthKernelProvider(1)
tk_space_prov = sepp.kernels.FixedBandwidthKernelProvider(20, cutoff=1000)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.Optimiser1SEMFactory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=25)
model


fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=1000, time_size=50, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model


fig = plot(model, space_size=1000, time_size=200, space_floor=np.exp(-20))


fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))


# HERE!! Check agree with 1st dataset.





# # Getting input data into the correct form
# 
# We use the same example data as from [PredictCode](https://github.com/QuantCrimAtLeeds/PredictCode/tree/master/quick_start).  This is extracted from the [Chicago data](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2) but simplified to make the following easier to explain.
# 
# This notebook takes a step by step approach.  See the notebook `example` for a more streamlined presentation.
# 

# ## Read the data in python
# 
# We need to:
# 
# - Understand the input data
# - Read the timestamps
# - Read the coordinates, and project them if necessary from Longitude and Latitude
# 
# We use `pandas` to make life a bit easier.
# 

import pandas as pd


frame = pd.read_csv("example.csv")
frame.head()


# ### Read the timestamps
# 
# We need to read the `date` field into a python `datetime` object.  We use the library [`python-dateutil`](https://dateutil.readthedocs.io/en/stable/) which makes converting timestamps really easy.
# 
# We also continue to use pandas, rather than reading the csv file directly.  We select the `Date` column with
# 
#     frame.Date
#     
# and then use `map` which applies a function to each row of that column.  The function we use is `dateutil.parser.parse` which takes the string timestamp, and converts it into a Python object.
# 

import dateutil.parser

timestamps = frame.Date.map(dateutil.parser.parse)
timestamps[:5]


# ## Project the coordinates
# 
# We use the [`pyproj`](https://pypi.python.org/pypi/pyproj) package which supports standard projection methods.  We use [EPSG:2790](http://spatialreference.org/ref/epsg/2790/) for Chicago.
# 

import pyproj
proj = pyproj.Proj({"init":"EPSG:2790"})


# We cannot use `map` now, as we need to combine the two columns of Longitude and Latitude.  Fortunately, we can parse lists to a `Proj` object.  The code
# 
#     frame.Longitude.values
#     
# selects the `Longitude` column, and then selects just the values.  (We sometimes need to do this: if we don't, `pyproj` gets confused).
# 

xcoords, ycoords = proj(frame.Longitude.values, frame.Latitude.values)
xcoords[:5], ycoords[:5]


# ## Combine
# 
# `open_cp` has its own data container which we build here.  We print the number of data points, and the time range.  They look okay.
# 

import open_cp.data

points = open_cp.data.TimedPoints.from_coords(timestamps, xcoords, ycoords)


points.number_data_points


points.time_range


# # Load some geometry and form a grid
# 
# We now load the outline of the South side of Chicago, using `geopandas` for convenience.
# 

import geopandas as gpd


frame = gpd.read_file("SouthSide")
frame


geo = list(frame.geometry)[0]


# ## Visualise
# 
# We use the package `descartes` and the usual `matplotlib` library to view the outline, and plot the points over it.
# 

import descartes
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect(1)
ax.add_patch(descartes.PolygonPatch(geo, fc="none"))
ax.scatter(xcoords, ycoords, marker="x", color="black", linewidth=1)


# ## Make a grid
# 
# For some of the SEPP methods, and for making any prediction, we need a grid.  `open_cp` provides some ways to do this.
# 

import open_cp.data
import open_cp.geometry

grid = open_cp.data.Grid(xsize=150, ysize=150, xoffset=0, yoffset=0)
grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)


import open_cp.plot
import matplotlib.collections

fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect(1)
lc = matplotlib.collections.LineCollection(open_cp.plot.lines_from_grid(grid), color="black", linewidth=1)
ax.add_collection(lc)
ax.scatter(xcoords, ycoords, marker="x", color="black", linewidth=0.5)


# # Run a simple SEPP method on this data
# 
# With this example data, we don't expect to learn anything...
# 

import sepp.sepp_grid
import datetime


trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1))


model


fig, ax = plt.subplots(figsize=(10,6))

ax.add_patch(descartes.PolygonPatch(geo, fc="none"))
ax.set_aspect(1)
pred = trainer.prediction_from_background(model)
mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=ax)
ax.set_title("Estimated background rate")
None


# ## And an example of a non-grid based method
# 

import sepp.kernels
import sepp.sepp_full
import numpy as np
import scipy.stats


tk_time_prov = sepp.kernels.FixedBandwidthKernelProvider(1)
tk_space_prov = sepp.kernels.FixedBandwidthKernelProvider(20, cutoff=750)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=5)
model


# Sorry, a load of code to produce some interesting plots...
# 

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(geo, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * model.trigger_time_kernel(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], model.trigger_space_kernel, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas


# We plot the background rate, and the predicted trigger components in time and space
# 

fig = plot(model, space_size=750, time_size=20, space_floor=np.exp(-20))


# It can be quite hard to think what the above means.  Instead, we take a "sample" of the process, and decide (probabilisticly) which events are background, and which are "triggered".
# 

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))





get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg, scipy.stats
import collections


# # Simulate a Hawkes process
# 

class Hawkes():
    def __init__(self):
        self.mu = 0.1
        self.theta = 0.5
        self.omega = 1

    def intensity(self, past, t):
        past = np.asarray(past)
        past = past[past < t]
        return self.mu + self.theta * self.omega * np.sum(np.exp(-self.omega * (t-past)))
        
    def next_event(self, past_events, T):
        t = T
        while True:
            max_intensity = self.intensity(past_events, t) + self.theta
            td = np.random.exponential(scale=1/max_intensity)
            if np.random.random() * max_intensity <= self.intensity(past_events, t+td):
                return t+td
            t += td
            
    def simulate(self, previous_events, Tstart, Tend):
        points = []
        all_points = list(previous_events)
        while True:
            t = points[-1] if len(points) > 0 else Tstart
            points.append( self.next_event(points, t) )
            all_points.append(points[-1])
            if points[-1] > Tend:
                break
        return np.asarray(points)


hawkes = Hawkes()
points = hawkes.simulate([], 0, 10000)


fig, ax = plt.subplots(figsize=(16,6))

x = np.linspace(0, 105, 2000)
y = [hawkes.intensity(points, t) for t in x]
ax.plot(x, y)
ax.set(xlim=[0,100])

ax.scatter(points, np.random.random(len(points))*.01, marker="x", color="black")


# ## Residual check
# 

# Exact integrate
ppoints = []
pts = points
for i, pt in enumerate(pts):
    intl = hawkes.mu * pt + hawkes.theta * np.sum(1 - np.exp(-hawkes.omega * (pt-pts[:i])))
    ppoints.append(intl)
ppoints = np.asarray(ppoints)


diffs = ppoints[1:] - ppoints[:-1]
diffs = np.sort(diffs)
# Should be from Exp(1)
diffs


# Exponential has density $f(x) = e^{-x}$ and so cdf $F(x) = 1 - e^{-x}$ and so $F^{-1}(y) = -\log(1-y)$.
# 

fig, ax = plt.subplots(figsize=(7,7))

x = np.linspace(0, 1, len(diffs)+1)
x = (x[:-1] + x[1:]) / 2
y = - np.log(1-x)
ax.scatter(diffs, y, marker="x", color="black", linewidth=1)
m=8
ax.plot([0,m], [0, m])
ax.set(xlabel="Differences from process", ylabel="Exp(1) theory")
None


# # Predict
# 
# Simulate 200 "days" of process for various $\mu, \theta, \omega$.  For each $T=100,101,\cdots,199$ we:
# - Look at the process up to $T$
# - Repeatedly simulate the next day
# - Compute the distribution of the number of points we get.
# 

def simulate(mu=2, omega=5, theta=0.5):
    hawkes = Hawkes()
    hawkes.mu = mu
    hawkes.omega = omega
    hawkes.theta = theta
    return hawkes, hawkes.simulate([], 0, 1000)

def plot_simulation(points, hawkes, xrange=(50,100)):
    fig, ax = plt.subplots(figsize=(16,5))

    x = np.linspace(50, 100, 2000)
    y = [hawkes.intensity(points, t) for t in x]
    ax.plot(x, y, color="grey")

    ax.scatter(points, np.random.random(len(points))*.01, marker="x", color="black")
    ax.set(xlim=xrange)

    return fig

Result = collections.namedtuple("Result", "dist actual int intint")

def predict(points, hawkes, Trange=(100,1000), trials=100):
    results = []
    for T in range(*Trange):
        counts = []
        for _ in range(trials):
            current_points = points[points<T]
            new_points = hawkes.simulate(current_points, T, T+1)
            new_points = new_points[new_points < T+1]
            counts.append( len(new_points) )
        intint = np.mean([hawkes.intensity(current_points, t) for t in np.linspace(T,T+1,100)[1:]])
        result = Result(collections.Counter(counts),
                        len(points[(points>=T)&(points<T+1)]),
                        hawkes.intensity(current_points, T+1/2),
                        intint)
        results.append(result)
    return results


hawkes, points = simulate(mu=2, omega=1, theta=0.5)
fig = plot_simulation(points, hawkes)
fig.set_size_inches(16, 3)
fig.axes[0].set_title("Sample simulation with $\mu=2, \omega=1, \\theta=0.5$")
fig.savefig("../hawkes_sample.pdf")


results1 = predict(points, hawkes)

hawkes, points = simulate(mu=1, omega=1, theta=0.2)
results2 = predict(points, hawkes)

hawkes, points = simulate(mu=1, omega=50, theta=0.5)
results3 = predict(points, hawkes)


for results in [results1, results2, results3]:
    print(np.percentile([r.int - r.intint for r in results], 2),
     np.percentile([r.int - r.intint for r in results], 50),
     np.percentile([r.int - r.intint for r in results], 98))


def plot_bar(ax, x, y, cutoff=0.0001):
    ax.bar(x, y, color="grey")
    i = x.index(0)
    ax.bar([0], y[i], color="black")
    ax.set(xlabel="estimated - actual", ylabel="Relative count")
    
    y = np.asarray(y)
    m = y < (cutoff * np.sum(y))
    want = np.asarray(x)[~m]
    ax.set(xlim=[np.min(want) - 1/2, np.max(want) + 1/2])
    
def plot_diffs(ax, diffs):
    di = np.sort(diffs)
    low, high = np.percentile(di, 2), np.percentile(di, 98)
    low, high = int(np.floor(low))-2, int(np.ceil(high))+2
    ax.hist(di, color="grey", bins=np.arange(low,high+1))
    ax.set(xlabel="estimated - actual", ylabel="Relative count")
    ax.set(xlim=[low, high])


fig, axesall = plt.subplots(nrows=3, ncols=3, figsize=(16,10))

for axes, results in zip(axesall, [results1, results2, results3]):
    plot_diffs(axes[0], [r.int - r.actual for r in results])
    plot_diffs(axes[1], [r.intint - r.actual for r in results])

    error_counts = collections.defaultdict(int)
    for result in results:
        for count, num in result.dist.items():
            error_counts[count - result.actual] += num

    x = list(error_counts)
    y = list(error_counts.values())
    plot_bar(axes[2], x, y)

    fig.tight_layout()


fig, axesall = plt.subplots(nrows=2, ncols=3, figsize=(16,7))

for axes, results in zip(axesall.T, [results1, results2, results3]):
    plot_diffs(axes[0], [r.intint - r.actual for r in results])

    error_counts = collections.defaultdict(int)
    for result in results:
        for count, num in result.dist.items():
            error_counts[count - result.actual] += num

    x = list(error_counts)
    y = list(error_counts.values())
    plot_bar(axes[1], x, y)
    
    axes[0].set_title("Estimate by mean intensity")
    axes[1].set_title("Estimate by simulation")

    fig.tight_layout()


fig, axesall = plt.subplots(nrows=2, ncols=3, figsize=(16,7))

for axes, results in zip(axesall.T, [results1, results2, results3]):
    plot_diffs(axes[0], [r.intint - r.actual for r in results])
    diffs = []
    for result in results:
        d = sum((count - result.actual)*num for count, num in result.dist.items())
        d /= sum(num for count, num in result.dist.items())
        diffs.append(d)
    plot_diffs(axes[1], diffs)
    
    axes[0].set_title("Estimate by mean intensity")
    axes[1].set_title("Estimate by simulation")
    x11, x12 = axes[0].get_xlim()
    x21, x22 = axes[1].get_xlim()
    for ax in axes:
        ax.set(xlim=[min(x11,x21), max(x12,x22)-1])

    fig.tight_layout()


fig.savefig("../hawkes_predictions.pdf")






get_ipython().magic('pylab inline')


import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import random
import sklearn as sk
from time import time
import pickle


from sklearn.ensemble import RandomForestClassifier


from utils import calc_cc, calc_dist, calc_emd, number_of_fake_edges, read_mtx_graph


from rand_pert import rand_pertub, link_mirage, smart_link_anon, smart_pertub, calc_features


get_ipython().run_cell_magic('time', '', 'from_file = True\nfilename = "data/p2p-Gnutella08/p2p-Gnutella08.mtx"\n#filename = "data/ca-AstroPh/ca-AstroPh.mtx"\nfilename = "data/email-Enron/email-Enron.mtx"\nfilename = "data/yeast/yeast.mtx"\nfilename = "data/USpowerGrid/USpowerGrid.mtx"\nfilename = "data/G22/G22.mtx"\n\nif from_file:\n    g = read_mtx_graph(filename)\nelse:\n    #g = nx.erdos_renyi_graph(n=1000, p=0.1)\n    g = nx.powerlaw_cluster_graph(n=4000, m=30, p=0.1)\n')


print(g.number_of_edges())
print(g.number_of_nodes())


print(g.number_of_edges())
print(g.number_of_nodes())





# ### Split the graph
# 

from attack import *


alpha = 0.25


get_ipython().run_cell_magic('time', '', 'G1, G2 = edge_split(g, alpha)')


M = 10
t = 3


get_ipython().run_cell_magic('time', '', 'f = calc_features(G1)\nG_aux = smart_link_anon(G1, M, t, 0.5, f)\nG_aux.add_nodes_from(G1.nodes())')


get_ipython().run_cell_magic('time', '', 'f = calc_features(G2)\nG_san = smart_link_anon(G2, M, t, 0.5, f)\nG_san.add_nodes_from(G2.nodes())')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G2.number_of_edges(), " Now edges ", G_san.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G2, G_san))\nprint("% of distortion")\nprint(calc_dist(G2, G_san))\nprint(" EMD of deg distibution")\nprint(calc_emd(G2, G_san))\nprint(" CC diff")\nprint(calc_cc(G2, G_san))')


get_ipython().run_cell_magic('time', '', 'cache = {}\nf = calc_features(G1)\nG_aux = smart_pertub(G1, M, t, 0.5, f, cache)\nG_aux.add_nodes_from(G1.nodes())')


get_ipython().run_cell_magic('time', '', 'cache = {}\nf = calc_features(G2)\nG_san = smart_pertub(G2, M, t, 0.5, f, cache)\nG_san.add_nodes_from(G2.nodes())')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G2.number_of_edges(), " Now edges ", G_san.number_of_edges())\nprint("% of fake edges")\nprint(number_of_fake_edges(G2, G_san))\nprint("% of distortion")\nprint(calc_dist(G2, G_san))\nprint(" EMD of deg distibution")\nprint(calc_emd(G2, G_san))\nprint(" CC diff")\nprint(calc_cc(G2, G_san))')


get_ipython().run_cell_magic('time', '', 'G_aux = link_mirage(G1, M, t, 0.5)\nG_aux.add_nodes_from(G1.nodes())')


get_ipython().run_cell_magic('time', '', 'G_san = link_mirage(G2, M, t, 0.5)\nG_san.add_nodes_from(G2.nodes())')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\n\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint(number_of_fake_edges(G2, G_san))\n\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(calc_dist(G2, G_san))\n\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(calc_emd(G2, G_san))\n\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))\nprint(calc_cc(G2, G_san))\n')


get_ipython().run_cell_magic('time', '', 'print("Originial edges ", G1.number_of_edges(), " Now edges ", G_aux.number_of_edges())\n\nprint("% of fake edges")\nprint(number_of_fake_edges(G1, G_aux))\nprint(number_of_fake_edges(G2, G_san))\n\nprint("% of distortion")\nprint(calc_dist(G1, G_aux))\nprint(calc_dist(G2, G_san))\n\nprint(" EMD of deg distibution")\nprint(calc_emd(G1, G_aux))\nprint(calc_emd(G2, G_san))\n\nprint(" CC diff")\nprint(calc_cc(G1, G_aux))\nprint(calc_cc(G2, G_san))\n')


X_naive, Y_naive = gen_trainset(G1, G2, alpha, "data/train_naive.dump")


X, Y = gen_trainset(G_aux, G_san, alpha, "data/train.dump")


get_ipython().run_cell_magic('time', '', 'forest = RandomForestClassifier(n_estimators = 400, n_jobs=8)\nforest.fit(X, Y)')


X_t, Y_t = gen_testset(G_aux, G_san, "data/test_test.dump")


X_t_naive, Y_t_naive = gen_testset(G1, G2, "data/test_test_naive.dump")


get_ipython().run_cell_magic('time', '', '\nforest = RandomForestClassifier(n_estimators = 400, n_jobs=8)\nprint("Naive:")\nforest.fit(X_naive, Y_naive)\nprint(forest.score(X_t_naive, Y_t_naive))')





from sklearn.metrics import roc_curve, auc
y_score = forest.predict_proba(X_t_naive)
y_true = Y_t_naive
fpr, tpr, thresholds = roc_curve(y_true, y_score[:, 1])
roc_auc = auc(fpr, tpr)


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='GS, AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


get_ipython().run_cell_magic('time', '', 'forest = RandomForestClassifier(n_estimators = 400, n_jobs=8)\nforest.fit(X, Y)\nprint("Anonymization")\nprint(forest.score(X_t, Y_t))')


get_ipython().run_cell_magic('time', '', 'forest = RandomForestClassifier(n_estimators = 400, n_jobs=8)\nforest.fit(X, Y)\nprint("Anonymization")\nprint(forest.score(X_t, Y_t))')


get_ipython().run_cell_magic('time', '', 'forest = RandomForestClassifier(n_estimators = 400, n_jobs=8)\nforest.fit(X, Y)\nprint("Anonymization")\nprint(forest.score(X_t, Y_t))')


get_ipython().run_cell_magic('time', '', '# Smart link 0.5\nforest = RandomForestClassifier(n_estimators = 400, n_jobs=8)\nforest.fit(X, Y)\nprint("Anonymization")\nprint(forest.score(X_t, Y_t))')








from sklearn.metrics import roc_curve, auc
y_score = forest.predict_proba(X_t)
y_true = Y_t
fpr, tpr, thresholds = roc_curve(y_true, y_score[:, 1])
roc_auc = auc(fpr, tpr)


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='GS, AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Smart pertubation
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='GS, AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Smart link 0.8
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='GS, AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Smart link 0.5
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='GS, AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()






# # A python node analysis jupyter notebook
# 

# **Synopsis:** This notebook will read in a spice like circuit netlist file and compute the node equations. The code follows Erik Cheever's Analysis of  Resistive Circuits [page](http://www.swarthmore.edu/NatSci/echeeve1/Ref/mna/MNA1.html) to generate modified nodal equations. I somewhat followed his matlab file.
# 
# **Description:**
# 

# ```
# Date started: April 17, 2017
# file name: node analysis.ipynb
# Requires: Python version 3 or higher
# Author: Tony
# 
# Revision History
# 7/1/2015: Ver 1 - coding started, derived from network.c code
# 8/18/2017
# changed approach, now implementing a modified nodal analysis
# 8/19/2017
# Wrote some code to generate symbolic matrices, works ok,
# so heading down the sympy path. Basic debugging finished,
# but still need to verify some circuits using Ls and Cs.
# 8/30/2017
# Started to add code for op amps
# 9/1/2017
# code added to process op ams, not debugged yet
# started a task list in CoCalc for this project
# Still to do:
# Build some test circuits for debugging the code.
# test the circuits in LTSpice compare solutions
# Add controlled sources
# Add coupled inductors
# ```
# 

import os
from sympy import *
import numpy as np
import pandas as pd
init_printing()


# initialize some variables, count the types of elements
num_rlc = 0 # number of passive elements
num_v = 0    # number of independent voltage sources
num_i = 0    # number of independent current sources
num_opamps = 0   # number of op amps
num_vcvs = 0     # number of controlled sources of various types
num_vccs = 0
num_cccs = 0
num_ccvs = 0
num_cpld_ind = 0 # number of coupled inductors


# ##### open file and preprocess it
# - remove blank lines and comments
# - convert first letter of element name to upper case
# - removes extra spaces between entries
# - count number of entries on each line, make sure the count is correct
# 

fn = 'example2'
fd1 = open(fn+'.net','r')
content = fd1.readlines()
content = [x.strip() for x in content]  #remove leading and trailing white space
# remove empty lines
while '' in content:
    content.pop(content.index(''))

# remove comment lines, these start with a asterisk *
content = [n for n in content if not n.startswith('*')]
# converts 1st letter to upper case
#content = [x.upper() for x in content] <- this converts all to upper case
content = [x.capitalize() for x in content]
# removes extra spaces between entries
content = [' '.join(x.split()) for x in content]


branch_cnt = len(content)
# check number of entries on each line
for i in range(branch_cnt):
    x = content[i][0]
    tk_cnt = len(content[i].split())

    if (x == 'R') or (x == 'L') or (x == 'C'):
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_rlc += 1
    elif x == 'V':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_v += 1
    elif x == 'I':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_i += 1
    elif x == 'O':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_opamps += 1
    elif x == 'E':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vcvs += 1
    elif x == 'G':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vccs += 1
    elif x == 'F':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_cccs += 1
    elif x == 'H':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_ccvs += 1
    elif x == 'K':
        if (tk_cnt != 4):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_cpld_ind += 1
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))


# ##### parser
# - puts branch elements into structure
# - counts number of nodes
# 

# build the pandas data frame
count = []        # data frame index
element = []      # type of element
p_node = []       # positive node
n_node = []       # neg node
cp_node = []      # controlling positive node of branch
cn_node = []      # controlling negitive node of branch
v_out = []        # op amp output node
value = []        # value of element or voltage
v_name = []       # voltage source through which the controlling current flows
l_name1 = []      # name of coupled inductor 1
l_name2 = []      # name of coupled inductor 2

df = pd.DataFrame(index=count, columns=['element','p node','n node','cp node','cn node',
    'v out','value','v name','l_name1','l_name2'])


# ##### functions to load branch elements into data frame
# 

# loads voltage or current sources into branch structure
def indep_source(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'value'] = float(tk[3])

# loads passive elements into branch structure
def rlc_element(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'value'] = float(tk[3])

'''
loads multi-terminal sub-networks
into branch structure
Types:
E - VCVS
G - VCCS
F - CCCS
H - CCVS
not implemented yet:
K - Coupled inductors
O - Op Amps
'''
def opamp_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'v out'] = int(tk[3])

def vccs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'cp node'] = int(tk[3])
    df.loc[br_nu,'cn node'] = int(tk[4])
    df.loc[br_nu,'value'] = float(tk[5])

def vcvs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'cp node'] = int(tk[3])
    df.loc[br_nu,'cn node'] = int(tk[4])
    df.loc[br_nu,'value'] = float(tk[5])

def cccs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'v name'] = tk[3]
    df.loc[br_nu,'value'] = float(tk[4])

def ccvs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'v name'] = tk[3]
    df.loc[br_nu,'value'] = float(tk[4])

def cpld_ind_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'l name1'] = tk[1]
    df.loc[br_nu,'l name2'] = tk[2]
    df.loc[br_nu,'value'] = float(tk[3])


# function to scan df and get largest node number
def count_nodes():
    # need to check that nodes are consecutive
    # fill array with node numbers
    p = np.zeros(branch_cnt+1)
    for i in range(branch_cnt-1):
        p[df['p node'][i]] = df['p node'][i]
        p[df['n node'][i]] = df['n node'][i]

    # find the largest node number
    if df['n node'].max() > df['p node'].max():
        largest = df['n node'].max()
    else:
        largest =  df['p node'].max()

        largest = int(largest)
    # check for unfilled elements, skip node 0
    for i in range(1,largest):
        if p[i] == 0:
            print("nodes not in continuous order");

    return largest


# load branches into data frame
for i in range(branch_cnt):
    x = content[i][0]

    if (x == 'R') or (x == 'L') or (x == 'C'):
        rlc_element(i)
    elif (x == 'V') or (x == 'I'):
        indep_source(i)
    elif x == 'O':
        opamp_sub_network(i)
    elif x == 'E':
        vcvs_sub_network(i)
    elif x == 'G':
        vccs_sub_network(i)
    elif x == 'F':
        cccs_sub_network(i)
    elif x == 'H':
        ccvs_sub_network(i)
    elif x == 'K':
        cpld_ind_sub_network(i)
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))

# count number of nodes
num_nodes = count_nodes()


# print a report
print('Net list report')
print('number of branches: {:d}'.format(branch_cnt))
print('number of nodes: {:d}'.format(num_nodes))
print('number of passive components: {:d}'.format(num_rlc))
print('number of independent voltage sources: {:d}'.format(num_v))
print('number of independent current sources: {:d}'.format(num_i))
print('number of op amps: {:d}'.format(num_opamps))

# not implemented yet
print('\nNot implemented yet')
print('number of E - VCVS: {:d}'.format(num_vcvs))
print('number of G - VCCS: {:d}'.format(num_vccs))
print('number of F - CCCS: {:d}'.format(num_cccs))
print('number of F - CCCS: {:d}'.format(num_ccvs))
print('number of K - Coupled inductors: {:d}'.format(num_cpld_ind))


# store the data frame as a pickle file
df.to_pickle(fn+'.pkl')


# initialize some symbolic matrix with zeros
# A is formed by [[G, C] [B, D]]
# Z = [I,E]
# X = [V, J]
V = zeros(num_nodes,1)
I = zeros(num_nodes,1)
G = zeros(num_nodes,num_nodes)
s = Symbol('s')

if (num_v+num_opamps) != 0:
    B = zeros(num_nodes,num_v+num_opamps)
    C = zeros(num_v+num_opamps,num_nodes)
    D = zeros(num_v+num_opamps,num_v+num_opamps)
    E = zeros(num_v+num_opamps,1)
    J = zeros(num_v+num_opamps,1)


# ##### G matrix
# The G matrix is n by n and is determined by the interconnections between the passive circuit elements (RLC's).  The G matrix is an nxn matrix formed in two steps:
# 1. Each element in the diagonal matrix is equal to the sum of the conductance (one over the resistance) of each element connected to the corresponding node.  So the first diagonal element is the sum of conductances connected to node 1, the second diagonal element is the sum of conductances connected to node 2, and so on.
# 2. The off diagonal elements are the negative conductance of the element connected to the pair of corresponding node.  Therefore a resistor between nodes 1 and 2 goes into the G matrix at location (1,2) and locations (2,1).
# 

# G matrix
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'R':
        g = 1/sympify(df.loc[i,'element'])
    if x == 'L':
        g = 1/(s/sympify(df.loc[i,'element']))
    if x == 'C':
        g = sympify(df.loc[i,'element'])*s

    if (x == 'R') or (x == 'L') or (x == 'C'):
        # If neither side of the element is connected to ground
        # then subtract it from appropriate location in matrix.
        if (n1 != 0) and (n2 != 0):
            G[n1-1,n2-1] += -g
            G[n2-1,n1-1] += -g

        # If node 1 is connected to ground, add element to diagonal of matrix
        if n1 != 0:
            G[n1-1,n1-1] += g

        # same for for node 2
        if n2 != 0:
            G[n2-1,n2-1] += g

G  # display the G matrix


# ##### I matrix
# The I matrix is an n by 1 matrix with each element of the matrix corresponding to a particular node.  The value of each element of I is determined by the sum of current sources into the corresponding node.  If there are no current sources connected to the node, the value is zero.
# 

# generate the I matrix
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'I':
        g = sympify(df.loc[i,'element'])
        # sum the current into each node
        if n1 != 0:
            I[n1-1] += g
        if n2 != 0:
            I[n2-1] -= g

I  # display the I matrix


# ##### V matrix
# The V matrixis an nx1 matrix formed of the node voltages.  Each element in V corresponds to the voltage at the equivalent node in the circuit
# 

# generate the V matrix
for i in range(num_nodes):
    V[i] = sympify('v{:d}'.format(i+1))

V  # display the V matrix


# ##### B Matrix
# Rules for making the B matrix
# The B matrix is an nxm matrix with only 0, 1 and -1 elements.  Each location in the matrix corresponds to a particular voltage source (first dimension) or a node (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a -1.  Otherwise, elements of the B matrix are zero.
# 

# generate the B Matrix
# loop through all the branches and process independent voltage sources
sn = 0   # count source number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(branch_cnt):
    n_vout = df.loc[i,'v out'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        B[n_vout,oam+num_v] = 1
        oan += 1   # increment op amp count

B   # display the B matrix


# ##### J matrix
# The is an mx1 matrix, with one entry for the current through each voltage source.
# 

# The J matrix is an mx1 matrix, with one entry for the current through each voltage source.
sn = 0   # count source number
oan = 0   #count op amp number
for i in range(branch_cnt):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        J[sn] = sympify('I_{:s}'.format(df.loc[i,'element']))
        sn += 1
    if x == 'O':  # this needs to be checked <---- needs debugging
        J[oan+num_v_ind] = sympify('I_{:s}'.format(df.loc[i,'element']))
        oan += 1

J  # diplay the J matrix


# ##### C matrix
# The C matrix is an mxn matrix with only 0, 1 and -1 elements.  Each location in the matrix corresponds to a particular node (first dimension) or voltage source (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a -1.  Otherwise, elements of the C matrix are zero.
# 

# generate the C matrix
sn = 0   # count source number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    n_vout = df.loc[i,'v out'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        if n1 != 0:
            C[i+numV,n1-1] = 1
        if n2 != 0:
            C[i+numV,n2-1] = -1
        oan += 1  # increment op amp count

C   # display the C matrix


# ##### D matrix
# The D matrix is an mxm matrix that is composed entirely of zeros.  (It can be non-zero if dependent sources are considered.)
# 

# display the The D matrix
D


# ##### E matrix
# The E matrix is mx1 and holds the values of the independent voltage sources.
# 

# generate the E matrix
sn = 0   # count source number
for i in range(branch_cnt):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        E[sn] = sympify(df.loc[i,'element'])
        sn += 1

E   # display the E matrix


# ##### Z matrix
# The Z matrix holds the independent voltage and current sources and is the combination of 2 smaller matrices I and E.  The Z matrix is (m+n) by 1, n is the number of nodes, and m is the number of independent voltage sources.  The I matrix is n by 1 and contains the sum of the currents through the passive elements into the corresponding node (either zero, or the sum of independent current sources). The E matrix is m by 1 and holds the values of the independent voltage sources.
# 

Z = I[:] + E[:]
Z  # display the Z matrix


# ##### X matrix
# The X matrix is an (n+m) by 1 vector that holds the unknown quantities (node voltages and the currents through the independent voltage sources). The top n elements are the n node voltages. The bottom m elements represent the currents through the m independent voltage sources in the circuit. The V matrix is n by 1 and holds the unknown voltages.  The J matrix is m by 1 and holds the unknown currents through the voltage sources
# 

X = V[:] + J[:]
X  # display the X matrix


# ##### A matrix
# The A matrix is (m+n) by (m+n) and will be developed as the combination of 4 smaller matrices, G, B, C, and D.
# 

n = num_nodes
m = num_v
A = zeros(m+n,m+n)
for i in range(n):
    for j in range(n):
        A[i,j] = G[i,j]

if num_v+num_opamps > 1:
    for i in range(n):
        for j in range(m):
            A[i,n+j] = B[i,j]
            A[n+j,i] = C[j,i]
else:
    for i in range(n):
        A[i,n] = B[i]
        A[n,i] = C[i]

A  # display the A matrix


# generate the node equations
n = num_nodes
m = num_v
eq1 = 0
equ = zeros(m+n,1)
for i in range(n+m):
    for j in range(n+m):
        eq1 += A[j,i]*X[j]
    equ[i] = Eq(eq1,Z[i])
    eq1 = 0

equ   # display the equations


# Declare some symbols to solve the node equations
R1, R2, R3 = symbols('R1 R2 R3')
v1, v2, v3 = symbols('v1 v2 v3')
Vb, Is, IVb= symbols('Vb Is IVb')


# enter the element values
equ1a = equ.subs({R1:5})
equ1a = equ1a.subs({R2:3})
equ1a = equ1a.subs({R3:10})

equ1a = equ1a.subs({Vb:30})
equ1a = equ1a.subs({Is:2})

equ1a  # display the equations


equ1a.row_del(0)


equ1a


solve(equ1a,[v1,v2])


# try solving for the branch currernts
branch_cnt





# # A python node analysis jupyter notebook
# 

# **Abstract:** This notebook will read in a spice like circuit netlist file and compute the network equations. These equations can then be copied to a different notebook where the node voltages can be solved using sympy or numpy.
# 
# **Description:** This node analysis code started as a translation from some C code to generate a nodal admittance matrix that I had written in 1988.  The original C code worked well and calculated numeric solutions.  I then started writing some C code to generate the matrices with symbolic values and then intended to use LISP to symbolically solve the equations.  I didn’t get too far with this effort.  The LISP code would generate huge symbolic strings with no simplification.  The output was a big pile of trash that was not in the least bit useful or decipherable.
# 
# In 2014, I started to use python for my little coding projects and engineering calculations.  There are some nice python libraries for numeric and symbolic calculations (such as numpy and sympy), so I decided to try writing a python script to generate the node equations based on the old C code I had written many years before.  Part way into this project I discovered that there is a new nodal analysis technique being taught today in engineering school called the modified nodal analysis (1,2).  The modified nodal analysis provides an algorithmic method for generating systems of independent equations for linear circuit analysis.  Some of my younger colleagues at work were taught this method, but I never heard of it until a short time ago.  These days, I never really analyze a circuit by hand, unless it’s so simple that you can almost do it by inspection.  Most problems that an electrical engineer encounters on the job are complex enough that they use computers to analyze the circuits.  LTspice is the version of spice that I use, since it’s free and does a good job converging when analyzing switching circuits.
# 
# The code follows Erik Cheever's Analysis of  Resistive Circuits [page](http://www.swarthmore.edu/NatSci/echeeve1/Ref/mna/MNA1.html) to generate modified nodal equations. I somewhat followed his matlab file for resistors, capacitors, opamps and independent sources.  The preprocessor and parser code was converted from my old C code.  The use of panda for a data frame is new and sympy is used to do the math.
# 
# After doing some verification testing with inductors and capacitors, it seems that inductors are not being treated correctly.  According to some research, the inductor stamp affects the B,C and D arrays.  Erik Cheever's code puts inductors into the G matrix as 1/s/L.  LTspice results are different than the python code.  Capacitors seem to work OK.
# 
# Reference:
# 1. The modified nodal approach to network analysis, Chung-Wen Ho, A. Ruehli, P. Brennan, IEEE Transactions on Circuits and Systems ( Volume: 22, Issue: 6, Jun 1975 )
# 2. https://en.wikipedia.org/wiki/Modified_nodal_analysis
# 3. ECE 570 Session 3, Computer Aided Engineering for Integrated Circuits, http://www2.engr.arizona.edu/~ece570/session3.pdf
# 
# Some notes from reference 1:
# Capacitances and inductances are considered only in the time domain and their contributions, shown in Table I, are obtained by applying finite differencing methods to their branch relations.
# 

# ```
# Date started: April 17, 2017
# file name: node analysis.ipynb
# Requires: Python version 3 or higher and a jupyter notebook
# Author: Tony
# 
# Revision History
# 7/1/2015: Ver 1 - coding started, derived from network.c code
# 8/18/2017
# changed approach, now implementing a modified nodal analysis
# 8/19/2017
# Wrote some code to generate symbolic matrices, works ok,
# so heading down the sympy path. Basic debugging finished,
# but still need to verify some circuits using Ls and Cs.
# 8/30/2017
# Started to add code for op amps
# 9/1/2017
# Code added to process op amps
# 9/3/2017
# Added code to remove spice directives.
# Fixed orientation of current sources in I matrix.
# N2 is the arrow end of the current source.
# 9/5/2017
# After doing some verification testing with inductors and capacitors,
# it seems that inductors are not being treated correctly.  According
# to some research, inductor stamp affects the B,C and D arrays.  Erik
# Cheever's code puts inductors into the G matrix as 1/s/L.  LTspice 
# results are different than the python code.  Capacitors seem to work OK.
# Plan is to add controlled sources, then get inductors working.
# 9/6/2017
# opamp_test_circuit_426 is not working.  Results not the same as LTspice
# Chebyshev_LPF_1dB_4pole: cut off frequency not correct, other features look OK
# still need to debug opamps and inductors
# Adding: VCCS = G type branch element: G needs to be modified
# CCVS = H type branch element: B, C and D need to be modified
# 
# left off editing at the B matrix
# 
# 9/10/2017
# researching formulation of B matrix
# what about a network with only 1 current source?
# 
# 
# CCVS = H type branch element: B, C and D need to be modified
# CCCS = F type branch element: B, C and D need to be modified
# VCCS = G type branch element: G needs to be modified
# VCVS = E type branch element: B and C need to be modified
# 
# For CCCS = F type branch element, for this type of element, need to add a zero volt voltage source to the net list through which the current flows.
# For CCVS = H type branch element, need to add a zero volt voltage source to the net list through which the current flows.  The dependent voltage source is already included in the net list as H type.
# ```

import os
from sympy import *
import numpy as np
import pandas as pd
init_printing()


# initialize some variables, count the types of elements
num_rlc = 0 # number of passive elements
num_ind = 0 # number of inductors
num_v = 0    # number of independent voltage sources
num_i = 0    # number of independent current sources
i_unk = 0  # number of current unknowns
num_opamps = 0   # number of op amps
num_vcvs = 0     # number of controlled sources of various types
num_vccs = 0
num_cccs = 0
num_ccvs = 0
num_cpld_ind = 0 # number of coupled inductors


# ##### open file and preprocess it, file name extenstion is defaulted to .net
# - remove blank lines and comments
# - convert first letter of element name to upper case
# - removes extra spaces between entries
# - count number of entries on each line, make sure the count is correct
# 

fn = 'example48'
fd1 = open(fn+'.net','r')
content = fd1.readlines()
content = [x.strip() for x in content]  #remove leading and trailing white space
# remove empty lines
while '' in content:
    content.pop(content.index(''))

# remove comment lines, these start with a asterisk *
content = [n for n in content if not n.startswith('*')]
# remove other comment lines, these start with a semicolon ;
content = [n for n in content if not n.startswith(';')]
# remove spice directives, these start with a period, .
content = [n for n in content if not n.startswith('.')]
# converts 1st letter to upper case
#content = [x.upper() for x in content] <- this converts all to upper case
content = [x.capitalize() for x in content]
# removes extra spaces between entries
content = [' '.join(x.split()) for x in content]


line_cnt = len(content) # number of lines in the netlist
branch_cnt = 0  # number of btanches in the netlist
# check number of entries on each line
for i in range(line_cnt):
    x = content[i][0]
    tk_cnt = len(content[i].split()) # split the line into tokens

    if (x == 'R') or (x == 'L') or (x == 'C'):
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_rlc += 1
        branch_cnt += 1
        if x == 'L':
            num_ind += 1
    elif x == 'V':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_v += 1
        branch_cnt += 1
    elif x == 'I':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_i += 1
        branch_cnt += 1
    elif x == 'O':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_opamps += 1
    elif x == 'E':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vcvs += 1
        branch_cnt += 1
    elif x == 'G':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vccs += 1
        branch_cnt += 1
    elif x == 'F':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_cccs += 1
        branch_cnt += 1
    elif x == 'H':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_ccvs += 1
        branch_cnt += 1
    elif x == 'K':
        if (tk_cnt != 4):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_cpld_ind += 1
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))


# ##### parser
# - puts branch elements into structure
# - counts number of nodes
# 
# data frame lables:
# - count: data frame index
# - element: type of element
# - p node: positive node
# - n node: negitive node, for a current source, the arrow terminal
# - cp node: controlling positive node of branch
# - cn node: controlling negitive node of branch
# - Vout: opamp output node
# - value: value of element or voltage
# - Vname: voltage source through which the controlling current flows. Need to add a zero volt voltage source to the controlling branch.
# - Lname1: name of coupled inductor 1
# - Lname2: name of coupled inductor 2
# 

# build the pandas data frame
df = pd.DataFrame(columns=['element','p node','n node','cp node','cn node',
    'Vout','value','Vname','Lname1','Lname2'])
# this data frame is for the unknown currents
df2 = pd.DataFrame(columns=['element','p node','n node'])


# ##### functions to load branch elements into data frame
# 

# loads voltage or current sources into branch structure
def indep_source(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

# loads passive elements into branch structure
def rlc_element(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

# loads multi-terminal elements into branch structure
# O - Op Amps
def opamp_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vout'] = int(tk[3])

# G - VCCS
def vccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

# E - VCVS
def vcvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

# F - CCCS
def cccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

# H - CCVS
def ccvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

# K - Coupled inductors
def cpld_ind_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'Lname1'] = tk[1].capitalize()
    df.loc[line_nu,'Lname2'] = tk[2].capitalize()
    df.loc[line_nu,'value'] = float(tk[3])


# function to scan df and get largest node number
def count_nodes():
    # need to check that nodes are consecutive
    # fill array with node numbers
    p = np.zeros(line_cnt+1)
    for i in range(line_cnt-1):
        p[df['p node'][i]] = df['p node'][i]
        p[df['n node'][i]] = df['n node'][i]

    # find the largest node number
    if df['n node'].max() > df['p node'].max():
        largest = df['n node'].max()
    else:
        largest =  df['p node'].max()

    largest = int(largest)
    # check for unfilled elements, skip node 0
    for i in range(1,largest):
        if p[i] == 0:
            print('nodes not in continuous order, node {:.0f} is missing'.format(p[i-1]+1))

    return largest


# ### new function
# 

# function to enumerate current unknowns which have input to B matrix
# need to be able to fine the column number and nodes for controlled sources

# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
unk_currents
count = 0
def func1():
    # need to walk through data frame and find these parameters
    for i in range(len(df)):
        n1 = df.loc[i,'p node']
        n2 = df.loc[i,'n node']

        # process all the elements creating unknown currents
        x = df.loc[i,'element'][0]   #get 1st letter of element name
        if (x == 'L') or (x == 'V') or (x == 'O') or (x == 'E') or (x == 'H'):
            df2.loc[count,'element'] = df.loc[i,'element']
            df2.loc[count,'p node'] = df.loc[i,'p_node']
            df2.loc[count,'n node'] = df.loc[i,'n node']
            count += 1
        
        










# load branch info into data frame
for i in range(line_cnt):
    x = content[i][0]

    if (x == 'R') or (x == 'L') or (x == 'C'):
        rlc_element(i)
    elif (x == 'V') or (x == 'I'):
        indep_source(i)
    elif x == 'O':
        opamp_sub_network(i)
    elif x == 'E':
        vcvs_sub_network(i)
    elif x == 'G':
        vccs_sub_network(i)
    elif x == 'F':
        cccs_sub_network(i)
    elif x == 'H':
        ccvs_sub_network(i)
    elif x == 'K':
        cpld_ind_sub_network(i)
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))

# count number of nodes
num_nodes = count_nodes()


# print a report
print('Net list report')
print('number of lines in netlist: {:d}'.format(line_cnt))
print('number of branches: {:d}'.format(branch_cnt))
print('number of nodes: {:d}'.format(num_nodes))
# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
print('number of unknown currents: {:d}'.format(i_unk))
print('number of passive components: {:d}'.format(num_rlc))
print('number of inductors: {:d}'.format(num_ind))
print('number of independent voltage sources: {:d}'.format(num_v))
print('number of independent current sources: {:d}'.format(num_i))
print('number of op amps: {:d}'.format(num_opamps))

# not implemented yet
print('\nNot implemented yet')
print('number of E - VCVS: {:d}'.format(num_vcvs))
print('number of G - VCCS: {:d}'.format(num_vccs))
print('number of F - CCCS: {:d}'.format(num_cccs))
print('number of H - CCVS: {:d}'.format(num_ccvs))
print('number of K - Coupled inductors: {:d}'.format(num_cpld_ind))


df


# store the data frame as a pickle file
# df.to_pickle(fn+'.pkl')  # <- uncomment if needed


# initialize some symbolic matrix with zeros
# A is formed by [[G, C] [B, D]]
# Z = [I,E]
# X = [V, J]
V = zeros(num_nodes,1)
I = zeros(num_nodes,1)
G = zeros(num_nodes,num_nodes)  # also called Yr, the reduced nodal matrix
s = Symbol('s')  # the Laplace variable

# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
# is is possible to have i_unk == 0 ?, what about a network with only current sources?
i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
if i_unk != 0:
    B = zeros(num_nodes,i_unk)
    C = zeros(i_unk,num_nodes)
    D = zeros(i_unk,i_unk)
    E = zeros(i_unk,1)
    J = zeros(i_unk,1)


# ##### G matrix <span style="color:red">\----need to check on inductor treatment, doesn't verify with LTspice testing, inductor stamp affects the B,C and D arrays</span>
# The G matrix is n by n and is determined by the interconnections between the passive circuit elements (RLC's).  The G matrix is an nxn matrix formed in two steps:
# 1. Each element in the diagonal matrix is equal to the sum of the conductance (one over the resistance) of each element connected to the corresponding node.  So the first diagonal element is the sum of conductances connected to node 1, the second diagonal element is the sum of conductances connected to node 2, and so on.
# 2. The off diagonal elements are the negative conductance of the element connected to the pair of corresponding node.  Therefore a resistor between nodes 1 and 2 goes into the G matrix at location (1,2) and locations (2,1).
# 
# Adding VCCS, G type element
# 
# In the orginal paper B is called Yr, where Yr, is a reduced form of the nodal matrix excluding the contributions due to voltage sources, current controlling elements, etc.
# 

# G matrix
for i in range(len(df)):  # don't use branch count use # of rows in data frame
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    cn1 = df.loc[i,'cp node']
    cn2 = df.loc[i,'cn node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'R':
        g = 1/sympify(df.loc[i,'element'])
#    if x == 'L':
#        g = 1/s/sympify(df.loc[i,'element'])  # this matches Eric's code, but I thinks is wrong
    if x == 'C':
        g = s*sympify(df.loc[i,'element'])
    if x == 'G':   #vccs type element
        g = sympify(df.loc[i,'element'].lower())  # use a symbol for gain value # df.loc[i,'value']   # get the gain value

    if (x == 'R') or (x == 'C'):   # fix this don't do L's <----
    #if (x == 'R') or (x == 'L') or (x == 'C'):   # fix this don't do L's <----
        # If neither side of the element is connected to ground
        # then subtract it from appropriate location in matrix.
        if (n1 != 0) and (n2 != 0):
            G[n1-1,n2-1] += -g
            G[n2-1,n1-1] += -g

        # If node 1 is connected to ground, add element to diagonal of matrix
        if n1 != 0:
            G[n1-1,n1-1] += g

        # same for for node 2
        if n2 != 0:
            G[n2-1,n2-1] += g

    if x == 'G':    #vccs type element
        # check to see if any terminal is grounded
        # then stamp the matrix
        if n1 != 0 and cn1 != 0:
            G[n1-1,cn1-1] += g

        if n2 != 0 and cn2 != 0:
            G[n2-1,cn2-1] += g

        if n1 != 0 and cn2 != 0:
            G[n1-1,cn2-1] -= g

        if n2 != 0 and cn1 != 0:
            G[n2-1,cn1-1] -= g

G  # display the G matrix


# ##### B Matrix
# Rules for making the B matrix
# The B matrix is an n by m matrix with only 0, 1 and -1 elements.  Each location in the matrix corresponds to a particular voltage source (first dimension) or a node (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a -1.  Otherwise, elements of the B matrix are zero.
# 
# coding notes:
# 
# not only voltage sources but controlled sources like cccs
# 
# probably need some code to make sure number of column equals the number of voltage sources.
# 
# number of columns = num_v+num_opamps+num_ccvs+num_cccs+num_vcvs ;5 element types, every column is a current i sub k
# if num_v_sources > 1, the B is n by m, otherwise B is n by 1.
# 
# B[row, column]
# 
# Is there a valid case for not having a B matrix, i_unk = 0? 
# Is there a valide op amp case where  B is n by 1?
# 

# find the the column position in the B matrix
def find_vnam(name):
    # vn1, vn2, col_num
    return 0, 5, 3  # for now return dummy values


# need n1, n2 and column number for B
# build a list of i_unk ?
# given element name
# find the colum number and return the node numbers
# 

i=6
#df.loc[i,'element']
#df.loc[i,'cp node'] # nodes for controlled sources
#df.loc[i,'cn node']
#df.loc[i,'Vout'] # node connected to op amp output
#df.loc[i,'Vname']
df.loc[i,'p node']
#df.loc[i,'n node']


# generate the B Matrix
# loop through all the branches and process elements that have stamps for the B matrix
# V: voltage sources, O: opamps, H: ccvs, F: cccs, E: vcvs and inductors, these are counted in variable i_unk
# The of the columns is as they appear in the netlist
# F: cccs does not get its own column because the controlling current is through a zero volt voltage source, called Vname
sn = 0   # count source number as code walks through the data frame
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    cn1 = df.loc[i,'cp node'] # nodes for controlled sources
    cn2 = df.loc[i,'cn node']
    n_vout = df.loc[i,'Vout'] # node connected to op amp output

    # process elements with input to B matrix
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if i_unk > 1:  #is B greater than 1 by n?
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'O':  # op amp type
        B[n_vout-1,sn] = 1
        sn += 1   # increment source count
    if (x == 'H') or (x == 'F'):  # H: ccvs, F: cccs,
        if i_unk > 1:  #is B greater than 1 by n?
            # check to see if any terminal is grounded
            # then stamp the matrix
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
            # need to find the vn for Vname
            vn1, vn2, col_num = find_vnam(df.loc[i,'Vname'])
            if vn2 != 0:
                B[col_num-1,vn1] = 1 # need to fix this, not cn
            if vn1 != 0:
                B[col_num-1,vn2] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'E':   # vcvs type, only ik column is altered at n1 and n2
        if i_unk > 1:  #is B greater than 1 by n?
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'L':
        if i_unk > 1:  #is B greater than 1 by n?
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count

# check source count
if sn != i_unk:
    print('source number not equal to i_unk in matrix B')

B   # display the B matrix


# ##### C matrix
# The C matrix is an m by n matrix with only 0, 1 and -1 elements (except for controlled sources).  Each location in the matrix corresponds to a particular node (first dimension) or voltage source (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a -1.  Otherwise, elements of the C matrix are zero.
# 
# <span style="color:red">C matrix needs to be fixed</span>
# 

# generate the C matrix
sn = 0   # count source number
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    n_vout = df.loc[i,'Vout'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        if n1 != 0:
            C[oan+num_v,n1-1] = 1
        if n2 != 0:
            C[oan+num_v,n2-1] = -1
        oan += 1  # increment op amp number

C   # display the C matrix


# ##### D matrix
# The D matrix is an mxm matrix that is composed entirely of zeros.  (It can be non-zero if controlled sources are considered.)
# 

# display the The D matrix
D


# ##### I matrix
# The I matrix is an n by 1 matrix with each element of the matrix corresponding to a particular node.  The value of each element of I is determined by the sum of current sources into the corresponding node.  If there are no current sources connected to the node, the value is zero.
# 

# generate the I matrix, current sources have N2 = arrow end
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'I':
        g = sympify(df.loc[i,'element'])
        # sum the current into each node
        if n1 != 0:
            I[n1-1] -= g
        if n2 != 0:
            I[n2-1] += g

I  # display the I matrix


# ##### V matrix
# The V matrixis an nx1 matrix formed of the node voltages.  Each element in V corresponds to the voltage at the equivalent node in the circuit
# 

# generate the V matrix
for i in range(num_nodes):
    V[i] = sympify('v{:d}'.format(i+1))

V  # display the V matrix


# ##### J matrix
# The is an m by 1 matrix, with one entry for the current through each voltage source.
# 

# The J matrix is an mx1 matrix, with one entry for the current through each voltage source.
sn = 0   # count source number
oan = 0   #count op amp number
for i in range(len(df)):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        J[sn] = sympify('I_{:s}'.format(df.loc[i,'element']))
        sn += 1
    if x == 'O':  # this needs to be checked <---- needs debugging
        J[oan+num_v] = sympify('I_{:s}'.format(df.loc[i,'element']))
        oan += 1

J  # diplay the J matrix


# ##### E matrix
# The E matrix is mx1 and holds the values of the independent voltage sources.
# 

# generate the E matrix
sn = 0   # count source number
for i in range(len(df)):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        E[sn] = sympify(df.loc[i,'element'])
        sn += 1

E   # display the E matrix


# ##### Z matrix
# The Z matrix holds the independent voltage and current sources and is the combination of 2 smaller matrices I and E.  The Z matrix is (m+n) by 1, n is the number of nodes, and m is the number of independent voltage sources.  The I matrix is n by 1 and contains the sum of the currents through the passive elements into the corresponding node (either zero, or the sum of independent current sources). The E matrix is m by 1 and holds the values of the independent voltage sources.
# 

Z = I[:] + E[:]
Z  # display the Z matrix


# ##### X matrix
# The X matrix is an (n+m) by 1 vector that holds the unknown quantities (node voltages and the currents through the independent voltage sources). The top n elements are the n node voltages. The bottom m elements represent the currents through the m independent voltage sources in the circuit. The V matrix is n by 1 and holds the unknown voltages.  The J matrix is m by 1 and holds the unknown currents through the voltage sources
# 

X = V[:] + J[:]
X  # display the X matrix


# ##### A matrix
# The A matrix is (m+n) by (m+n) and will be developed as the combination of 4 smaller matrices, G, B, C, and D.
# 

n = num_nodes
m = num_v+num_opamps
A = zeros(m+n,m+n)
for i in range(n):
    for j in range(n):
        A[i,j] = G[i,j]

if num_v+num_opamps > 1:
    for i in range(n):
        for j in range(m):
            A[i,n+j] = B[i,j]
            A[n+j,i] = C[j,i]
else:
    for i in range(n):
        A[i,n] = B[i]
        A[n,i] = C[i]

A  # display the A matrix


# generate the circuit equations
n = num_nodes
m = num_v+num_opamps
eq_temp = 0  # temporary equation used to build up the equation
equ = zeros(m+n,1)  #initialize the array to hold the equations
for i in range(n+m):
    for j in range(n+m):
        eq_temp += A[i,j]*X[j]
    equ[i] = Eq(eq_temp,Z[i])
    eq_temp = 0

equ   # display the equations


# Use the str() function to convert sympy equations to strings.  These strings can be copid to a new notebook.
# 

str(equ)


str(equ.free_symbols)


str(X)


df








# # A python node analysis jupyter notebook
# 

# **Synopsis:** This notebook will read in a spice like circuit netlist file and compute the node equations. The code follows Erik Cheever's Analysis of  Resistive Circuits [page](http://www.swarthmore.edu/NatSci/echeeve1/Ref/mna/MNA1.html) to generate modified nodal equations. I somewhat followed his matlab file.
# 
# **Description:**
# 

# ```
# Date started: April 17, 2017
# file name: node analysis.ipynb
# Requires: Python version 3 or higher
# Author: Tony
# 
# Revision History
# 7/1/2015: Ver 1 - coding started, derived from network.c code
# 8/18/2017
# changed approach, now implementing a modified nodal analysis
# 8/19/2017
# Wrote some code to generate symbolic matrices, works ok,
# so heading down the sympy path. Basic debugging finished,
# but still need to verify some circuits using Ls and Cs.
# 8/30/2017
# Started to add code for op amps
# 9/1/2017
# Code added to process op amps
# 9/3/2017
# Added code to remove spice directives.
# Fixed orientation of current sources in I matrix.
# N2 is the arrow end of the current source.
# 
# Started a task list in CoCalc for this project:
# Build some test circuits for debugging the code
# test the circuits in LTSpice compare solutions
# Add controlled sources
# Add coupled inductors
# ```
# 

import os
from sympy import *
import numpy as np
import pandas as pd
init_printing()


# initialize some variables, count the types of elements
num_rlc = 0 # number of passive elements
num_v = 0    # number of independent voltage sources
num_i = 0    # number of independent current sources
num_opamps = 0   # number of op amps
num_vcvs = 0     # number of controlled sources of various types
num_vccs = 0
num_cccs = 0
num_ccvs = 0
num_cpld_ind = 0 # number of coupled inductors


# ##### open file and preprocess it, file name extenstion is .net
# - remove blank lines and comments
# - convert first letter of element name to upper case
# - removes extra spaces between entries
# - count number of entries on each line, make sure the count is correct
# 

fn = 'example420'
fd1 = open(fn+'.net','r')
content = fd1.readlines()
content = [x.strip() for x in content]  #remove leading and trailing white space
# remove empty lines
while '' in content:
    content.pop(content.index(''))

# remove comment lines, these start with a asterisk *
content = [n for n in content if not n.startswith('*')]
# remove spice directives. these start with a period, .
content = [n for n in content if not n.startswith('.')]
# converts 1st letter to upper case
#content = [x.upper() for x in content] <- this converts all to upper case
content = [x.capitalize() for x in content]
# removes extra spaces between entries
content = [' '.join(x.split()) for x in content]


branch_cnt = len(content)
# check number of entries on each line
for i in range(branch_cnt):
    x = content[i][0]
    tk_cnt = len(content[i].split())

    if (x == 'R') or (x == 'L') or (x == 'C'):
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_rlc += 1
    elif x == 'V':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_v += 1
    elif x == 'I':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_i += 1
    elif x == 'O':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_opamps += 1
    elif x == 'E':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vcvs += 1
    elif x == 'G':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vccs += 1
    elif x == 'F':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_cccs += 1
    elif x == 'H':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_ccvs += 1
    elif x == 'K':
        if (tk_cnt != 4):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_cpld_ind += 1
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))


# ##### parser
# - puts branch elements into structure
# - counts number of nodes
# 

# build the pandas data frame
count = []        # data frame index
element = []      # type of element
p_node = []       # positive node
n_node = []       # neg node, for a current source, the arrow terminal
cp_node = []      # controlling positive node of branch
cn_node = []      # controlling negitive node of branch
v_out = []        # op amp output node
value = []        # value of element or voltage
v_name = []       # voltage source through which the controlling current flows
l_name1 = []      # name of coupled inductor 1
l_name2 = []      # name of coupled inductor 2

df = pd.DataFrame(index=count, columns=['element','p node','n node','cp node','cn node',
    'v out','value','v name','l_name1','l_name2'])


# ##### functions to load branch elements into data frame
# 

# loads voltage or current sources into branch structure
def indep_source(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'value'] = float(tk[3])

# loads passive elements into branch structure
def rlc_element(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'value'] = float(tk[3])

'''
loads multi-terminal sub-networks
into branch structure
Types:
E - VCVS
G - VCCS
F - CCCS
H - CCVS
not implemented yet:
K - Coupled inductors
O - Op Amps
'''
def opamp_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'v out'] = int(tk[3])

def vccs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'cp node'] = int(tk[3])
    df.loc[br_nu,'cn node'] = int(tk[4])
    df.loc[br_nu,'value'] = float(tk[5])

def vcvs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'cp node'] = int(tk[3])
    df.loc[br_nu,'cn node'] = int(tk[4])
    df.loc[br_nu,'value'] = float(tk[5])

def cccs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'v name'] = tk[3]
    df.loc[br_nu,'value'] = float(tk[4])

def ccvs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'v name'] = tk[3]
    df.loc[br_nu,'value'] = float(tk[4])

def cpld_ind_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'l name1'] = tk[1]
    df.loc[br_nu,'l name2'] = tk[2]
    df.loc[br_nu,'value'] = float(tk[3])


# function to scan df and get largest node number
def count_nodes():
    # need to check that nodes are consecutive
    # fill array with node numbers
    p = np.zeros(branch_cnt+1)
    for i in range(branch_cnt-1):
        p[df['p node'][i]] = df['p node'][i]
        p[df['n node'][i]] = df['n node'][i]

    # find the largest node number
    if df['n node'].max() > df['p node'].max():
        largest = df['n node'].max()
    else:
        largest =  df['p node'].max()

        largest = int(largest)
    # check for unfilled elements, skip node 0
    for i in range(1,largest):
        if p[i] == 0:
            print("nodes not in continuous order");

    return largest


# load branches into data frame
for i in range(branch_cnt):
    x = content[i][0]

    if (x == 'R') or (x == 'L') or (x == 'C'):
        rlc_element(i)
    elif (x == 'V') or (x == 'I'):
        indep_source(i)
    elif x == 'O':
        opamp_sub_network(i)
    elif x == 'E':
        vcvs_sub_network(i)
    elif x == 'G':
        vccs_sub_network(i)
    elif x == 'F':
        cccs_sub_network(i)
    elif x == 'H':
        ccvs_sub_network(i)
    elif x == 'K':
        cpld_ind_sub_network(i)
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))

# count number of nodes
num_nodes = count_nodes()


# print a report
print('Net list report')
print('number of branches: {:d}'.format(branch_cnt))
print('number of nodes: {:d}'.format(num_nodes))
print('number of passive components: {:d}'.format(num_rlc))
print('number of independent voltage sources: {:d}'.format(num_v))
print('number of independent current sources: {:d}'.format(num_i))
print('number of op amps: {:d}'.format(num_opamps))

# not implemented yet
print('\nNot implemented yet')
print('number of E - VCVS: {:d}'.format(num_vcvs))
print('number of G - VCCS: {:d}'.format(num_vccs))
print('number of F - CCCS: {:d}'.format(num_cccs))
print('number of F - CCCS: {:d}'.format(num_ccvs))
print('number of K - Coupled inductors: {:d}'.format(num_cpld_ind))


# store the data frame as a pickle file
df.to_pickle(fn+'.pkl')


# initialize some symbolic matrix with zeros
# A is formed by [[G, C] [B, D]]
# Z = [I,E]
# X = [V, J]
V = zeros(num_nodes,1)
I = zeros(num_nodes,1)
G = zeros(num_nodes,num_nodes)
s = Symbol('s')  # the Laplace variable

if (num_v+num_opamps) != 0:
    B = zeros(num_nodes,num_v+num_opamps)
    C = zeros(num_v+num_opamps,num_nodes)
    D = zeros(num_v+num_opamps,num_v+num_opamps)
    E = zeros(num_v+num_opamps,1)
    J = zeros(num_v+num_opamps,1)


# ##### G matrix
# The G matrix is n by n and is determined by the interconnections between the passive circuit elements (RLC's).  The G matrix is an nxn matrix formed in two steps:
# 1. Each element in the diagonal matrix is equal to the sum of the conductance (one over the resistance) of each element connected to the corresponding node.  So the first diagonal element is the sum of conductances connected to node 1, the second diagonal element is the sum of conductances connected to node 2, and so on.
# 2. The off diagonal elements are the negative conductance of the element connected to the pair of corresponding node.  Therefore a resistor between nodes 1 and 2 goes into the G matrix at location (1,2) and locations (2,1).
# 

# G matrix
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'R':
        g = 1/sympify(df.loc[i,'element'])
    if x == 'L':
        g = 1/(s/sympify(df.loc[i,'element']))
    if x == 'C':
        g = sympify(df.loc[i,'element'])*s

    if (x == 'R') or (x == 'L') or (x == 'C'):
        # If neither side of the element is connected to ground
        # then subtract it from appropriate location in matrix.
        if (n1 != 0) and (n2 != 0):
            G[n1-1,n2-1] += -g
            G[n2-1,n1-1] += -g

        # If node 1 is connected to ground, add element to diagonal of matrix
        if n1 != 0:
            G[n1-1,n1-1] += g

        # same for for node 2
        if n2 != 0:
            G[n2-1,n2-1] += g

G  # display the G matrix


# ##### I matrix
# The I matrix is an n by 1 matrix with each element of the matrix corresponding to a particular node.  The value of each element of I is determined by the sum of current sources into the corresponding node.  If there are no current sources connected to the node, the value is zero.
# 

# generate the I matrix, current sources have N2 = arrow end
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'I':
        g = sympify(df.loc[i,'element'])
        # sum the current into each node
        if n1 != 0:
            I[n1-1] -= g
        if n2 != 0:
            I[n2-1] += g

I  # display the I matrix


# ##### V matrix
# The V matrixis an nx1 matrix formed of the node voltages.  Each element in V corresponds to the voltage at the equivalent node in the circuit
# 

# generate the V matrix
for i in range(num_nodes):
    V[i] = sympify('v{:d}'.format(i+1))

V  # display the V matrix


# ##### B Matrix
# Rules for making the B matrix
# The B matrix is an nxm matrix with only 0, 1 and -1 elements.  Each location in the matrix corresponds to a particular voltage source (first dimension) or a node (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a -1.  Otherwise, elements of the B matrix are zero.
# 

# generate the B Matrix
# loop through all the branches and process independent voltage sources
sn = 0   # count source number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(branch_cnt):
    n_vout = df.loc[i,'v out'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        B[n_vout-1,oan+num_v] = 1
        oan += 1   # increment op amp count

B   # display the B matrix


# ##### J matrix
# The is an mx1 matrix, with one entry for the current through each voltage source.
# 

# The J matrix is an mx1 matrix, with one entry for the current through each voltage source.
sn = 0   # count source number
oan = 0   #count op amp number
for i in range(branch_cnt):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        J[sn] = sympify('I_{:s}'.format(df.loc[i,'element']))
        sn += 1
    if x == 'O':  # this needs to be checked <---- needs debugging
        J[oan+num_v] = sympify('I_{:s}'.format(df.loc[i,'element']))
        oan += 1

J  # diplay the J matrix


# ##### C matrix
# The C matrix is an mxn matrix with only 0, 1 and -1 elements.  Each location in the matrix corresponds to a particular node (first dimension) or voltage source (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a -1.  Otherwise, elements of the C matrix are zero.
# 

# generate the C matrix
sn = 0   # count source number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    n_vout = df.loc[i,'v out'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        if n1 != 0:
            C[oan+num_v,n1-1] = 1
        if n2 != 0:
            C[oan+num_v,n2-1] = -1
        oan += 1  # increment op amp number

C   # display the C matrix


# ##### D matrix
# The D matrix is an mxm matrix that is composed entirely of zeros.  (It can be non-zero if dependent sources are considered.)
# 

# display the The D matrix
D


# ##### E matrix
# The E matrix is mx1 and holds the values of the independent voltage sources.
# 

# generate the E matrix
sn = 0   # count source number
for i in range(branch_cnt):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        E[sn] = sympify(df.loc[i,'element'])
        sn += 1

E   # display the E matrix


# ##### Z matrix
# The Z matrix holds the independent voltage and current sources and is the combination of 2 smaller matrices I and E.  The Z matrix is (m+n) by 1, n is the number of nodes, and m is the number of independent voltage sources.  The I matrix is n by 1 and contains the sum of the currents through the passive elements into the corresponding node (either zero, or the sum of independent current sources). The E matrix is m by 1 and holds the values of the independent voltage sources.
# 

Z = I[:] + E[:]
Z  # display the Z matrix


# ##### X matrix
# The X matrix is an (n+m) by 1 vector that holds the unknown quantities (node voltages and the currents through the independent voltage sources). The top n elements are the n node voltages. The bottom m elements represent the currents through the m independent voltage sources in the circuit. The V matrix is n by 1 and holds the unknown voltages.  The J matrix is m by 1 and holds the unknown currents through the voltage sources
# 

X = V[:] + J[:]
X  # display the X matrix


# ##### A matrix
# The A matrix is (m+n) by (m+n) and will be developed as the combination of 4 smaller matrices, G, B, C, and D.
# 

n = num_nodes
m = num_v+num_opamps
A = zeros(m+n,m+n)
for i in range(n):
    for j in range(n):
        A[i,j] = G[i,j]

if num_v+num_opamps > 1:
    for i in range(n):
        for j in range(m):
            A[i,n+j] = B[i,j]
            A[n+j,i] = C[j,i]
else:
    for i in range(n):
        A[i,n] = B[i]
        A[n,i] = C[i]

A  # display the A matrix


# generate the circuit equations
n = num_nodes
m = num_v+num_opamps
eq_temp = 0  # temporary equation used to build up the equation
equ = zeros(m+n,1)  #initialize the array to hold the equations
for i in range(n+m):
    for j in range(n+m):
        eq_temp += A[i,j]*X[j]
    equ[i] = Eq(eq_temp,Z[i])
    eq_temp = 0

equ   # display the equations


# Use the str() function to convert sympy equations to strings.  These strings can be copid to a new notebook.
# 

str(equ)


str(equ.free_symbols)


df








# # Symbolic modified nodal analysis
# Last update: 9/30/2017
# 

# **Abstract:** This notebook will read in a spice like circuit netlist file and compute the network equations in symbolic form. These equations can then be copied to a different notebook where the node voltages can be solved using sympy or numpy.
# 
# **Description:** This node analysis code started as a translation from some C code to generate a nodal admittance matrix that I had written in 1988.  The original C code worked well and calculated numeric solutions.  I then started writing some C code to generate the matrices with symbolic values and then intended to use LISP to symbolically solve the equations.  I didn’t get too far with this effort.  The LISP code would generate huge symbolic strings with no simplification.  The output was a big pile of trash that was not in the least bit useful or decipherable.
# 
# In 2014, I started to use python for my little coding projects and engineering calculations.  There are some nice python libraries for numeric and symbolic calculations (such as numpy and sympy), so I decided to try writing a python script to generate the node equations based on the old C code I had written many years before.  Part way into this project I discovered that there is a new nodal analysis technique being taught today in engineering school called the modified nodal analysis (1,2).  The modified nodal analysis provides an algorithmic method for generating systems of independent equations for linear circuit analysis.  Some of my younger colleagues at work were taught this method, but I never heard of it until a short time ago.  These days, I never really analyze a circuit by hand, unless it’s so simple that you can almost do it by inspection.  Most problems that an electrical engineer encounters on the job are complex enough that they use computers to analyze the circuits.  LTspice is the version of spice that I use, since it’s free and does a good job converging when analyzing switching circuits.
# 
# The code follows Erik Cheever's Analysis of  Resistive Circuits [page](http://www.swarthmore.edu/NatSci/echeeve1/Ref/mna/MNA1.html) to generate modified nodal equations. I somewhat followed his matlab file for resistors, capacitors, opamps and independent sources.  The preprocessor and parser code was converted from my old C code.  The use of pandas for a data frame is new and sympy is used to do the math.
# 
# After doing some verification testing with inductors and capacitors, it seems that inductors are not being treated correctly.  According to some research, the inductor stamp affects the B,C and D arrays.  Erik Cheever's code puts inductors into the G matrix as 1/s/L.  LTspice results are different than the python code.  Capacitors seem to work OK.
# 
# Reference:
# 1. The modified nodal approach to network analysis, Chung-Wen Ho, A. Ruehli, P. Brennan, IEEE Transactions on Circuits and Systems ( Volume: 22, Issue: 6, Jun 1975 )
# 2. https://en.wikipedia.org/wiki/Modified_nodal_analysis
# 3. ECE 570 Session 3, Computer Aided Engineering for Integrated Circuits, http://www2.engr.arizona.edu/~ece570/session3.pdf
# 
# Some notes from reference 1:
# Capacitances and inductances are considered only in the time domain and their contributions, shown in Table I, are obtained by applying finite differencing methods to their branch relations.
# 
# links: http://www.solved-problems.com/circuits/electrical-circuits-problems/716/supernode-dependent-voltage-source/
# 

# ```
# Date started: April 17, 2017
# file name: node analysis.ipynb
# Requires: Python version 3 or higher and a jupyter notebook
# Author: Tony
# 
# Revision History
# 7/1/2015: Ver 1 - coding started, derived from network.c code
# 8/18/2017
# changed approach, now implementing a modified nodal analysis
# 8/19/2017
# Wrote some code to generate symbolic matrices, works ok,
# so heading down the sympy path. Basic debugging finished,
# but still need to verify some circuits using Ls and Cs.
# 8/30/2017
# Started to add code for op amps
# 9/1/2017
# Code added to process op amps
# 9/3/2017
# Added code to remove spice directives.
# Fixed orientation of current sources in I matrix.
# N2 is the arrow end of the current source.
# 9/5/2017
# After doing some verification testing with inductors and capacitors,
# it seems that inductors are not being treated correctly.  According
# to some research, inductor stamp affects the B,C and D arrays.  Erik
# Cheever's code puts inductors into the G matrix as 1/s/L.  LTspice 
# results are different than the python code.  Capacitors seem to work OK.
# Plan is to add controlled sources, then get inductors working.
# 9/6/2017
# opamp_test_circuit_426 is not working.  Results not the same as LTspice
# Chebyshev_LPF_1dB_4pole: cut off frequency not correct, other features look OK
# still need to debug opamps and inductors
# Adding: VCCS = G type branch element: G needs to be modified
# CCVS = H type branch element: B, C and D need to be modified
# 9/10/2017
# researching formulation of B matrix
# what about a network with only 1 current source?  The B, C and D matrix would be 0 by 0.
# Think about changing the name of the G matrix to Yr, to keep same as Ho's IEEE paper.
# 
# CCVS = H type branch element: B, C and D need to be modified
# CCCS = F type branch element: B, C and D need to be modified
# VCCS = G type branch element: G needs to be modified
# VCVS = E type branch element: B and C need to be modified
# 
# For CCCS = F type branch element, for this type of element, need to add a zero volt voltage source to the net list through which the current flows.
# For CCVS = H type branch element, need to add a zero volt voltage source to the net list through which the current flows.  The dependent voltage source is already included in the net list as H type.
# 
# 9/12/2017
# still working on the B matrix
# 9/18/2017
# still debugging B matrix, looks like we don't need find_vname() or df2.  This is because a zero volt voltage source is add to the net list in spice.
# need to add cccs type to the list of i_unk.
# Filled out some B matrices by hand and got the same answer as the code.
# 9/30/2017
# working on C & D matrix
# ```
# 

import os
from sympy import *
import numpy as np
import pandas as pd
init_printing()


# initialize some variables, count the types of elements
num_rlc = 0 # number of passive elements
num_ind = 0 # number of inductors
num_v = 0    # number of independent voltage sources
num_i = 0    # number of independent current sources
i_unk = 0  # number of current unknowns
num_opamps = 0   # number of op amps
num_vcvs = 0     # number of controlled sources of various types
num_vccs = 0
num_cccs = 0
num_ccvs = 0
num_cpld_ind = 0 # number of coupled inductors


# ## open file and preprocess it
# - file name extenstion is defaulted to .net
# - remove blank lines and comments
# - convert first letter of element name to upper case
# - removes extra spaces between entries
# - count number of entries on each line, make sure the count is correct
# 

fn = 'example48'
fd1 = open(fn+'.net','r')
content = fd1.readlines()
content = [x.strip() for x in content]  #remove leading and trailing white space
# remove empty lines
while '' in content:
    content.pop(content.index(''))

# remove comment lines, these start with a asterisk *
content = [n for n in content if not n.startswith('*')]
# remove other comment lines, these start with a semicolon ;
content = [n for n in content if not n.startswith(';')]
# remove spice directives, these start with a period, .
content = [n for n in content if not n.startswith('.')]
# converts 1st letter to upper case
#content = [x.upper() for x in content] <- this converts all to upper case
content = [x.capitalize() for x in content]
# removes extra spaces between entries
content = [' '.join(x.split()) for x in content]


line_cnt = len(content) # number of lines in the netlist
branch_cnt = 0  # number of btanches in the netlist
# check number of entries on each line
for i in range(line_cnt):
    x = content[i][0]
    tk_cnt = len(content[i].split()) # split the line into tokens

    if (x == 'R') or (x == 'L') or (x == 'C'):
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_rlc += 1
        branch_cnt += 1
        if x == 'L':
            num_ind += 1
    elif x == 'V':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_v += 1
        branch_cnt += 1
    elif x == 'I':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_i += 1
        branch_cnt += 1
    elif x == 'O':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_opamps += 1
    elif x == 'E':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vcvs += 1
        branch_cnt += 1
    elif x == 'G':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vccs += 1
        branch_cnt += 1
    elif x == 'F':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_cccs += 1
        branch_cnt += 1
    elif x == 'H':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_ccvs += 1
        branch_cnt += 1
    elif x == 'K':
        if (tk_cnt != 4):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_cpld_ind += 1
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))


# ## parser
# - puts branch elements into structure
# - counts number of nodes
# 
# data frame lables:
# - count: data frame index
# - element: type of element
# - p node: positive node
# - n node: negitive node, for a current source, the arrow terminal
# - cp node: controlling positive node of branch
# - cn node: controlling negitive node of branch
# - Vout: opamp output node
# - value: value of element or voltage
# - Vname: voltage source through which the controlling current flows. Need to add a zero volt voltage source to the controlling branch.
# - Lname1: name of coupled inductor 1
# - Lname2: name of coupled inductor 2
# 

# build the pandas data frame
df = pd.DataFrame(columns=['element','p node','n node','cp node','cn node',
    'Vout','value','Vname','Lname1','Lname2'])

# this data frame is for branches with unknown currents, need better name
#df2 = pd.DataFrame(columns=['element','p node','n node'])


# ### functions to load branch elements into data frame
# 

# loads voltage or current sources into branch structure
def indep_source(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

# loads passive elements into branch structure
def rlc_element(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

# loads multi-terminal elements into branch structure
# O - Op Amps
def opamp_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vout'] = int(tk[3])

# G - VCCS
def vccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

# E - VCVS
def vcvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

# F - CCCS
def cccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

# H - CCVS
def ccvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

# K - Coupled inductors
def cpld_ind_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'Lname1'] = tk[1].capitalize()
    df.loc[line_nu,'Lname2'] = tk[2].capitalize()
    df.loc[line_nu,'value'] = float(tk[3])


# function to scan df and get largest node number
def count_nodes():
    # need to check that nodes are consecutive
    # fill array with node numbers
    p = np.zeros(line_cnt+1)
    for i in range(line_cnt-1):
        p[df['p node'][i]] = df['p node'][i]
        p[df['n node'][i]] = df['n node'][i]

    # find the largest node number
    if df['n node'].max() > df['p node'].max():
        largest = df['n node'].max()
    else:
        largest =  df['p node'].max()

    largest = int(largest)
    # check for unfilled elements, skip node 0
    for i in range(1,largest):
        if p[i] == 0:
            print('nodes not in continuous order, node {:.0f} is missing'.format(p[i-1]+1))

    return largest


# load branch info into data frame
for i in range(line_cnt):
    x = content[i][0]

    if (x == 'R') or (x == 'L') or (x == 'C'):
        rlc_element(i)
    elif (x == 'V') or (x == 'I'):
        indep_source(i)
    elif x == 'O':
        opamp_sub_network(i)
    elif x == 'E':
        vcvs_sub_network(i)
    elif x == 'G':
        vccs_sub_network(i)
    elif x == 'F':
        cccs_sub_network(i)
    elif x == 'H':
        ccvs_sub_network(i)
    elif x == 'K':
        cpld_ind_sub_network(i)
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))

# count number of nodes
num_nodes = count_nodes()


# ### new function
# don't need, delete when debugging is done.
# 

# print a report
print('Net list report')
print('number of lines in netlist: {:d}'.format(line_cnt))
print('number of branches: {:d}'.format(branch_cnt))
print('number of nodes: {:d}'.format(num_nodes))
# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
print('number of unknown currents: {:d}'.format(i_unk))
print('number of passive components: {:d}'.format(num_rlc))
print('number of inductors: {:d}'.format(num_ind))
print('number of independent voltage sources: {:d}'.format(num_v))
print('number of independent current sources: {:d}'.format(num_i))
print('number of op amps: {:d}'.format(num_opamps))

# not implemented yet
print('\nNot implemented yet')
print('number of E - VCVS: {:d}'.format(num_vcvs))
print('number of G - VCCS: {:d}'.format(num_vccs))
print('number of F - CCCS: {:d}'.format(num_cccs))
print('number of H - CCVS: {:d}'.format(num_ccvs))
print('number of K - Coupled inductors: {:d}'.format(num_cpld_ind))


df


# store the data frame as a pickle file
# df.to_pickle(fn+'.pkl')  # <- uncomment if needed


# initialize some symbolic matrix with zeros
# A is formed by [[G, C] [B, D]]
# Z = [I,E]
# X = [V, J]
V = zeros(num_nodes,1)
I = zeros(num_nodes,1)
G = zeros(num_nodes,num_nodes)  # also called Yr, the reduced nodal matrix
s = Symbol('s')  # the Laplace variable

# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
# is is possible to have i_unk == 0 ?, what about a network with only current sources?
i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind+num_cccs
if i_unk != 0:
    B = zeros(num_nodes,i_unk)
    C = zeros(i_unk,num_nodes)
    D = zeros(i_unk,i_unk)
    E = zeros(i_unk,1)
    J = zeros(i_unk,1)


# ## G matrix
# The G matrix is n by n and is determined by the interconnections between Rs and Cs.  The G matrix is formed in three steps:  
# 1) Each element in the diagonal matrix is equal to the sum of the conductance (one over the resistance) of each element connected to the corresponding node.  So the first diagonal element is the sum of conductances connected to node 1, the second diagonal element is the sum of conductances connected to node 2, and so on.  
# 2) The off diagonal elements are the negative conductance of the element connected to the pair of corresponding node.  Therefore a resistor between nodes 1 and 2 goes into the G matrix at location (1,2) and locations (2,1).  
# 3) Add vccs element type according to the elemet stamp rule.  (insert discription of the rule)
# 
# **Notes:**  
# In the orginal paper G is called Yr, where Yr, is a reduced form of the nodal matrix excluding the contributions due to voltage sources, current controlling elements, etc.
# 
# <span style="color:red">\----need to check on inductor treatment, doesn't verify with LTspice testing, inductor stamp affects the B,C and D arrays</span>
# 

# G matrix
for i in range(len(df)):  # process each row in the data frame
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    cn1 = df.loc[i,'cp node']
    cn2 = df.loc[i,'cn node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'R':
        g = 1/sympify(df.loc[i,'element'])
#    if x == 'L':
#        g = 1/s/sympify(df.loc[i,'element'])  # this matches Eric's code, but I thinks is wrong
    if x == 'C':
        g = s*sympify(df.loc[i,'element'])
    if x == 'G':   #vccs type element
        g = sympify(df.loc[i,'element'].lower())  # use a symbol for gain value

    if (x == 'R') or (x == 'C'):   # fix this don't do L's <----
    #if (x == 'R') or (x == 'L') or (x == 'C'):   # fix this don't do L's <----
        # If neither side of the element is connected to ground
        # then subtract it from appropriate location in matrix.
        if (n1 != 0) and (n2 != 0):
            G[n1-1,n2-1] += -g
            G[n2-1,n1-1] += -g

        # If node 1 is connected to ground, add element to diagonal of matrix
        if n1 != 0:
            G[n1-1,n1-1] += g

        # same for for node 2
        if n2 != 0:
            G[n2-1,n2-1] += g

    if x == 'G':    #vccs type element
        # check to see if any terminal is grounded
        # then stamp the matrix
        if n1 != 0 and cn1 != 0:
            G[n1-1,cn1-1] += g

        if n2 != 0 and cn2 != 0:
            G[n2-1,cn2-1] += g

        if n1 != 0 and cn2 != 0:
            G[n1-1,cn2-1] -= g

        if n2 != 0 and cn1 != 0:
            G[n2-1,cn1-1] -= g

G  # display the G matrix


# ## B Matrix
# Rules for making the B matrix
# The B matrix is an n by m matrix with only 0, 1 and -1 elements.  There is one column for each unknown current.  loop through all the branches and process elements that have stamps for the B matrix:  
# V: voltage sources, O: opamps, H: ccvs, F: cccs, E: vcvs and inductors, these are counted in variable i_unk  
# The of the columns is as they appear in the netlist  
# F: cccs does not get its own column because the controlling current is through a zero volt voltage source, called Vname  <- not true?  cccs is an unknown current
# 
# ```
# old notes  
# Each location in the matrix corresponds to an unknown current particular voltage source (first dimension) or a node (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a -1.  Otherwise, elements of the B matrix are zero.
# 
# coding notes:
# 
# not only voltage sources but controlled sources like cccs
# 
# probably need some code to make sure number of column equals the number of voltage sources.
# 
# number of columns = num_v+num_opamps+num_ccvs+num_cccs+num_vcvs ;5 element types, every column is a current i sub k
# if num_v_sources > 1, the B is n by m, otherwise B is n by 1.
# 
# B[row, column]
# 
# Is there a valid case for not having a B matrix, i_unk = 0? 
# Is there a valid op amp case where  B is n by 1?
# 
# 
# loop through all the branches and process elements that have stamps for the B matrix
# V: voltage sources, O: opamps, H: ccvs, F: cccs, E: vcvs and inductors, these are counted in variable i_unk
# The of the columns is as they appear in the netlist
# F: cccs does not get its own column because the controlling current is through a zero volt voltage source, called Vname  <- not true?  cccs is an unknown current
# ```
# 

i=6
#df.loc[i,'element']
#df.loc[i,'cp node'] # nodes for controlled sources
#df.loc[i,'cn node']
#df.loc[i,'Vout'] # node connected to op amp output
find_vname(df.loc[i,'Vname'])
#df.loc[i,'p node']
#df.loc[i,'n node']


i_unk


sn


# generate the B Matrix
sn = 0   # count source number as code walks through the data frame
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    #cn1 = df.loc[i,'cp node'] # nodes for controlled sources
    #cn2 = df.loc[i,'cn node']
    n_vout = df.loc[i,'Vout'] # node connected to op amp output

    # process elements with input to B matrix
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if i_unk > 1:  #is B greater than 1 by n?, V
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'O':  # op amp type
        B[n_vout-1,sn] = 1
        sn += 1   # increment source count
    if (x == 'H') or (x == 'F'):  # H: ccvs, F: cccs,
        if i_unk > 1:  #is B greater than 1 by n?, H, F
            # check to see if any terminal is grounded
            # then stamp the matrix
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
            # need to find the vn for Vname
            # for H, maybe don't need this because a V source is included in the netlist  <---
#            vn1, vn2, col_num = find_vname(df.loc[i,'Vname'])
#            if vn2 != 0:
#                B[col_num-1,vn1] = 1 # need to fix this, not cn
#            if vn1 != 0:
#                B[col_num-1,vn2] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'E':   # vcvs type, only ik column is altered at n1 and n2
        if i_unk > 1:  #is B greater than 1 by n?, E
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count

    if x == 'L':
        if i_unk > 1:  #is B greater than 1 by n?, L
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count

# check source count
if sn != i_unk:
    print('source number not equal to i_unk in matrix B')

B   # display the B matrix


# ## C matrix
# 
# 
# vcvc code need to be fixed, the controlling node gets and entry
# ~~~
# old notes
# The C matrix is an m by n matrix with only 0, 1 and -1 elements (except for controlled sources).  Each location in the matrix corresponds to a particular node (first dimension) or voltage source (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a -1.  Otherwise, elements of the C matrix are zero.
# 
# <span style="color:red">C matrix needs to be fixed</span>
# 
# Follow the B matric example  
# copied code from B matrix, changing B to C
# swapping index
# ~~~
# 

# generate the C Matrix
sn = 0   # count source number as code walks through the data frame
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    #cn1 = df.loc[i,'cp node'] # nodes for controlled sources
    #cn2 = df.loc[i,'cn node']
    n_vout = df.loc[i,'Vout'] # node connected to op amp output

    # process elements with input to B matrix
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if i_unk > 1:  #is B greater than 1 by n?, V
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1
        sn += 1   #increment source count
    if x == 'O':  # op amp type
        C[sn,n_vout-1] = 1
        sn += 1   # increment source count
    if (x == 'H') or (x == 'F'):  # H: ccvs, F: cccs,
        if i_unk > 1:  #is B greater than 1 by n?, H, F
            # check to see if any terminal is grounded
            # then stamp the matrix
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
            # need to find the vn for Vname
            # for H, maybe don't need this because a V source is included in the netlist  <---
#            vn1, vn2, col_num = find_vname(df.loc[i,'Vname'])
#            if vn2 != 0:
#                B[col_num-1,vn1] = 1 # need to fix this, not cn
#            if vn1 != 0:
#                B[col_num-1,vn2] = -1
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1
        sn += 1   #increment source count
    if x == 'E':   # vcvs type, only ik column is altered at n1 and n2
        if i_unk > 1:  #is B greater than 1 by n?, E
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1
        sn += 1   #increment source count

    if x == 'L':
        if i_unk > 1:  #is B greater than 1 by n?, L
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1
        sn += 1   #increment source count

# check source count
if sn != i_unk:
    print('source number not equal to i_unk in matrix C')

C   # display the C matrix





# ## D matrix
# The D matrix is an mxm matrix, where m is the number of unknown currents.
# > m = i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind+num_cccs  
# 
# Stamps that affect the D matrix are: inductor, ccvs and cccs
# 

# generate the D Matrix
sn = 0   # count source number as code walks through the data frame
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    #cn1 = df.loc[i,'cp node'] # nodes for controlled sources
    #cn2 = df.loc[i,'cn node']
    #n_vout = df.loc[i,'Vout'] # node connected to op amp output

    # process elements with input to B matrix
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'L':
        if i_unk > 1:  #is D greater than 1 by 1?
            D[sn,sn] += 1/s/sympify(df.loc[i,'element'])
        else:
            D[sn] += 1/s/sympify(df.loc[i,'element'])
        sn += 1   #increment source count

    if (x == 'H') or (x == 'F'):  # H: ccvs, F: cccs,
        if i_unk > 1:  #is B greater than 1 by n?, H, F
            # check to see if any terminal is grounded
            # then stamp the matrix
            if n1 != 0:
                D[sn,n1-1] += sympify(df.loc[i,'element'].lower())  # use a symbol for gain value
            if n2 != 0:
                D[sn,n2-1] = -1
            # need to find the vn for Vname
            # for H, maybe don't need this because a V source is included in the netlist  <---
#            vn1, vn2, col_num = find_vname(df.loc[i,'Vname'])
#            if vn2 != 0:
#                B[col_num-1,vn1] = 1 # need to fix this, not cn
#            if vn1 != 0:
#                B[col_num-1,vn2] = -1
        else:
            if n1 != 0:
                D[n1-1] += sympify(df.loc[i,'element'].lower())  # use a symbol for gain value
            if n2 != 0:
                D[n2-1] = -1
        sn += 1   #increment source count

# check source count
if sn != i_unk:
    print('source number not equal to i_unk in matrix D')

# display the The D matrix
D


# ## I matrix
# The I matrix is an n by 1 matrix with each element of the matrix corresponding to a particular node.  The value of each element of I is determined by the sum of current sources into the corresponding node.  If there are no current sources connected to the node, the value is zero.
# 

# generate the I matrix, current sources have N2 = arrow end
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'I':
        g = sympify(df.loc[i,'element'])
        # sum the current into each node
        if n1 != 0:
            I[n1-1] -= g
        if n2 != 0:
            I[n2-1] += g

I  # display the I matrix


# ## V matrix
# The V matrixis an nx1 matrix formed of the node voltages.  Each element in V corresponds to the voltage at the equivalent node in the circuit
# 

# generate the V matrix
for i in range(num_nodes):
    V[i] = sympify('v{:d}'.format(i+1))

V  # display the V matrix


# ## J matrix
# The is an m by 1 matrix, with one entry for the current through each voltage source.
# 

# The J matrix is an mx1 matrix, with one entry for the current through each voltage source.
sn = 0   # count source number
oan = 0   #count op amp number
for i in range(len(df)):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        J[sn] = sympify('I_{:s}'.format(df.loc[i,'element']))
        sn += 1
    if x == 'O':  # this needs to be checked <---- needs debugging
        J[oan+num_v] = sympify('I_{:s}'.format(df.loc[i,'element']))
        oan += 1

J  # diplay the J matrix


# ## E matrix
# The E matrix is mx1 and holds the values of the independent voltage sources.
# 

# generate the E matrix
sn = 0   # count source number
for i in range(len(df)):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        E[sn] = sympify(df.loc[i,'element'])
        sn += 1

E   # display the E matrix


# ## Z matrix
# The Z matrix holds the independent voltage and current sources and is the combination of 2 smaller matrices I and E.  The Z matrix is (m+n) by 1, n is the number of nodes, and m is the number of independent voltage sources.  The I matrix is n by 1 and contains the sum of the currents through the passive elements into the corresponding node (either zero, or the sum of independent current sources). The E matrix is m by 1 and holds the values of the independent voltage sources.
# 

Z = I[:] + E[:]
Z  # display the Z matrix


# ## X matrix
# The X matrix is an (n+m) by 1 vector that holds the unknown quantities (node voltages and the currents through the independent voltage sources). The top n elements are the n node voltages. The bottom m elements represent the currents through the m independent voltage sources in the circuit. The V matrix is n by 1 and holds the unknown voltages.  The J matrix is m by 1 and holds the unknown currents through the voltage sources
# 

X = V[:] + J[:]
X  # display the X matrix


# ## A matrix
# The A matrix is (m+n) by (m+n) and will be developed as the combination of 4 smaller matrices, G, B, C, and D.
# 

n = num_nodes
m = num_v+num_opamps
A = zeros(m+n,m+n)
for i in range(n):
    for j in range(n):
        A[i,j] = G[i,j]

if num_v+num_opamps > 1:
    for i in range(n):
        for j in range(m):
            A[i,n+j] = B[i,j]
            A[n+j,i] = C[j,i]
else:
    for i in range(n):
        A[i,n] = B[i]
        A[n,i] = C[i]

A  # display the A matrix


# generate the circuit equations
n = num_nodes
m = num_v+num_opamps
eq_temp = 0  # temporary equation used to build up the equation
equ = zeros(m+n,1)  #initialize the array to hold the equations
for i in range(n+m):
    for j in range(n+m):
        eq_temp += A[i,j]*X[j]
    equ[i] = Eq(eq_temp,Z[i])
    eq_temp = 0

equ   # display the equations


# Use the str() function to convert sympy equations to strings.  These strings can be copid to a new notebook.
# 

str(equ)


str(equ.free_symbols)


str(X)


df








# # A python node analysis jupyter notebook
# 

# **Abstract:** This notebook will read in a spice like circuit netlist file and compute the network equations. These equations can then be copied to a different notebook where the node voltages can be solved using sympy or numpy.
# 
# **Description:** This node analysis code started as a translation from some C code to generate a nodal admittance matrix that I had written in 1988.  The original C code worked well and calculated numeric solutions.  I then started writing some C code to generate the matrices with symbolic values and then intended to use LISP to symbolically solve the equations.  I didn’t get too far with this effort.  The LISP code would generate huge symbolic strings with no simplification.  The output was a big pile of trash that was not in the least bit useful or decipherable.
# 
# In 2014, I started to use python for my little coding projects and engineering calculations.  There are some nice python libraries for numeric and symbolic calculations (such as numpy and sympy), so I decided to try writing a python script to generate the node equations based on the old C code I had written many years before.  Part way into this project I discovered that there is a new nodal analysis technique being taught today in engineering school called the modified nodal analysis (1,2).  The modified nodal analysis provides an algorithmic method for generating systems of independent equations for linear circuit analysis.  Some of my younger colleagues at work were taught this method, but I never heard of it until a short time ago.  These days, I never really analyze a circuit by hand, unless it’s so simple that you can almost do it by inspection.  Most problems that an electrical engineer encounters on the job are complex enough that they use computers to analyze the circuits.  LTspice is the version of spice that I use, since it’s free and does a good job converging when analyzing switching circuits.
# 
# The code follows Erik Cheever's Analysis of  Resistive Circuits [page](http://www.swarthmore.edu/NatSci/echeeve1/Ref/mna/MNA1.html) to generate modified nodal equations. I somewhat followed his matlab file for resistors, capacitors, opamps and independent sources.  The preprocessor and parser code was converted from my old C code.  The use of panda for a data frame is new and sympy is used to do the math.
# 
# After doing some verification testing with inductors and capacitors, it seems that inductors are not being treated correctly.  According to some research, the inductor stamp affects the B,C and D arrays.  Erik Cheever's code puts inductors into the G matrix as 1/s/L.  LTspice results are different than the python code.  Capacitors seem to work OK.
# 
# Reference:
# 1. The modified nodal approach to network analysis, Chung-Wen Ho, A. Ruehli, P. Brennan, IEEE Transactions on Circuits and Systems ( Volume: 22, Issue: 6, Jun 1975 )
# 2. https://en.wikipedia.org/wiki/Modified_nodal_analysis
# 3. ECE 570 Session 3, Computer Aided Engineering for Integrated Circuits, http://www2.engr.arizona.edu/~ece570/session3.pdf
# 
# Some notes from reference 1:
# Capacitances and inductances are considered only in the time domain and their contributions, shown in Table I, are obtained by applying finite differencing methods to their branch relations.
# 

# ```
# Date started: April 17, 2017
# file name: node analysis.ipynb
# Requires: Python version 3 or higher and a jupyter notebook
# Author: Tony
# 
# Revision History
# 7/1/2015: Ver 1 - coding started, derived from network.c code
# 8/18/2017
# changed approach, now implementing a modified nodal analysis
# 8/19/2017
# Wrote some code to generate symbolic matrices, works ok,
# so heading down the sympy path. Basic debugging finished,
# but still need to verify some circuits using Ls and Cs.
# 8/30/2017
# Started to add code for op amps
# 9/1/2017
# Code added to process op amps
# 9/3/2017
# Added code to remove spice directives.
# Fixed orientation of current sources in I matrix.
# N2 is the arrow end of the current source.
# 9/5/2017
# After doing some verification testing with inductors and capacitors,
# it seems that inductors are not being treated correctly.  According
# to some research, inductor stamp affects the B,C and D arrays.  Erik
# Cheever's code puts inductors into the G matrix as 1/s/L.  LTspice 
# results are different than the python code.  Capacitors seem to work OK.
# Plan is to add controlled sources, then get inductors working.
# 9/6/2017
# opamp_test_circuit_426 is not working.  Results not the same as LTspice
# Chebyshev_LPF_1dB_4pole: cut off frequency not correct, other features look OK
# still need to debug opamps and inductors
# Adding: VCCS = G type branch element: G needs to be modified
# CCVS = H type branch element: B, C and D need to be modified
# 
# left off editing at the B matrix
# 
# 9/10/2017
# researching formulation of B matrix
# what about a network with only 1 current source?  The B, C and D matrix would be 0 by 0.
# Think about changing the name of the G matrix to Yr, to keep same as Ho's IEEE paper.
# 
# CCVS = H type branch element: B, C and D need to be modified
# CCCS = F type branch element: B, C and D need to be modified
# VCCS = G type branch element: G needs to be modified
# VCVS = E type branch element: B and C need to be modified
# 
# For CCCS = F type branch element, for this type of element, need to add a zero volt voltage source to the net list through which the current flows.
# For CCVS = H type branch element, need to add a zero volt voltage source to the net list through which the current flows.  The dependent voltage source is already included in the net list as H type.
# 
# 9/12/2017
# still working on the B matrix
# ```
# 

import os
from sympy import *
import numpy as np
import pandas as pd
init_printing()


# initialize some variables, count the types of elements
num_rlc = 0 # number of passive elements
num_ind = 0 # number of inductors
num_v = 0    # number of independent voltage sources
num_i = 0    # number of independent current sources
i_unk = 0  # number of current unknowns
num_opamps = 0   # number of op amps
num_vcvs = 0     # number of controlled sources of various types
num_vccs = 0
num_cccs = 0
num_ccvs = 0
num_cpld_ind = 0 # number of coupled inductors


# ##### open file and preprocess it, file name extenstion is defaulted to .net
# - remove blank lines and comments
# - convert first letter of element name to upper case
# - removes extra spaces between entries
# - count number of entries on each line, make sure the count is correct
# 

fn = 'example48'
fd1 = open(fn+'.net','r')
content = fd1.readlines()
content = [x.strip() for x in content]  #remove leading and trailing white space
# remove empty lines
while '' in content:
    content.pop(content.index(''))

# remove comment lines, these start with a asterisk *
content = [n for n in content if not n.startswith('*')]
# remove other comment lines, these start with a semicolon ;
content = [n for n in content if not n.startswith(';')]
# remove spice directives, these start with a period, .
content = [n for n in content if not n.startswith('.')]
# converts 1st letter to upper case
#content = [x.upper() for x in content] <- this converts all to upper case
content = [x.capitalize() for x in content]
# removes extra spaces between entries
content = [' '.join(x.split()) for x in content]


line_cnt = len(content) # number of lines in the netlist
branch_cnt = 0  # number of btanches in the netlist
# check number of entries on each line
for i in range(line_cnt):
    x = content[i][0]
    tk_cnt = len(content[i].split()) # split the line into tokens

    if (x == 'R') or (x == 'L') or (x == 'C'):
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_rlc += 1
        branch_cnt += 1
        if x == 'L':
            num_ind += 1
    elif x == 'V':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_v += 1
        branch_cnt += 1
    elif x == 'I':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_i += 1
        branch_cnt += 1
    elif x == 'O':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_opamps += 1
    elif x == 'E':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vcvs += 1
        branch_cnt += 1
    elif x == 'G':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vccs += 1
        branch_cnt += 1
    elif x == 'F':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_cccs += 1
        branch_cnt += 1
    elif x == 'H':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_ccvs += 1
        branch_cnt += 1
    elif x == 'K':
        if (tk_cnt != 4):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_cpld_ind += 1
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))


# ##### parser
# - puts branch elements into structure
# - counts number of nodes
# 
# data frame lables:
# - count: data frame index
# - element: type of element
# - p node: positive node
# - n node: negitive node, for a current source, the arrow terminal
# - cp node: controlling positive node of branch
# - cn node: controlling negitive node of branch
# - Vout: opamp output node
# - value: value of element or voltage
# - Vname: voltage source through which the controlling current flows. Need to add a zero volt voltage source to the controlling branch.
# - Lname1: name of coupled inductor 1
# - Lname2: name of coupled inductor 2
# 

# build the pandas data frame
df = pd.DataFrame(columns=['element','p node','n node','cp node','cn node',
    'Vout','value','Vname','Lname1','Lname2'])

# this data frame is for branches with unknown currents, need better name
df2 = pd.DataFrame(columns=['element','p node','n node'])


# ##### functions to load branch elements into data frame
# 

# loads voltage or current sources into branch structure
def indep_source(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

# loads passive elements into branch structure
def rlc_element(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

# loads multi-terminal elements into branch structure
# O - Op Amps
def opamp_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vout'] = int(tk[3])

# G - VCCS
def vccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

# E - VCVS
def vcvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

# F - CCCS
def cccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

# H - CCVS
def ccvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

# K - Coupled inductors
def cpld_ind_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'Lname1'] = tk[1].capitalize()
    df.loc[line_nu,'Lname2'] = tk[2].capitalize()
    df.loc[line_nu,'value'] = float(tk[3])


# function to scan df and get largest node number
def count_nodes():
    # need to check that nodes are consecutive
    # fill array with node numbers
    p = np.zeros(line_cnt+1)
    for i in range(line_cnt-1):
        p[df['p node'][i]] = df['p node'][i]
        p[df['n node'][i]] = df['n node'][i]

    # find the largest node number
    if df['n node'].max() > df['p node'].max():
        largest = df['n node'].max()
    else:
        largest =  df['p node'].max()

    largest = int(largest)
    # check for unfilled elements, skip node 0
    for i in range(1,largest):
        if p[i] == 0:
            print('nodes not in continuous order, node {:.0f} is missing'.format(p[i-1]+1))

    return largest


# load branch info into data frame
for i in range(line_cnt):
    x = content[i][0]

    if (x == 'R') or (x == 'L') or (x == 'C'):
        rlc_element(i)
    elif (x == 'V') or (x == 'I'):
        indep_source(i)
    elif x == 'O':
        opamp_sub_network(i)
    elif x == 'E':
        vcvs_sub_network(i)
    elif x == 'G':
        vccs_sub_network(i)
    elif x == 'F':
        cccs_sub_network(i)
    elif x == 'H':
        ccvs_sub_network(i)
    elif x == 'K':
        cpld_ind_sub_network(i)
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))

# count number of nodes
num_nodes = count_nodes()


# ### new function
# 

# This is a function to generate a data frame of branches that have current unknowns
# this is needed to be able to find the column number and nodes for controlled sources
def func1():  # need a better name
    count = 0
    # need to walk through data frame and find these parameters
    for i in range(len(df)):
        n1 = df.loc[i,'p node']
        n2 = df.loc[i,'n node']

        # process all the elements creating unknown currents
        x = df.loc[i,'element'][0]   #get 1st letter of element name
        if (x == 'L') or (x == 'V') or (x == 'O') or (x == 'E') or (x == 'H'):
            df2.loc[count,'element'] = df.loc[i,'element']
            df2.loc[count,'p node'] = df.loc[i,'p node']
            df2.loc[count,'n node'] = df.loc[i,'n node']
            count += 1


func1()


df2


# print a report
print('Net list report')
print('number of lines in netlist: {:d}'.format(line_cnt))
print('number of branches: {:d}'.format(branch_cnt))
print('number of nodes: {:d}'.format(num_nodes))
# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
print('number of unknown currents: {:d}'.format(i_unk))
print('number of passive components: {:d}'.format(num_rlc))
print('number of inductors: {:d}'.format(num_ind))
print('number of independent voltage sources: {:d}'.format(num_v))
print('number of independent current sources: {:d}'.format(num_i))
print('number of op amps: {:d}'.format(num_opamps))

# not implemented yet
print('\nNot implemented yet')
print('number of E - VCVS: {:d}'.format(num_vcvs))
print('number of G - VCCS: {:d}'.format(num_vccs))
print('number of F - CCCS: {:d}'.format(num_cccs))
print('number of H - CCVS: {:d}'.format(num_ccvs))
print('number of K - Coupled inductors: {:d}'.format(num_cpld_ind))


df


# store the data frame as a pickle file
# df.to_pickle(fn+'.pkl')  # <- uncomment if needed


# initialize some symbolic matrix with zeros
# A is formed by [[G, C] [B, D]]
# Z = [I,E]
# X = [V, J]
V = zeros(num_nodes,1)
I = zeros(num_nodes,1)
G = zeros(num_nodes,num_nodes)  # also called Yr, the reduced nodal matrix
s = Symbol('s')  # the Laplace variable

# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
# is is possible to have i_unk == 0 ?, what about a network with only current sources?
i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
if i_unk != 0:
    B = zeros(num_nodes,i_unk)
    C = zeros(i_unk,num_nodes)
    D = zeros(i_unk,i_unk)
    E = zeros(i_unk,1)
    J = zeros(i_unk,1)


# ##### G matrix 
# 
# <span style="color:red">\----need to check on inductor treatment, doesn't verify with LTspice testing, inductor stamp affects the B,C and D arrays</span>
# 
# The G matrix is n by n and is determined by the interconnections between the passive circuit elements (RLC's).  The G matrix is an nxn matrix formed in two steps:
# 1. Each element in the diagonal matrix is equal to the sum of the conductance (one over the resistance) of each element connected to the corresponding node.  So the first diagonal element is the sum of conductances connected to node 1, the second diagonal element is the sum of conductances connected to node 2, and so on.
# 2. The off diagonal elements are the negative conductance of the element connected to the pair of corresponding node.  Therefore a resistor between nodes 1 and 2 goes into the G matrix at location (1,2) and locations (2,1).
# 
# Adding VCCS, G type element
# 
# In the orginal paper B is called Yr, where Yr, is a reduced form of the nodal matrix excluding the contributions due to voltage sources, current controlling elements, etc.
# 

# G matrix
for i in range(len(df)):  # don't use branch count use # of rows in data frame
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    cn1 = df.loc[i,'cp node']
    cn2 = df.loc[i,'cn node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'R':
        g = 1/sympify(df.loc[i,'element'])
#    if x == 'L':
#        g = 1/s/sympify(df.loc[i,'element'])  # this matches Eric's code, but I thinks is wrong
    if x == 'C':
        g = s*sympify(df.loc[i,'element'])
    if x == 'G':   #vccs type element
        g = sympify(df.loc[i,'element'].lower())  # use a symbol for gain value

    if (x == 'R') or (x == 'C'):   # fix this don't do L's <----
    #if (x == 'R') or (x == 'L') or (x == 'C'):   # fix this don't do L's <----
        # If neither side of the element is connected to ground
        # then subtract it from appropriate location in matrix.
        if (n1 != 0) and (n2 != 0):
            G[n1-1,n2-1] += -g
            G[n2-1,n1-1] += -g

        # If node 1 is connected to ground, add element to diagonal of matrix
        if n1 != 0:
            G[n1-1,n1-1] += g

        # same for for node 2
        if n2 != 0:
            G[n2-1,n2-1] += g

    if x == 'G':    #vccs type element
        # check to see if any terminal is grounded
        # then stamp the matrix
        if n1 != 0 and cn1 != 0:
            G[n1-1,cn1-1] += g

        if n2 != 0 and cn2 != 0:
            G[n2-1,cn2-1] += g

        if n1 != 0 and cn2 != 0:
            G[n1-1,cn2-1] -= g

        if n2 != 0 and cn1 != 0:
            G[n2-1,cn1-1] -= g

G  # display the G matrix


# ##### B Matrix
# Rules for making the B matrix
# The B matrix is an n by m matrix with only 0, 1 and -1 elements.  Each location in the matrix corresponds to a particular voltage source (first dimension) or a node (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a -1.  Otherwise, elements of the B matrix are zero.
# 
# coding notes:
# 
# not only voltage sources but controlled sources like cccs
# 
# probably need some code to make sure number of column equals the number of voltage sources.
# 
# number of columns = num_v+num_opamps+num_ccvs+num_cccs+num_vcvs ;5 element types, every column is a current i sub k
# if num_v_sources > 1, the B is n by m, otherwise B is n by 1.
# 
# B[row, column]
# 
# Is there a valid case for not having a B matrix, i_unk = 0? 
# Is there a valide op amp case where  B is n by 1?
# 
# 
# loop through all the branches and process elements that have stamps for the B matrix
# V: voltage sources, O: opamps, H: ccvs, F: cccs, E: vcvs and inductors, these are counted in variable i_unk
# The of the columns is as they appear in the netlist
# F: cccs does not get its own column because the controlling current is through a zero volt voltage source, called Vname
# 

# need n1, n2 and column number for B
# build a list of i_unk ?
# given element name
# find the colum number and return the node numbers
# 

# find the the column position in the B matrix
def find_vname(name):
    # need to walk through data frame and find these parameters
    for i in range(len(df2)):
        n1 = df2.loc[i,'p node']
        n2 = df2.loc[i,'n node']

        # process all the elements creating unknown currents
        if name == df2.loc[i,'element']:
            return n1, n2, i+1  # vn1, vn2, col_num

    print('failed to find matching branch element in find_vname')


i=6
#df.loc[i,'element']
#df.loc[i,'cp node'] # nodes for controlled sources
#df.loc[i,'cn node']
#df.loc[i,'Vout'] # node connected to op amp output
find_vname(df.loc[i,'Vname'])
#df.loc[i,'p node']
#df.loc[i,'n node']


# generate the B Matrix
sn = 0   # count source number as code walks through the data frame
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    cn1 = df.loc[i,'cp node'] # nodes for controlled sources
    cn2 = df.loc[i,'cn node']
    n_vout = df.loc[i,'Vout'] # node connected to op amp output

    # process elements with input to B matrix
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if i_unk > 1:  #is B greater than 1 by n?
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'O':  # op amp type
        B[n_vout-1,sn] = 1
        sn += 1   # increment source count
    if (x == 'H') or (x == 'F'):  # H: ccvs, F: cccs,
        if i_unk > 1:  #is B greater than 1 by n?
            # check to see if any terminal is grounded
            # then stamp the matrix
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
            # need to find the vn for Vname
            vn1, vn2, col_num = find_vname(df.loc[i,'Vname'])
            if vn2 != 0:
                B[col_num-1,vn1] = 1 # need to fix this, not cn
            if vn1 != 0:
                B[col_num-1,vn2] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'E':   # vcvs type, only ik column is altered at n1 and n2
        if i_unk > 1:  #is B greater than 1 by n?
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'L':
        if i_unk > 1:  #is B greater than 1 by n?
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count

# check source count
if sn != i_unk:
    print('source number not equal to i_unk in matrix B')

B   # display the B matrix


# ##### C matrix
# The C matrix is an m by n matrix with only 0, 1 and -1 elements (except for controlled sources).  Each location in the matrix corresponds to a particular node (first dimension) or voltage source (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a -1.  Otherwise, elements of the C matrix are zero.
# 
# <span style="color:red">C matrix needs to be fixed</span>
# 

# generate the C matrix
sn = 0   # count source number
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    n_vout = df.loc[i,'Vout'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        if n1 != 0:
            C[oan+num_v,n1-1] = 1
        if n2 != 0:
            C[oan+num_v,n2-1] = -1
        oan += 1  # increment op amp number

C   # display the C matrix


# ##### D matrix
# The D matrix is an mxm matrix that is composed entirely of zeros.  (It can be non-zero if controlled sources are considered.)
# 

# display the The D matrix
D


# ##### I matrix
# The I matrix is an n by 1 matrix with each element of the matrix corresponding to a particular node.  The value of each element of I is determined by the sum of current sources into the corresponding node.  If there are no current sources connected to the node, the value is zero.
# 

# generate the I matrix, current sources have N2 = arrow end
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'I':
        g = sympify(df.loc[i,'element'])
        # sum the current into each node
        if n1 != 0:
            I[n1-1] -= g
        if n2 != 0:
            I[n2-1] += g

I  # display the I matrix


# ##### V matrix
# The V matrixis an nx1 matrix formed of the node voltages.  Each element in V corresponds to the voltage at the equivalent node in the circuit
# 

# generate the V matrix
for i in range(num_nodes):
    V[i] = sympify('v{:d}'.format(i+1))

V  # display the V matrix


# ##### J matrix
# The is an m by 1 matrix, with one entry for the current through each voltage source.
# 

# The J matrix is an mx1 matrix, with one entry for the current through each voltage source.
sn = 0   # count source number
oan = 0   #count op amp number
for i in range(len(df)):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        J[sn] = sympify('I_{:s}'.format(df.loc[i,'element']))
        sn += 1
    if x == 'O':  # this needs to be checked <---- needs debugging
        J[oan+num_v] = sympify('I_{:s}'.format(df.loc[i,'element']))
        oan += 1

J  # diplay the J matrix


# ##### E matrix
# The E matrix is mx1 and holds the values of the independent voltage sources.
# 

# generate the E matrix
sn = 0   # count source number
for i in range(len(df)):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        E[sn] = sympify(df.loc[i,'element'])
        sn += 1

E   # display the E matrix


# ##### Z matrix
# The Z matrix holds the independent voltage and current sources and is the combination of 2 smaller matrices I and E.  The Z matrix is (m+n) by 1, n is the number of nodes, and m is the number of independent voltage sources.  The I matrix is n by 1 and contains the sum of the currents through the passive elements into the corresponding node (either zero, or the sum of independent current sources). The E matrix is m by 1 and holds the values of the independent voltage sources.
# 

Z = I[:] + E[:]
Z  # display the Z matrix


# ##### X matrix
# The X matrix is an (n+m) by 1 vector that holds the unknown quantities (node voltages and the currents through the independent voltage sources). The top n elements are the n node voltages. The bottom m elements represent the currents through the m independent voltage sources in the circuit. The V matrix is n by 1 and holds the unknown voltages.  The J matrix is m by 1 and holds the unknown currents through the voltage sources
# 

X = V[:] + J[:]
X  # display the X matrix


# ##### A matrix
# The A matrix is (m+n) by (m+n) and will be developed as the combination of 4 smaller matrices, G, B, C, and D.
# 

n = num_nodes
m = num_v+num_opamps
A = zeros(m+n,m+n)
for i in range(n):
    for j in range(n):
        A[i,j] = G[i,j]

if num_v+num_opamps > 1:
    for i in range(n):
        for j in range(m):
            A[i,n+j] = B[i,j]
            A[n+j,i] = C[j,i]
else:
    for i in range(n):
        A[i,n] = B[i]
        A[n,i] = C[i]

A  # display the A matrix


# generate the circuit equations
n = num_nodes
m = num_v+num_opamps
eq_temp = 0  # temporary equation used to build up the equation
equ = zeros(m+n,1)  #initialize the array to hold the equations
for i in range(n+m):
    for j in range(n+m):
        eq_temp += A[i,j]*X[j]
    equ[i] = Eq(eq_temp,Z[i])
    eq_temp = 0

equ   # display the equations


# Use the str() function to convert sympy equations to strings.  These strings can be copid to a new notebook.
# 

str(equ)


str(equ.free_symbols)


str(X)


df








# # A python node analysis jupyter notebook
# 

# **Abstract:** This notebook will read in a spice like circuit netlist file and compute the network equations. These equations can then be copied to a different notebook where the node voltages can be solved using sympy or numpy.
# 
# **Description:** This node analysis code started as a translation from some C code to generate a nodal admittance matrix that I had written in 1988.  The original C code worked well and calculated numeric solutions.  I then started writing some C code to generate the matrices with symbolic values and then intended to use LISP to symbolically solve the equations.  I didn’t get too far with this effort.  The LISP code would generate huge symbolic strings with no simplification.  The output was a big pile of trash that was not in the least bit useful or decipherable.
# 
# In 2014, I started to use python for my little coding projects and engineering calculations.  There are some nice python libraries for numeric and symbolic calculations (such as numpy and sympy), so I decided to try writing a python script to generate the node equations based on the old C code I had written many years before.  Part way into this project I discovered that there is a new nodal analysis technique being taught today in engineering school called the modified nodal analysis (1,2).  The modified nodal analysis provides an algorithmic method for generating systems of independent equations for linear circuit analysis.  Some of my younger colleagues at work were taught this method, but I never heard of it until a short time ago.  These days, I never really analyze a circuit by hand, unless it’s so simple that you can almost do it by inspection.  Most problems that an electrical engineer encounters on the job are complex enough that they use computers to analyze the circuits.  LTspice is the version of spice that I use, since it’s free and does a good job converging when analyzing switching circuits.
# 
# The code follows Erik Cheever's Analysis of  Resistive Circuits [page](http://www.swarthmore.edu/NatSci/echeeve1/Ref/mna/MNA1.html) to generate modified nodal equations. I somewhat followed his matlab file for resistors, capacitors, opamps and independent sources.  The preprocessor and parser code was converted from my old C code.  The use of panda for a data frame is new and sympy is used to do the math.
# 
# After doing some verification testing with inductors and capacitors, it seems that inductors are not being treated correctly.  According to some research, the inductor stamp affects the B,C and D arrays.  Erik Cheever's code puts inductors into the G matrix as 1/s/L.  LTspice results are different than the python code.  Capacitors seem to work OK.
# 
# Reference:
# 1. The modified nodal approach to network analysis, Chung-Wen Ho, A. Ruehli, P. Brennan, IEEE Transactions on Circuits and Systems ( Volume: 22, Issue: 6, Jun 1975 )
# 2. https://en.wikipedia.org/wiki/Modified_nodal_analysis
# 3. ECE 570 Session 3, Computer Aided Engineering for Integrated Circuits, http://www2.engr.arizona.edu/~ece570/session3.pdf
# 
# Some notes from reference 1:
# Capacitances and inductances are considered only in the time domain and their contributions, shown in Table I, are obtained by applying finite differencing methods to their branch relations.
# 

# ```
# Date started: April 17, 2017
# file name: node analysis.ipynb
# Requires: Python version 3 or higher and a jupyter notebook
# Author: Tony
# 
# Revision History
# 7/1/2015: Ver 1 - coding started, derived from network.c code
# 8/18/2017
# changed approach, now implementing a modified nodal analysis
# 8/19/2017
# Wrote some code to generate symbolic matrices, works ok,
# so heading down the sympy path. Basic debugging finished,
# but still need to verify some circuits using Ls and Cs.
# 8/30/2017
# Started to add code for op amps
# 9/1/2017
# Code added to process op amps
# 9/3/2017
# Added code to remove spice directives.
# Fixed orientation of current sources in I matrix.
# N2 is the arrow end of the current source.
# 9/5/2017
# After doing some verification testing with inductors and capacitors,
# it seems that inductors are not being treated correctly.  According
# to some research, inductor stamp affects the B,C and D arrays.  Erik
# Cheever's code puts inductors into the G matrix as 1/s/L.  LTspice 
# results are different than the python code.  Capacitors seem to work OK.
# Plan is to add controlled sources, then get inductors working.
# 9/6/2017
# opamp_test_circuit_426 is not working.  Results not the same as LTspice
# Chebyshev_LPF_1dB_4pole: cut off frequency not correct, other features look OK
# still need to debug opamps and inductors
# Adding: VCCS = G type branch element: G needs to be modified
# CCVS = H type branch element: B, C and D need to be modified
# 
# left off editing at the B matrix
# 
# 9/10/2017
# researching formulation of B matrix
# what about a network with only 1 current source?
# 
# 
# CCVS = H type branch element: B, C and D need to be modified
# CCCS = F type branch element: B, C and D need to be modified
# VCCS = G type branch element: G needs to be modified
# VCVS = E type branch element: B and C need to be modified
# 
# For CCCS = F type branch element, for this type of element, need to add a zero volt voltage source to the net list through which the current flows.
# For CCVS = H type branch element, need to add a zero volt voltage source to the net list through which the current flows.  The dependent voltage source is already included in the net list as H type.
# ```

import os
from sympy import *
import numpy as np
import pandas as pd
init_printing()


# initialize some variables, count the types of elements
num_rlc = 0 # number of passive elements
num_ind = 0 # number of inductors
num_v = 0    # number of independent voltage sources
num_i = 0    # number of independent current sources
i_unk = 0  # number of current unknowns
num_opamps = 0   # number of op amps
num_vcvs = 0     # number of controlled sources of various types
num_vccs = 0
num_cccs = 0
num_ccvs = 0
num_cpld_ind = 0 # number of coupled inductors


# ##### open file and preprocess it, file name extenstion is defaulted to .net
# - remove blank lines and comments
# - convert first letter of element name to upper case
# - removes extra spaces between entries
# - count number of entries on each line, make sure the count is correct
# 

fn = 'example48'
fd1 = open(fn+'.net','r')
content = fd1.readlines()
content = [x.strip() for x in content]  #remove leading and trailing white space
# remove empty lines
while '' in content:
    content.pop(content.index(''))

# remove comment lines, these start with a asterisk *
content = [n for n in content if not n.startswith('*')]
# remove other comment lines, these start with a semicolon ;
content = [n for n in content if not n.startswith(';')]
# remove spice directives, these start with a period, .
content = [n for n in content if not n.startswith('.')]
# converts 1st letter to upper case
#content = [x.upper() for x in content] <- this converts all to upper case
content = [x.capitalize() for x in content]
# removes extra spaces between entries
content = [' '.join(x.split()) for x in content]


line_cnt = len(content) # number of lines in the netlist
branch_cnt = 0  # number of btanches in the netlist
# check number of entries on each line
for i in range(line_cnt):
    x = content[i][0]
    tk_cnt = len(content[i].split()) # split the line into tokens

    if (x == 'R') or (x == 'L') or (x == 'C'):
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_rlc += 1
        branch_cnt += 1
        if x == 'L':
            num_ind += 1
    elif x == 'V':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_v += 1
        branch_cnt += 1
    elif x == 'I':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_i += 1
        branch_cnt += 1
    elif x == 'O':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_opamps += 1
    elif x == 'E':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vcvs += 1
        branch_cnt += 1
    elif x == 'G':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vccs += 1
        branch_cnt += 1
    elif x == 'F':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_cccs += 1
        branch_cnt += 1
    elif x == 'H':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_ccvs += 1
        branch_cnt += 1
    elif x == 'K':
        if (tk_cnt != 4):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_cpld_ind += 1
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))


# ##### parser
# - puts branch elements into structure
# - counts number of nodes
# 
# data frame lables:
# - count: data frame index
# - element: type of element
# - p node: positive node
# - n node: negitive node, for a current source, the arrow terminal
# - cp node: controlling positive node of branch
# - cn node: controlling negitive node of branch
# - Vout: opamp output node
# - value: value of element or voltage
# - Vname: voltage source through which the controlling current flows. Need to add a zero volt voltage source to the controlling branch.
# - Lname1: name of coupled inductor 1
# - Lname2: name of coupled inductor 2
# 

# build the pandas data frame
df = pd.DataFrame(columns=['element','p node','n node','cp node','cn node',
    'Vout','value','Vname','Lname1','Lname2'])

# this data frame is for branches with unknown currents
df2 = pd.DataFrame(columns=['element','p node','n node'])


# ##### functions to load branch elements into data frame
# 

# loads voltage or current sources into branch structure
def indep_source(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

# loads passive elements into branch structure
def rlc_element(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

# loads multi-terminal elements into branch structure
# O - Op Amps
def opamp_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vout'] = int(tk[3])

# G - VCCS
def vccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

# E - VCVS
def vcvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

# F - CCCS
def cccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

# H - CCVS
def ccvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

# K - Coupled inductors
def cpld_ind_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'Lname1'] = tk[1].capitalize()
    df.loc[line_nu,'Lname2'] = tk[2].capitalize()
    df.loc[line_nu,'value'] = float(tk[3])


# function to scan df and get largest node number
def count_nodes():
    # need to check that nodes are consecutive
    # fill array with node numbers
    p = np.zeros(line_cnt+1)
    for i in range(line_cnt-1):
        p[df['p node'][i]] = df['p node'][i]
        p[df['n node'][i]] = df['n node'][i]

    # find the largest node number
    if df['n node'].max() > df['p node'].max():
        largest = df['n node'].max()
    else:
        largest =  df['p node'].max()

    largest = int(largest)
    # check for unfilled elements, skip node 0
    for i in range(1,largest):
        if p[i] == 0:
            print('nodes not in continuous order, node {:.0f} is missing'.format(p[i-1]+1))

    return largest


# load branch info into data frame
for i in range(line_cnt):
    x = content[i][0]

    if (x == 'R') or (x == 'L') or (x == 'C'):
        rlc_element(i)
    elif (x == 'V') or (x == 'I'):
        indep_source(i)
    elif x == 'O':
        opamp_sub_network(i)
    elif x == 'E':
        vcvs_sub_network(i)
    elif x == 'G':
        vccs_sub_network(i)
    elif x == 'F':
        cccs_sub_network(i)
    elif x == 'H':
        ccvs_sub_network(i)
    elif x == 'K':
        cpld_ind_sub_network(i)
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))

# count number of nodes
num_nodes = count_nodes()


# ### new function
# 

# function to enumerate current unknowns which have input to B matrix
# need to be able to fine the column number and nodes for controlled sources

# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
#i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
#unk_currents
def func1():  # need a better name
    count = 0  # is this count needed?
    # need to walk through data frame and find these parameters
    for i in range(len(df)):
        n1 = df.loc[i,'p node']
        n2 = df.loc[i,'n node']

        # process all the elements creating unknown currents
        x = df.loc[i,'element'][0]   #get 1st letter of element name
        if (x == 'L') or (x == 'V') or (x == 'O') or (x == 'E') or (x == 'H'):
            df2.loc[count,'element'] = df.loc[i,'element']
            df2.loc[count,'p node'] = df.loc[i,'p node']
            df2.loc[count,'n node'] = df.loc[i,'n node']
            count += 1


func1()


df2


# print a report
print('Net list report')
print('number of lines in netlist: {:d}'.format(line_cnt))
print('number of branches: {:d}'.format(branch_cnt))
print('number of nodes: {:d}'.format(num_nodes))
# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
print('number of unknown currents: {:d}'.format(i_unk))
print('number of passive components: {:d}'.format(num_rlc))
print('number of inductors: {:d}'.format(num_ind))
print('number of independent voltage sources: {:d}'.format(num_v))
print('number of independent current sources: {:d}'.format(num_i))
print('number of op amps: {:d}'.format(num_opamps))

# not implemented yet
print('\nNot implemented yet')
print('number of E - VCVS: {:d}'.format(num_vcvs))
print('number of G - VCCS: {:d}'.format(num_vccs))
print('number of F - CCCS: {:d}'.format(num_cccs))
print('number of H - CCVS: {:d}'.format(num_ccvs))
print('number of K - Coupled inductors: {:d}'.format(num_cpld_ind))


df


# store the data frame as a pickle file
# df.to_pickle(fn+'.pkl')  # <- uncomment if needed


# initialize some symbolic matrix with zeros
# A is formed by [[G, C] [B, D]]
# Z = [I,E]
# X = [V, J]
V = zeros(num_nodes,1)
I = zeros(num_nodes,1)
G = zeros(num_nodes,num_nodes)  # also called Yr, the reduced nodal matrix
s = Symbol('s')  # the Laplace variable

# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
# is is possible to have i_unk == 0 ?, what about a network with only current sources?
i_unk = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
if i_unk != 0:
    B = zeros(num_nodes,i_unk)
    C = zeros(i_unk,num_nodes)
    D = zeros(i_unk,i_unk)
    E = zeros(i_unk,1)
    J = zeros(i_unk,1)


# ##### G matrix <span style="color:red">\----need to check on inductor treatment, doesn't verify with LTspice testing, inductor stamp affects the B,C and D arrays</span>
# The G matrix is n by n and is determined by the interconnections between the passive circuit elements (RLC's).  The G matrix is an nxn matrix formed in two steps:
# 1. Each element in the diagonal matrix is equal to the sum of the conductance (one over the resistance) of each element connected to the corresponding node.  So the first diagonal element is the sum of conductances connected to node 1, the second diagonal element is the sum of conductances connected to node 2, and so on.
# 2. The off diagonal elements are the negative conductance of the element connected to the pair of corresponding node.  Therefore a resistor between nodes 1 and 2 goes into the G matrix at location (1,2) and locations (2,1).
# 
# Adding VCCS, G type element
# 
# In the orginal paper B is called Yr, where Yr, is a reduced form of the nodal matrix excluding the contributions due to voltage sources, current controlling elements, etc.
# 

# G matrix
for i in range(len(df)):  # don't use branch count use # of rows in data frame
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    cn1 = df.loc[i,'cp node']
    cn2 = df.loc[i,'cn node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'R':
        g = 1/sympify(df.loc[i,'element'])
#    if x == 'L':
#        g = 1/s/sympify(df.loc[i,'element'])  # this matches Eric's code, but I thinks is wrong
    if x == 'C':
        g = s*sympify(df.loc[i,'element'])
    if x == 'G':   #vccs type element
        g = sympify(df.loc[i,'element'].lower())  # use a symbol for gain value

    if (x == 'R') or (x == 'C'):   # fix this don't do L's <----
    #if (x == 'R') or (x == 'L') or (x == 'C'):   # fix this don't do L's <----
        # If neither side of the element is connected to ground
        # then subtract it from appropriate location in matrix.
        if (n1 != 0) and (n2 != 0):
            G[n1-1,n2-1] += -g
            G[n2-1,n1-1] += -g

        # If node 1 is connected to ground, add element to diagonal of matrix
        if n1 != 0:
            G[n1-1,n1-1] += g

        # same for for node 2
        if n2 != 0:
            G[n2-1,n2-1] += g

    if x == 'G':    #vccs type element
        # check to see if any terminal is grounded
        # then stamp the matrix
        if n1 != 0 and cn1 != 0:
            G[n1-1,cn1-1] += g

        if n2 != 0 and cn2 != 0:
            G[n2-1,cn2-1] += g

        if n1 != 0 and cn2 != 0:
            G[n1-1,cn2-1] -= g

        if n2 != 0 and cn1 != 0:
            G[n2-1,cn1-1] -= g

G  # display the G matrix


# ##### B Matrix
# Rules for making the B matrix
# The B matrix is an n by m matrix with only 0, 1 and -1 elements.  Each location in the matrix corresponds to a particular voltage source (first dimension) or a node (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a -1.  Otherwise, elements of the B matrix are zero.
# 
# coding notes:
# 
# not only voltage sources but controlled sources like cccs
# 
# probably need some code to make sure number of column equals the number of voltage sources.
# 
# number of columns = num_v+num_opamps+num_ccvs+num_cccs+num_vcvs ;5 element types, every column is a current i sub k
# if num_v_sources > 1, the B is n by m, otherwise B is n by 1.
# 
# B[row, column]
# 
# Is there a valid case for not having a B matrix, i_unk = 0? 
# Is there a valide op amp case where  B is n by 1?
# 

# find the the column position in the B matrix
def find_vnam(name):
    # need to walk through data frame and find these parameters
    for i in range(len(df2)):
        n1 = df2.loc[i,'p node']
        n2 = df2.loc[i,'n node']

        # process all the elements creating unknown currents
        if name == df2.loc[i,'element']:
            return n1, n2, i+1  # vn1, vn2, col_num


    print('failed to find matching branch element in find_vname')


# need n1, n2 and column number for B
# build a list of i_unk ?
# given element name
# find the colum number and return the node numbers
# 

i=6
#df.loc[i,'element']
#df.loc[i,'cp node'] # nodes for controlled sources
#df.loc[i,'cn node']
#df.loc[i,'Vout'] # node connected to op amp output
find_vnam(df.loc[i,'Vname'])
#df.loc[i,'p node']
#df.loc[i,'n node']


# generate the B Matrix
# loop through all the branches and process elements that have stamps for the B matrix
# V: voltage sources, O: opamps, H: ccvs, F: cccs, E: vcvs and inductors, these are counted in variable i_unk
# The of the columns is as they appear in the netlist
# F: cccs does not get its own column because the controlling current is through a zero volt voltage source, called Vname
sn = 0   # count source number as code walks through the data frame
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    cn1 = df.loc[i,'cp node'] # nodes for controlled sources
    cn2 = df.loc[i,'cn node']
    n_vout = df.loc[i,'Vout'] # node connected to op amp output

    # process elements with input to B matrix
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if i_unk > 1:  #is B greater than 1 by n?
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'O':  # op amp type
        B[n_vout-1,sn] = 1
        sn += 1   # increment source count
    if (x == 'H') or (x == 'F'):  # H: ccvs, F: cccs,
        if i_unk > 1:  #is B greater than 1 by n?
            # check to see if any terminal is grounded
            # then stamp the matrix
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
            # need to find the vn for Vname
            vn1, vn2, col_num = find_vnam(df.loc[i,'Vname'])
            if vn2 != 0:
                B[col_num-1,vn1] = 1 # need to fix this, not cn
            if vn1 != 0:
                B[col_num-1,vn2] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'E':   # vcvs type, only ik column is altered at n1 and n2
        if i_unk > 1:  #is B greater than 1 by n?
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count
    if x == 'L':
        if i_unk > 1:  #is B greater than 1 by n?
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1
        sn += 1   #increment source count

# check source count
if sn != i_unk:
    print('source number not equal to i_unk in matrix B')

B   # display the B matrix


# ##### C matrix
# The C matrix is an m by n matrix with only 0, 1 and -1 elements (except for controlled sources).  Each location in the matrix corresponds to a particular node (first dimension) or voltage source (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a -1.  Otherwise, elements of the C matrix are zero.
# 
# <span style="color:red">C matrix needs to be fixed</span>
# 

# generate the C matrix
sn = 0   # count source number
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    n_vout = df.loc[i,'Vout'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        if n1 != 0:
            C[oan+num_v,n1-1] = 1
        if n2 != 0:
            C[oan+num_v,n2-1] = -1
        oan += 1  # increment op amp number

C   # display the C matrix


# ##### D matrix
# The D matrix is an mxm matrix that is composed entirely of zeros.  (It can be non-zero if controlled sources are considered.)
# 

# display the The D matrix
D


# ##### I matrix
# The I matrix is an n by 1 matrix with each element of the matrix corresponding to a particular node.  The value of each element of I is determined by the sum of current sources into the corresponding node.  If there are no current sources connected to the node, the value is zero.
# 

# generate the I matrix, current sources have N2 = arrow end
for i in range(len(df)):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'I':
        g = sympify(df.loc[i,'element'])
        # sum the current into each node
        if n1 != 0:
            I[n1-1] -= g
        if n2 != 0:
            I[n2-1] += g

I  # display the I matrix


# ##### V matrix
# The V matrixis an nx1 matrix formed of the node voltages.  Each element in V corresponds to the voltage at the equivalent node in the circuit
# 

# generate the V matrix
for i in range(num_nodes):
    V[i] = sympify('v{:d}'.format(i+1))

V  # display the V matrix


# ##### J matrix
# The is an m by 1 matrix, with one entry for the current through each voltage source.
# 

# The J matrix is an mx1 matrix, with one entry for the current through each voltage source.
sn = 0   # count source number
oan = 0   #count op amp number
for i in range(len(df)):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        J[sn] = sympify('I_{:s}'.format(df.loc[i,'element']))
        sn += 1
    if x == 'O':  # this needs to be checked <---- needs debugging
        J[oan+num_v] = sympify('I_{:s}'.format(df.loc[i,'element']))
        oan += 1

J  # diplay the J matrix


# ##### E matrix
# The E matrix is mx1 and holds the values of the independent voltage sources.
# 

# generate the E matrix
sn = 0   # count source number
for i in range(len(df)):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        E[sn] = sympify(df.loc[i,'element'])
        sn += 1

E   # display the E matrix


# ##### Z matrix
# The Z matrix holds the independent voltage and current sources and is the combination of 2 smaller matrices I and E.  The Z matrix is (m+n) by 1, n is the number of nodes, and m is the number of independent voltage sources.  The I matrix is n by 1 and contains the sum of the currents through the passive elements into the corresponding node (either zero, or the sum of independent current sources). The E matrix is m by 1 and holds the values of the independent voltage sources.
# 

Z = I[:] + E[:]
Z  # display the Z matrix


# ##### X matrix
# The X matrix is an (n+m) by 1 vector that holds the unknown quantities (node voltages and the currents through the independent voltage sources). The top n elements are the n node voltages. The bottom m elements represent the currents through the m independent voltage sources in the circuit. The V matrix is n by 1 and holds the unknown voltages.  The J matrix is m by 1 and holds the unknown currents through the voltage sources
# 

X = V[:] + J[:]
X  # display the X matrix


# ##### A matrix
# The A matrix is (m+n) by (m+n) and will be developed as the combination of 4 smaller matrices, G, B, C, and D.
# 

n = num_nodes
m = num_v+num_opamps
A = zeros(m+n,m+n)
for i in range(n):
    for j in range(n):
        A[i,j] = G[i,j]

if num_v+num_opamps > 1:
    for i in range(n):
        for j in range(m):
            A[i,n+j] = B[i,j]
            A[n+j,i] = C[j,i]
else:
    for i in range(n):
        A[i,n] = B[i]
        A[n,i] = C[i]

A  # display the A matrix


# generate the circuit equations
n = num_nodes
m = num_v+num_opamps
eq_temp = 0  # temporary equation used to build up the equation
equ = zeros(m+n,1)  #initialize the array to hold the equations
for i in range(n+m):
    for j in range(n+m):
        eq_temp += A[i,j]*X[j]
    equ[i] = Eq(eq_temp,Z[i])
    eq_temp = 0

equ   # display the equations


# Use the str() function to convert sympy equations to strings.  These strings can be copid to a new notebook.
# 

str(equ)


str(equ.free_symbols)


str(X)


df








# # A python node analysis jupyter notebook
# 

# **Synopsis:** This notebook will read in a spice like circuit netlist file and compute the node equations. The code follows Erik Cheever's Analysis of  Resistive Circuits [page](http://www.swarthmore.edu/NatSci/echeeve1/Ref/mna/MNA1.html) to generate modified nodal equations. I somewhat followed his matlab file.
# 
# **Description:**
# 

# ```
# Date started: April 17, 2017
# file name: node analysis.ipynb
# Requires: Python version 3 or higher and a jupyter notebook
# Author: Tony
# 
# Revision History
# 7/1/2015: Ver 1 - coding started, derived from network.c code
# 8/18/2017
# changed approach, now implementing a modified nodal analysis
# 8/19/2017
# Wrote some code to generate symbolic matrices, works ok,
# so heading down the sympy path. Basic debugging finished,
# but still need to verify some circuits using Ls and Cs.
# 8/30/2017
# Started to add code for op amps
# 9/1/2017
# Code added to process op amps
# 9/3/2017
# Added code to remove spice directives.
# Fixed orientation of current sources in I matrix.
# N2 is the arrow end of the current source.
# 9/4/2017
# 
# ```
# 

import os
from sympy import *
import numpy as np
import pandas as pd
init_printing()


# initialize some variables, count the types of elements
num_rlc = 0 # number of passive elements
num_ind = 0 # number of inductors
num_v = 0    # number of independent voltage sources
num_i = 0    # number of independent current sources
num_opamps = 0   # number of op amps
num_vcvs = 0     # number of controlled sources of various types
num_vccs = 0
num_cccs = 0
num_ccvs = 0
num_cpld_ind = 0 # number of coupled inductors


# ##### open file and preprocess it, file name extenstion is defaulted to .net
# - remove blank lines and comments
# - convert first letter of element name to upper case
# - removes extra spaces between entries
# - count number of entries on each line, make sure the count is correct
# 

fn = 'example-rc'
fd1 = open(fn+'.net','r')
content = fd1.readlines()
content = [x.strip() for x in content]  #remove leading and trailing white space
# remove empty lines
while '' in content:
    content.pop(content.index(''))

# remove comment lines, these start with a asterisk *
content = [n for n in content if not n.startswith('*')]
# remove other comment lines, these start with a semicolon ;
content = [n for n in content if not n.startswith(';')]
# remove spice directives, these start with a period, .
content = [n for n in content if not n.startswith('.')]
# converts 1st letter to upper case
#content = [x.upper() for x in content] <- this converts all to upper case
content = [x.capitalize() for x in content]
# removes extra spaces between entries
content = [' '.join(x.split()) for x in content]


line_cnt = len(content) # number of lines in the netlist
branch_cnt = 0  # number of btanches in the netlist
# check number of entries on each line
for i in range(line_cnt):
    x = content[i][0]
    tk_cnt = len(content[i].split())

    if (x == 'R') or (x == 'L') or (x == 'C'):
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_rlc += 1
        branch_cnt += 1
        if x == 'L':
            num_ind += 1
    elif x == 'V':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_v += 1
        branch_cnt += 1
    elif x == 'I':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_i += 1
        branch_cnt += 1
    elif x == 'O':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_opamps += 1
    elif x == 'E':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vcvs += 1
        branch_cnt += 1
    elif x == 'G':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vccs += 1
        branch_cnt += 1
    elif x == 'F':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_cccs += 1
        branch_cnt += 1
    elif x == 'H':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_ccvs += 1
        branch_cnt += 1
    elif x == 'K':
        if (tk_cnt != 4):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_cpld_ind += 1
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))


# ##### parser
# - puts branch elements into structure
# - counts number of nodes
# 
# data frame lables:
# - count: data frame index
# - element: type of element
# - p node: positive node
# - n node: negitive node, for a current source, the arrow terminal
# - cp node: controlling positive node of branch
# - cn node: controlling negitive node of branch
# - Vout: opamp output node
# - value: value of element or voltage
# - Vname: voltage source through which the controlling current flows. Need to add a zero volt voltage source to the controlling branch.
# - Lname1: name of coupled inductor 1
# - Lname2: name of coupled inductor 2
# 
# ```
# temp code delete later
# count = []        # data frame index
# element = []      # type of element
# p_node = []       # positive node
# n_node = []       # neg node, for a current source, the arrow terminal
# cp_node = []      # controlling positive node of branch
# cn_node = []      # controlling negitive node of branch
# Vout = []         # op amp output node
# value = []        # value of element or voltage
# v_name = []       # voltage source through which the controlling current flows
# l_name1 = []      # name of coupled inductor 1
# l_name2 = []      # name of coupled inductor 2
# df = pd.DataFrame(index=count, columns=['element','p node','n node','cp node','cn node',
#     'Vout','value','Vname','Lname1','Lname2'])
# ```
# 

# build the pandas data frame
df = pd.DataFrame(columns=['element','p node','n node','cp node','cn node',
    'Vout','value','Vname','Lname1','Lname2'])


# ##### functions to load branch elements into data frame
# 

# loads voltage or current sources into branch structure
def indep_source(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

# loads passive elements into branch structure
def rlc_element(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

'''
loads multi-terminal elements into branch structure
Types:
E - VCVS
G - VCCS
F - CCCS
H - CCVS
K - Coupled inductors
O - Op Amps
'''
def opamp_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vout'] = int(tk[3])

def vccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

def vcvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

def cccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

def ccvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

def cpld_ind_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'Lname1'] = tk[1].capitalize()
    df.loc[line_nu,'Lname2'] = tk[2].capitalize()
    df.loc[line_nu,'value'] = float(tk[3])


# function to scan df and get largest node number
def count_nodes():
    # need to check that nodes are consecutive
    # fill array with node numbers
    p = np.zeros(line_cnt+1)
    for i in range(line_cnt-1):
        p[df['p node'][i]] = df['p node'][i]
        p[df['n node'][i]] = df['n node'][i]

    # find the largest node number
    if df['n node'].max() > df['p node'].max():
        largest = df['n node'].max()
    else:
        largest =  df['p node'].max()

    largest = int(largest)
    # check for unfilled elements, skip node 0
    for i in range(1,largest):
        if p[i] == 0:
            print('nodes not in continuous order, node {:.0f} is missing'.format(p[i-1]+1))

    return largest


# load branch info into data frame
for i in range(line_cnt):
    x = content[i][0]

    if (x == 'R') or (x == 'L') or (x == 'C'):
        rlc_element(i)
    elif (x == 'V') or (x == 'I'):
        indep_source(i)
    elif x == 'O':
        opamp_sub_network(i)
    elif x == 'E':
        vcvs_sub_network(i)
    elif x == 'G':
        vccs_sub_network(i)
    elif x == 'F':
        cccs_sub_network(i)
    elif x == 'H':
        ccvs_sub_network(i)
    elif x == 'K':
        cpld_ind_sub_network(i)
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))

# count number of nodes
num_nodes = count_nodes()


# print a report
print('Net list report')
print('number of lines in netlist: {:d}'.format(line_cnt))
print('number of branches: {:d}'.format(branch_cnt))
print('number of nodes: {:d}'.format(num_nodes))
print('number of passive components: {:d}'.format(num_rlc))
print('number of inductors: {:d}'.format(num_ind))
print('number of independent voltage sources: {:d}'.format(num_v))
print('number of independent current sources: {:d}'.format(num_i))
print('number of op amps: {:d}'.format(num_opamps))

# not implemented yet
print('\nNot implemented yet')
print('number of E - VCVS: {:d}'.format(num_vcvs))
print('number of G - VCCS: {:d}'.format(num_vccs))
print('number of F - CCCS: {:d}'.format(num_cccs))
print('number of F - CCCS: {:d}'.format(num_ccvs))
print('number of K - Coupled inductors: {:d}'.format(num_cpld_ind))


content


df


# store the data frame as a pickle file
# df.to_pickle(fn+'.pkl')  # <- uncomment if needed


# initialize some symbolic matrix with zeros
# A is formed by [[G, C] [B, D]]
# Z = [I,E]
# X = [V, J]
V = zeros(num_nodes,1)
I = zeros(num_nodes,1)
G = zeros(num_nodes,num_nodes)
s = Symbol('s')  # the Laplace variable

# count the number of element types that affect the size of the B, C, D, E and J arrays
k = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
if (num_v+num_opamps) != 0:
    B = zeros(num_nodes,num_v+num_opamps)
    C = zeros(num_v+num_opamps,num_nodes)
    D = zeros(num_v+num_opamps,num_v+num_opamps)
    E = zeros(num_v+num_opamps,1)
    J = zeros(num_v+num_opamps,1)


# ##### G matrix                  <span style="color:red">\----need to check on inductor treatment, dosen't verify with LTspice testing, inductor stamp affects the B,C and D arrays</span>
# The G matrix is n by n and is determined by the interconnections between the passive circuit elements (RLC's).  The G matrix is an nxn matrix formed in two steps:
# 1. Each element in the diagonal matrix is equal to the sum of the conductance (one over the resistance) of each element connected to the corresponding node.  So the first diagonal element is the sum of conductances connected to node 1, the second diagonal element is the sum of conductances connected to node 2, and so on.
# 2. The off diagonal elements are the negative conductance of the element connected to the pair of corresponding node.  Therefore a resistor between nodes 1 and 2 goes into the G matrix at location (1,2) and locations (2,1).
# 

# G matrix
for i in range(branch_cnt):  # don't use branch count use # of rows in data frame
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'R':
        g = 1/sympify(df.loc[i,'element'])
    if x == 'L':
        g = 1/(s*sympify(df.loc[i,'element']))
    if x == 'C':
        g = sympify(df.loc[i,'element'])*s

    if (x == 'R') or (x == 'L') or (x == 'C'):
        # If neither side of the element is connected to ground
        # then subtract it from appropriate location in matrix.
        if (n1 != 0) and (n2 != 0):
            G[n1-1,n2-1] += -g
            G[n2-1,n1-1] += -g

        # If node 1 is connected to ground, add element to diagonal of matrix
        if n1 != 0:
            G[n1-1,n1-1] += g

        # same for for node 2
        if n2 != 0:
            G[n2-1,n2-1] += g

G  # display the G matrix


# ##### I matrix
# The I matrix is an n by 1 matrix with each element of the matrix corresponding to a particular node.  The value of each element of I is determined by the sum of current sources into the corresponding node.  If there are no current sources connected to the node, the value is zero.
# 

# generate the I matrix, current sources have N2 = arrow end
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'I':
        g = sympify(df.loc[i,'element'])
        # sum the current into each node
        if n1 != 0:
            I[n1-1] -= g
        if n2 != 0:
            I[n2-1] += g

I  # display the I matrix


# ##### V matrix
# The V matrixis an nx1 matrix formed of the node voltages.  Each element in V corresponds to the voltage at the equivalent node in the circuit
# 

# generate the V matrix
for i in range(num_nodes):
    V[i] = sympify('v{:d}'.format(i+1))

V  # display the V matrix


# ##### B Matrix
# Rules for making the B matrix
# The B matrix is an n by m matrix with only 0, 1 and -1 elements (except for controlled sources).  Each location in the matrix corresponds to a particular voltage source (first dimension) or a node (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a -1.  Otherwise, elements of the B matrix are zero.
# 

# generate the B Matrix
# loop through all the branches and process independent voltage sources
sn = 0   # count source number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(branch_cnt):
    n_vout = df.loc[i,'Vout'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        B[n_vout-1,oan+num_v] = 1
        oan += 1   # increment op amp count

B   # display the B matrix


# ##### J matrix
# The is an m by 1 matrix, with one entry for the current through each voltage source.
# 

# The J matrix is an mx1 matrix, with one entry for the current through each voltage source.
sn = 0   # count source number
oan = 0   #count op amp number
for i in range(branch_cnt):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        J[sn] = sympify('I_{:s}'.format(df.loc[i,'element']))
        sn += 1
    if x == 'O':  # this needs to be checked <---- needs debugging
        J[oan+num_v] = sympify('I_{:s}'.format(df.loc[i,'element']))
        oan += 1

J  # diplay the J matrix


# ##### C matrix
# The C matrix is an m by n matrix with only 0, 1 and -1 elements (except for controlled sources).  Each location in the matrix corresponds to a particular node (first dimension) or voltage source (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a -1.  Otherwise, elements of the C matrix are zero.
# 

# generate the C matrix
sn = 0   # count source number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    n_vout = df.loc[i,'Vout'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        if n1 != 0:
            C[oan+num_v,n1-1] = 1
        if n2 != 0:
            C[oan+num_v,n2-1] = -1
        oan += 1  # increment op amp number

C   # display the C matrix


# ##### D matrix
# The D matrix is an mxm matrix that is composed entirely of zeros.  (It can be non-zero if controlled sources are considered.)
# 

# display the The D matrix
D


# ##### E matrix
# The E matrix is mx1 and holds the values of the independent voltage sources.
# 

# generate the E matrix
sn = 0   # count source number
for i in range(branch_cnt):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        E[sn] = sympify(df.loc[i,'element'])
        sn += 1

E   # display the E matrix


# ##### Z matrix
# The Z matrix holds the independent voltage and current sources and is the combination of 2 smaller matrices I and E.  The Z matrix is (m+n) by 1, n is the number of nodes, and m is the number of independent voltage sources.  The I matrix is n by 1 and contains the sum of the currents through the passive elements into the corresponding node (either zero, or the sum of independent current sources). The E matrix is m by 1 and holds the values of the independent voltage sources.
# 

Z = I[:] + E[:]
Z  # display the Z matrix


# ##### X matrix
# The X matrix is an (n+m) by 1 vector that holds the unknown quantities (node voltages and the currents through the independent voltage sources). The top n elements are the n node voltages. The bottom m elements represent the currents through the m independent voltage sources in the circuit. The V matrix is n by 1 and holds the unknown voltages.  The J matrix is m by 1 and holds the unknown currents through the voltage sources
# 

X = V[:] + J[:]
X  # display the X matrix


# ##### A matrix
# The A matrix is (m+n) by (m+n) and will be developed as the combination of 4 smaller matrices, G, B, C, and D.
# 

n = num_nodes
m = num_v+num_opamps
A = zeros(m+n,m+n)
for i in range(n):
    for j in range(n):
        A[i,j] = G[i,j]

if num_v+num_opamps > 1:
    for i in range(n):
        for j in range(m):
            A[i,n+j] = B[i,j]
            A[n+j,i] = C[j,i]
else:
    for i in range(n):
        A[i,n] = B[i]
        A[n,i] = C[i]

A  # display the A matrix


# generate the circuit equations
n = num_nodes
m = num_v+num_opamps
eq_temp = 0  # temporary equation used to build up the equation
equ = zeros(m+n,1)  #initialize the array to hold the equations
for i in range(n+m):
    for j in range(n+m):
        eq_temp += A[i,j]*X[j]
    equ[i] = Eq(eq_temp,Z[i])
    eq_temp = 0

equ   # display the equations


# Use the str() function to convert sympy equations to strings.  These strings can be copid to a new notebook.
# 

str(equ)


str(equ.free_symbols)


str(X)


df








# # A python node analysis jupyter notebook
# 

# **Synopsis:** This notebook will read in a spice like circuit netlist file and compute the node equations. The code follows Erik Cheever's Analysis of  Resistive Circuits [page](http://www.swarthmore.edu/NatSci/echeeve1/Ref/mna/MNA1.html) to generate modified nodal equations. I somewhat followed his matlab file.
# 
# **Description:**
# 

# ```
# Date started: April 17, 2017
# file name: node analysis.ipynb
# Requires: Python version 3 or higher and a jupyter notebook
# Author: Tony
# 
# Revision History
# 7/1/2015: Ver 1 - coding started, derived from network.c code
# 8/18/2017
# changed approach, now implementing a modified nodal analysis
# 8/19/2017
# Wrote some code to generate symbolic matrices, works ok,
# so heading down the sympy path. Basic debugging finished,
# but still need to verify some circuits using Ls and Cs.
# 8/30/2017
# Started to add code for op amps
# 9/1/2017
# Code added to process op amps
# 9/3/2017
# Added code to remove spice directives.
# Fixed orientation of current sources in I matrix.
# N2 is the arrow end of the current source.
# 9/4/2017
# 
# ```
# 

import os
from sympy import *
import numpy as np
import pandas as pd
init_printing()


# initialize some variables, count the types of elements
num_rlc = 0 # number of passive elements
num_ind = 0 # number of inductors
num_v = 0    # number of independent voltage sources
num_i = 0    # number of independent current sources
num_opamps = 0   # number of op amps
num_vcvs = 0     # number of controlled sources of various types
num_vccs = 0
num_cccs = 0
num_ccvs = 0
num_cpld_ind = 0 # number of coupled inductors


# ##### open file and preprocess it, file name extenstion is defaulted to .net
# - remove blank lines and comments
# - convert first letter of element name to upper case
# - removes extra spaces between entries
# - count number of entries on each line, make sure the count is correct
# 

fn = 'RCL circuit'
fd1 = open(fn+'.net','r')
content = fd1.readlines()
content = [x.strip() for x in content]  #remove leading and trailing white space
# remove empty lines
while '' in content:
    content.pop(content.index(''))

# remove comment lines, these start with a asterisk *
content = [n for n in content if not n.startswith('*')]
# remove other comment lines, these start with a semicolon ;
content = [n for n in content if not n.startswith(';')]
# remove spice directives, these start with a period, .
content = [n for n in content if not n.startswith('.')]
# converts 1st letter to upper case
#content = [x.upper() for x in content] <- this converts all to upper case
content = [x.capitalize() for x in content]
# removes extra spaces between entries
content = [' '.join(x.split()) for x in content]


line_cnt = len(content) # number of lines in the netlist
branch_cnt = 0  # number of btanches in the netlist
# check number of entries on each line
for i in range(line_cnt):
    x = content[i][0]
    tk_cnt = len(content[i].split())

    if (x == 'R') or (x == 'L') or (x == 'C'):
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_rlc += 1
        branch_cnt += 1
        if x == 'L':
            num_ind += 1
    elif x == 'V':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_v += 1
        branch_cnt += 1
    elif x == 'I':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_i += 1
        branch_cnt += 1
    elif x == 'O':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_opamps += 1
    elif x == 'E':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vcvs += 1
        branch_cnt += 1
    elif x == 'G':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vccs += 1
        branch_cnt += 1
    elif x == 'F':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_cccs += 1
        branch_cnt += 1
    elif x == 'H':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_ccvs += 1
        branch_cnt += 1
    elif x == 'K':
        if (tk_cnt != 4):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_cpld_ind += 1
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))


# ##### parser
# - puts branch elements into structure
# - counts number of nodes
# 
# data frame lables:
# - count: data frame index
# - element: type of element
# - p node: positive node
# - n node: negitive node, for a current source, the arrow terminal
# - cp node: controlling positive node of branch
# - cn node: controlling negitive node of branch
# - Vout: opamp output node
# - value: value of element or voltage
# - Vname: voltage source through which the controlling current flows. Need to add a zero volt voltage source to the controlling branch.
# - Lname1: name of coupled inductor 1
# - Lname2: name of coupled inductor 2
# 
# ```
# temp code delete later
# count = []        # data frame index
# element = []      # type of element
# p_node = []       # positive node
# n_node = []       # neg node, for a current source, the arrow terminal
# cp_node = []      # controlling positive node of branch
# cn_node = []      # controlling negitive node of branch
# Vout = []         # op amp output node
# value = []        # value of element or voltage
# v_name = []       # voltage source through which the controlling current flows
# l_name1 = []      # name of coupled inductor 1
# l_name2 = []      # name of coupled inductor 2
# df = pd.DataFrame(index=count, columns=['element','p node','n node','cp node','cn node',
#     'Vout','value','Vname','Lname1','Lname2'])
# ```
# 

# build the pandas data frame
df = pd.DataFrame(columns=['element','p node','n node','cp node','cn node',
    'Vout','value','Vname','Lname1','Lname2'])


# ##### functions to load branch elements into data frame
# 

# loads voltage or current sources into branch structure
def indep_source(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

# loads passive elements into branch structure
def rlc_element(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'value'] = float(tk[3])

'''
loads multi-terminal elements into branch structure
Types:
E - VCVS
G - VCCS
F - CCCS
H - CCVS
K - Coupled inductors
O - Op Amps
'''
def opamp_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vout'] = int(tk[3])

def vccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

def vcvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'cp node'] = int(tk[3])
    df.loc[line_nu,'cn node'] = int(tk[4])
    df.loc[line_nu,'value'] = float(tk[5])

def cccs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

def ccvs_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'p node'] = int(tk[1])
    df.loc[line_nu,'n node'] = int(tk[2])
    df.loc[line_nu,'Vname'] = tk[3].capitalize()
    df.loc[line_nu,'value'] = float(tk[4])

def cpld_ind_sub_network(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu,'element'] = tk[0]
    df.loc[line_nu,'Lname1'] = tk[1].capitalize()
    df.loc[line_nu,'Lname2'] = tk[2].capitalize()
    df.loc[line_nu,'value'] = float(tk[3])


# function to scan df and get largest node number
def count_nodes():
    # need to check that nodes are consecutive
    # fill array with node numbers
    p = np.zeros(line_cnt+1)
    for i in range(line_cnt-1):
        p[df['p node'][i]] = df['p node'][i]
        p[df['n node'][i]] = df['n node'][i]

    # find the largest node number
    if df['n node'].max() > df['p node'].max():
        largest = df['n node'].max()
    else:
        largest =  df['p node'].max()

    largest = int(largest)
    # check for unfilled elements, skip node 0
    for i in range(1,largest):
        if p[i] == 0:
            print('nodes not in continuous order, node {:.0f} is missing'.format(p[i-1]+1))

    return largest


# load branch info into data frame
for i in range(line_cnt):
    x = content[i][0]

    if (x == 'R') or (x == 'L') or (x == 'C'):
        rlc_element(i)
    elif (x == 'V') or (x == 'I'):
        indep_source(i)
    elif x == 'O':
        opamp_sub_network(i)
    elif x == 'E':
        vcvs_sub_network(i)
    elif x == 'G':
        vccs_sub_network(i)
    elif x == 'F':
        cccs_sub_network(i)
    elif x == 'H':
        ccvs_sub_network(i)
    elif x == 'K':
        cpld_ind_sub_network(i)
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))

# count number of nodes
num_nodes = count_nodes()


# print a report
print('Net list report')
print('number of lines in netlist: {:d}'.format(line_cnt))
print('number of branches: {:d}'.format(branch_cnt))
print('number of nodes: {:d}'.format(num_nodes))
print('number of passive components: {:d}'.format(num_rlc))
print('number of inductors: {:d}'.format(num_ind))
print('number of independent voltage sources: {:d}'.format(num_v))
print('number of independent current sources: {:d}'.format(num_i))
print('number of op amps: {:d}'.format(num_opamps))

# not implemented yet
print('\nNot implemented yet')
print('number of E - VCVS: {:d}'.format(num_vcvs))
print('number of G - VCCS: {:d}'.format(num_vccs))
print('number of F - CCCS: {:d}'.format(num_cccs))
print('number of F - CCCS: {:d}'.format(num_ccvs))
print('number of K - Coupled inductors: {:d}'.format(num_cpld_ind))


content


df


# store the data frame as a pickle file
# df.to_pickle(fn+'.pkl')  # <- uncomment if needed


# initialize some symbolic matrix with zeros
# A is formed by [[G, C] [B, D]]
# Z = [I,E]
# X = [V, J]
V = zeros(num_nodes,1)
I = zeros(num_nodes,1)
G = zeros(num_nodes,num_nodes)
s = Symbol('s')  # the Laplace variable

# count the number of element types that affect the size of the B, C, D, E and J arrays
k = num_v+num_opamps+num_vcvs+num_ccvs+num_ind
if (num_v+num_opamps) != 0:
    B = zeros(num_nodes,num_v+num_opamps)
    C = zeros(num_v+num_opamps,num_nodes)
    D = zeros(num_v+num_opamps,num_v+num_opamps)
    E = zeros(num_v+num_opamps,1)
    J = zeros(num_v+num_opamps,1)


# ##### G matrix                  <span style="color:red">\----need to check on inductor treatment, doesn't verify with LTspice testing, inductor stamp affects the B,C and D arrays</span>
# The G matrix is n by n and is determined by the interconnections between the passive circuit elements (RLC's).  The G matrix is an nxn matrix formed in two steps:
# 1. Each element in the diagonal matrix is equal to the sum of the conductance (one over the resistance) of each element connected to the corresponding node.  So the first diagonal element is the sum of conductances connected to node 1, the second diagonal element is the sum of conductances connected to node 2, and so on.
# 2. The off diagonal elements are the negative conductance of the element connected to the pair of corresponding node.  Therefore a resistor between nodes 1 and 2 goes into the G matrix at location (1,2) and locations (2,1).
# 

# G matrix
for i in range(branch_cnt):  # don't use branch count use # of rows in data frame
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'R':
        g = 1/sympify(df.loc[i,'element'])
    if x == 'L':
        g = 1/s/sympify(df.loc[i,'element'])  # this matches Eric's code
    if x == 'C':
        g = s*sympify(df.loc[i,'element'])

    if (x == 'R') or (x == 'L') or (x == 'C'):
        # If neither side of the element is connected to ground
        # then subtract it from appropriate location in matrix.
        if (n1 != 0) and (n2 != 0):
            G[n1-1,n2-1] += -g
            G[n2-1,n1-1] += -g

        # If node 1 is connected to ground, add element to diagonal of matrix
        if n1 != 0:
            G[n1-1,n1-1] += g

        # same for for node 2
        if n2 != 0:
            G[n2-1,n2-1] += g

G  # display the G matrix


# ##### I matrix
# The I matrix is an n by 1 matrix with each element of the matrix corresponding to a particular node.  The value of each element of I is determined by the sum of current sources into the corresponding node.  If there are no current sources connected to the node, the value is zero.
# 

# generate the I matrix, current sources have N2 = arrow end
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'I':
        g = sympify(df.loc[i,'element'])
        # sum the current into each node
        if n1 != 0:
            I[n1-1] -= g
        if n2 != 0:
            I[n2-1] += g

I  # display the I matrix


# ##### V matrix
# The V matrixis an nx1 matrix formed of the node voltages.  Each element in V corresponds to the voltage at the equivalent node in the circuit
# 

# generate the V matrix
for i in range(num_nodes):
    V[i] = sympify('v{:d}'.format(i+1))

V  # display the V matrix


# ##### B Matrix
# Rules for making the B matrix
# The B matrix is an n by m matrix with only 0, 1 and -1 elements (except for controlled sources).  Each location in the matrix corresponds to a particular voltage source (first dimension) or a node (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (i,k) in the B matrix is a -1.  Otherwise, elements of the B matrix are zero.
# 

# generate the B Matrix
# loop through all the branches and process independent voltage sources
sn = 0   # count source number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(branch_cnt):
    n_vout = df.loc[i,'Vout'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        B[n_vout-1,oan+num_v] = 1
        oan += 1   # increment op amp count

B   # display the B matrix


# ##### J matrix
# The is an m by 1 matrix, with one entry for the current through each voltage source.
# 

# The J matrix is an mx1 matrix, with one entry for the current through each voltage source.
sn = 0   # count source number
oan = 0   #count op amp number
for i in range(branch_cnt):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        J[sn] = sympify('I_{:s}'.format(df.loc[i,'element']))
        sn += 1
    if x == 'O':  # this needs to be checked <---- needs debugging
        J[oan+num_v] = sympify('I_{:s}'.format(df.loc[i,'element']))
        oan += 1

J  # diplay the J matrix


# ##### C matrix
# The C matrix is an m by n matrix with only 0, 1 and -1 elements (except for controlled sources).  Each location in the matrix corresponds to a particular node (first dimension) or voltage source (second dimension).  If the positive terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a 1.  If the negative terminal of the ith voltage source is connected to node k, then the element (k,i) in the C matrix is a -1.  Otherwise, elements of the C matrix are zero.
# 

# generate the C matrix
sn = 0   # count source number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    n_vout = df.loc[i,'Vout'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        if n1 != 0:
            C[oan+num_v,n1-1] = 1
        if n2 != 0:
            C[oan+num_v,n2-1] = -1
        oan += 1  # increment op amp number

C   # display the C matrix


# ##### D matrix
# The D matrix is an mxm matrix that is composed entirely of zeros.  (It can be non-zero if controlled sources are considered.)
# 

# display the The D matrix
D


# ##### E matrix
# The E matrix is mx1 and holds the values of the independent voltage sources.
# 

# generate the E matrix
sn = 0   # count source number
for i in range(branch_cnt):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        E[sn] = sympify(df.loc[i,'element'])
        sn += 1

E   # display the E matrix


# ##### Z matrix
# The Z matrix holds the independent voltage and current sources and is the combination of 2 smaller matrices I and E.  The Z matrix is (m+n) by 1, n is the number of nodes, and m is the number of independent voltage sources.  The I matrix is n by 1 and contains the sum of the currents through the passive elements into the corresponding node (either zero, or the sum of independent current sources). The E matrix is m by 1 and holds the values of the independent voltage sources.
# 

Z = I[:] + E[:]
Z  # display the Z matrix


# ##### X matrix
# The X matrix is an (n+m) by 1 vector that holds the unknown quantities (node voltages and the currents through the independent voltage sources). The top n elements are the n node voltages. The bottom m elements represent the currents through the m independent voltage sources in the circuit. The V matrix is n by 1 and holds the unknown voltages.  The J matrix is m by 1 and holds the unknown currents through the voltage sources
# 

X = V[:] + J[:]
X  # display the X matrix


# ##### A matrix
# The A matrix is (m+n) by (m+n) and will be developed as the combination of 4 smaller matrices, G, B, C, and D.
# 

n = num_nodes
m = num_v+num_opamps
A = zeros(m+n,m+n)
for i in range(n):
    for j in range(n):
        A[i,j] = G[i,j]

if num_v+num_opamps > 1:
    for i in range(n):
        for j in range(m):
            A[i,n+j] = B[i,j]
            A[n+j,i] = C[j,i]
else:
    for i in range(n):
        A[i,n] = B[i]
        A[n,i] = C[i]

A  # display the A matrix


# generate the circuit equations
n = num_nodes
m = num_v+num_opamps
eq_temp = 0  # temporary equation used to build up the equation
equ = zeros(m+n,1)  #initialize the array to hold the equations
for i in range(n+m):
    for j in range(n+m):
        eq_temp += A[i,j]*X[j]
    equ[i] = Eq(eq_temp,Z[i])
    eq_temp = 0

equ   # display the equations


# Use the str() function to convert sympy equations to strings.  These strings can be copid to a new notebook.
# 

str(equ)


str(equ.free_symbols)


str(X)


df









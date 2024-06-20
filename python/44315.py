# # A Modular version
# 

# written October 2016 by Sam Greydanus
# 

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

import numpy as np
import gym
import tensorflow as tf


class Actor():
    def __init__(self, n_obs, h, n_actions):
        self.n_obs = n_obs                  # dimensionality of observations
        self.h = h                          # number of hidden layer neurons
        self.n_actions = n_actions          # number of available actions
        
        self.model = model = {}
        with tf.variable_scope('actor_l1',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
            model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
        with tf.variable_scope('actor_l2',reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(h), dtype=tf.float32)
            model['W2'] = tf.get_variable("W2", [h,n_actions], initializer=xavier_l2)
            
    def policy_forward(self, x): #x ~ [1,D]
        h = tf.matmul(x, self.model['W1'])
        h = tf.nn.relu(h)
        logp = tf.matmul(h, self.model['W2'])
        p = tf.nn.softmax(logp)
        return p


class Agent():
    def __init__(self):
        self.gamma = .9             # discount factor for reward
        self.xs, self.rs, self.ys = [],[],[]
        
        self.actor_lr = 1e-2        # learning rate for policy
        self.decay = 0.9
        self.n_obs = n_obs = 4              # dimensionality of observations
        self.n_actions = n_actions = 2          # number of available actions
        
        # make actor part of brain
        self.actor = Actor(n_obs=self.n_obs, h=128, n_actions=self.n_actions)
        
        #placeholders
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs],name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="y")
        self.r = tf.placeholder(dtype=tf.float32, shape=[None,1], name="r")
        
        #gradient processing (PG magic)
        self.discounted_r = self.discount_rewards(self.r, self.gamma)
        mean, variance= tf.nn.moments(self.discounted_r, [0], shift=None, name="reward_moments")
        self.discounted_r -= mean
        self.discounted_r /= tf.sqrt(variance + 1e-6)
        
        # initialize tf graph
        self.aprob = self.actor.policy_forward(self.x)
        self.loss = tf.nn.l2_loss(self.y-self.aprob)
        self.optimizer = tf.train.RMSPropOptimizer(self.actor_lr, decay=self.decay)
        self.grads = self.optimizer.compute_gradients(self.loss,                                     var_list=tf.trainable_variables(), grad_loss=self.discounted_r)
        self.train_op = self.optimizer.apply_gradients(self.grads)

        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
    
    def act(self, x):
        feed = {self.x: x}
        aprob = self.sess.run(self.aprob, feed) ; aprob = aprob[0,:]
        action = np.random.choice(self.n_actions, p=aprob)
        
        label = np.zeros_like(aprob) ; label[action] = 1
        self.xs.append(x)
        self.ys.append(label)
        
        return action
    
    def learn(self):
        epx = np.vstack(self.xs)
        epr = np.vstack(self.rs)
        epy = np.vstack(self.ys)
        self.xs, self.rs, self.ys = [],[],[] # reset game history
        
        feed = {self.x: epx, self.r: epr, self.y: epy}
        _ = self.sess.run(self.train_op,feed) # parameter update
        
    @staticmethod
    def discount_rewards(r, gamma):
        discount_f = lambda a, v: a*gamma + v;
        r_reverse = tf.scan(discount_f, tf.reverse(r,[True, False]))
        discounted_r = tf.reverse(r_reverse,[True, False])
        return discounted_r


def plt_dynamic(x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()


agent = Agent()
env = gym.make("CartPole-v0")
observation = env.reset()
running_reward = 10 # usually starts around 10 for cartpole
reward_sum = 0
episode_number = 0
total_steps = 500


fig,ax = plt.subplots(1,1)
ax.set_xlabel('X') ; ax.set_ylabel('Y')
ax.set_xlim(0,total_steps) ; ax.set_ylim(0,200)
pxs, pys = [], []

print 'episode {}: starting up...'.format(episode_number)
while episode_number <= total_steps and running_reward < 225:
#     if episode_number%25==0: env.render()

    # stochastically sample a policy from the network
    x = observation
    action = agent.act(np.reshape(x, (1,-1)))

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    agent.rs.append(reward)
    reward_sum += reward
    
    if done:
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        agent.learn()

        # visualization
        pxs.append(episode_number)
        pys.append(running_reward)
        if episode_number % 25 == 0:
            print 'ep: {}, reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            plt_dynamic(pxs, pys, ax)
        
        # lame stuff
        episode_number += 1 # the Next Episode
        observation = env.reset() # reset env
        reward_sum = 0
        
plt_dynamic(pxs, pys, ax)
if running_reward > 225:
    print "ep: {}: SOLVED! (running reward hit {} which is greater than 200)".format(
        episode_number, running_reward)





# # A modular version
# 

# written October 2016 by Sam Greydanus
# 

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

import numpy as np
import gym
import tensorflow as tf


class Actor():
    def __init__(self, n_obs, h, n_actions):
        self.n_obs = n_obs                  # dimensionality of observations
        self.h = h                          # number of hidden layer neurons
        self.n_actions = n_actions          # number of available actions
        
        self.model = model = {}
        with tf.variable_scope('actor',reuse=False):
            # convolutional layer 1
            self.model['Wc1'] = tf.Variable(tf.truncated_normal([5, 5, 2, 8], stddev=0.1))
            self.model['bc1'] = tf.Variable(tf.constant(0.1, shape=[8]))

            # convolutional layer 2
            self.model['Wc2'] = tf.Variable(tf.truncated_normal([5, 5, 8, 8], stddev=0.1))
            self.model['bc2'] = tf.Variable(tf.constant(0.1, shape=[8]))

            # fully connected 1
            self.model['W3'] = tf.Variable(tf.truncated_normal([14*10*8, 8], stddev=0.1))
            self.model['b3'] = tf.Variable(tf.constant(0.1, shape=[8]))

            # fully connected 2
            self.model['W4'] = tf.Variable(tf.truncated_normal([8, n_actions], stddev=0.1))
            self.model['b4'] = tf.Variable(tf.constant(0.1, shape=[n_actions]))
            
    def policy_forward(self, x):
        x_image = tf.reshape(x, [-1, 105, 80, 2])
                                      
        zc1 = tf.nn.conv2d(x_image, self.model['Wc1'], strides=[1, 1, 1, 1], padding='SAME') + self.model['bc1']
        hc1 = tf.nn.relu(zc1)
        hc1 = tf.nn.max_pool(hc1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        zc2 = tf.nn.conv2d(hc1, self.model['Wc2'], strides=[1, 1, 1, 1], padding='SAME') + self.model['bc2']
        hc2 = tf.nn.relu(zc2)
        hc2 = tf.nn.max_pool(hc2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        print hc2.get_shape()
        
        hc2_flat = tf.reshape(hc2, [-1, 14*10*8])
        h3 = tf.nn.relu(tf.matmul(hc2_flat, self.model['W3']) + self.model['b3'])
        h3 = tf.nn.dropout(h3, 0.9)
        
        h4 = tf.matmul(h3, self.model['W4']) + self.model['b4']
        return tf.nn.softmax(h4)


class Agent():
    def __init__(self, n_obs, n_actions, gamma=0.99, actor_lr = 1e-4, decay=0.95, epsilon = 0.1):
        self.gamma = gamma            # discount factor for reward
        self.epsilon = epsilon
        self.global_step = 0
        self.xs, self.rs, self.ys = [],[],[]
        
        self.actor_lr = actor_lr               # learning rate for policy
        self.decay = decay
        self.n_obs = n_obs                     # dimensionality of observations
        self.n_actions = n_actions             # number of available actions
        self.save_path ='models/pong.ckpt'
        
        # make actor part of brain
        self.actor = Actor(n_obs=self.n_obs, h=200, n_actions=self.n_actions)
        
        #placeholders
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs],name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="y")
        self.r = tf.placeholder(dtype=tf.float32, shape=[None,1], name="r")
        
        #gradient processing (PG magic)
        self.discounted_r = self.discount_rewards(self.r, self.gamma)
        mean, variance= tf.nn.moments(self.discounted_r, [0], shift=None, name="reward_moments")
        self.discounted_r -= mean
        self.discounted_r /= tf.sqrt(variance + 1e-6)
        
        # initialize tf graph
        self.aprob = self.actor.policy_forward(self.x)
        self.loss = tf.nn.l2_loss(self.y-self.aprob)
        self.optimizer = tf.train.RMSPropOptimizer(self.actor_lr, decay=self.decay)
        self.grads = self.optimizer.compute_gradients(self.loss,                                     var_list=tf.trainable_variables(), grad_loss=self.discounted_r)
        self.train_op = self.optimizer.apply_gradients(self.grads)

        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(tf.all_variables())
    
    def act(self, x):
        aprob = self.sess.run(self.aprob, {self.x: x})
        aprob = aprob[0,:]
        action = np.random.choice(self.n_actions,p=aprob) if np.random.rand() > self.epsilon else np.random.randint(self.n_actions)
        
        label = np.zeros_like(aprob) ; label[action] = 1
        self.xs.append(x)
        self.ys.append(label)
        
        return action
    
    def learn(self):
        epx = np.vstack(self.xs)
        epr = np.vstack(self.rs)
        epy = np.vstack(self.ys)
        self.xs, self.rs, self.ys = [],[],[] # reset game history
        
        feed = {self.x: epx, self.r: epr, self.y: epy}
        _ = self.sess.run(self.train_op,feed) # parameter update
        self.global_step += 1
        
    def try_load_model(self):
        load_was_success = True # yes, I'm being optimistic
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, load_path)
        except:
            print "no saved model to load. starting new session"
            load_was_success = False
        else:
            print "loaded model: {}".format(load_path)
            self.saver = tf.train.Saver(tf.all_variables())
            self.global_step = int(load_path.split('-')[-1])
            
    def save(self):
        self.saver.save(self.sess, self.save_path, global_step=self.global_step)
        
    @staticmethod
    def discount_rewards(r, gamma):
        discount_f = lambda a, v: a*gamma + v;
        r_reverse = tf.scan(discount_f, tf.reverse(r,[True, False]))
        discounted_r = tf.reverse(r_reverse,[True, False])
        return discounted_r


# downsampling
def prepro(o):
    rgb = o
    gray = 0.3*rgb[:,:,0:1] + 0.4*rgb[:,:,1:2] + 0.3*rgb[:,:,2:3]
    gray = gray[::2,::2,:]
    gray -= np.mean(gray) ; gray /= 100
    return gray.astype(np.float)

def plt_dynamic(x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()


# env = gym.make("Pong-v0")
# observation = env.reset()
# print prepro(observation).shape
# print 105*80


n_obs = 2*105*80   # dimensionality of observations
n_actions = 3
agent = Agent(n_obs, n_actions, gamma = 0.99, actor_lr=1e-3, decay=0.99, epsilon = 0.1)
agent.try_load_model()

env = gym.make("Pong-v0")
observation = env.reset()
cur_x = None
running_reward = -20.48 # usually starts around 10 for cartpole
reward_sum = 0
episode_number = agent.global_step


total_parameters = 0 ; print "Model overview:"
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    print '\tvariable "{}" has {} parameters'         .format(variable.name, variable_parameters)
    total_parameters += variable_parameters
print "Total of {} parameters".format(total_parameters)


fig,ax = plt.subplots(1,1)
ax.set_xlabel('X') ; ax.set_ylabel('Y')
ax.set_xlim(0,500) ; ax.set_ylim(-21,-19)
pxs, pys = [], []

print 'episode {}: starting up...'.format(episode_number)
while True:
#     if episode_number%25==0: env.render()

    # preprocess the observation, set input to network to be difference image
    prev_x = cur_x if cur_x is not None else np.zeros((105,80,1))
    cur_x = prepro(observation)
    x = np.concatenate((cur_x, prev_x),axis=-1).ravel()

    # stochastically sample a policy from the network
    action = agent.act(np.reshape(x, (1,-1)))

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action + 1)
    agent.rs.append(reward)
    reward_sum += reward
    
    if done:
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        agent.learn()

        # visualization
        pxs.append(episode_number)
        pys.append(running_reward)
        if episode_number % 10 == 0:
            print 'ep: {}, reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            plt_dynamic(pxs, pys, ax)
        else:
            print '\tep: {}, reward: {}'.format(episode_number, reward_sum)
            feed = {agent.x: np.reshape(x, (1,-1))}
            aprob = agent.sess.run(agent.aprob, feed) ; aprob = aprob[0,:]
            print'\t', aprob
            
#         if episode_number % 50 == 0: agent.save() ; print "SAVED MODEL #{}".format(agent.global_step)
        
        # lame stuff
        cur_x = None
        episode_number += 1 # the Next Episode
        observation = env.reset() # reset env
        reward_sum = 0


def prepro(o):
    rgb = o
    gray = 0.3*rgb[:,:,0:1] + 0.4*rgb[:,:,1:2] + 0.3*rgb[:,:,2:3]
    gray = gray[::2,::2,:]
    gray -= np.mean(gray) ; gray /= 100
    return gray.astype(np.float)


print np.reshape(x, (1,-1)).shape
print agent.x.get_shape()
agent.act(np.reshape(x, (1,-1)))


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = prepro(observation)
observation, reward, done, info = env.step(2)
cur_x = prepro(observation)


get_ipython().magic('matplotlib inline')
print cur_x.shape
plt.imshow(cur_x[:,:,0])


x = np.concatenate((cur_x, prev_x),axis=-1).ravel()
print x.shape


p = np.reshape(x,(-1,105,80,2))
plt.imshow(p[0,:,:,1])


plt.imshow(p[0,:,:,0])





# # Solves the Cartpole problem using Policy Gradients in Tensorflow
# 

# written October 2016 by Sam Greydanus
# 
# inspired by gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# 

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

import numpy as np
import gym
import tensorflow as tf


n_obs = 4              # dimensionality of observations
h = 128                # number of hidden layer neurons
n_actions = 2          # number of available actions

learning_rate = 1e-2
gamma = .9             # discount factor for reward
decay = 0.9            # decay rate for RMSProp gradients


tf_model = {}
with tf.variable_scope('layer_one',reuse=False):
    xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
    tf_model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
with tf.variable_scope('layer_two',reuse=False):
    xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(h), dtype=tf.float32)
    tf_model['W2'] = tf.get_variable("W2", [h,n_actions], initializer=xavier_l2)


def tf_discount_rewards(tf_r): #tf_r ~ [game_steps,1]
    discount_f = lambda a, v: a*gamma + v;
    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
    tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
    return tf_discounted_r

def tf_policy_forward(x): #x ~ [1,D]
    h = tf.matmul(x, tf_model['W1'])
    h = tf.nn.relu(h)
    logp = tf.matmul(h, tf_model['W2'])
    p = tf.nn.softmax(logp)
    return p

def plt_dynamic(x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()


env = gym.make("CartPole-v0")
observation = env.reset()
xs,rs,ys = [],[],[]
running_reward = 10 # usually starts around 10 for cartpole
reward_sum = 0
episode_number = 0
total_steps = 500


#placeholders
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs],name="tf_x")
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="tf_y")
tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")

#gradient processing (PG magic)
tf_discounted_epr = tf_discount_rewards(tf_epr)
tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
tf_discounted_epr -= tf_mean
tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

# initialize tf graph
tf_aprob = tf_policy_forward(tf_x)
loss = tf.nn.l2_loss(tf_y-tf_aprob)
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
train_op = optimizer.apply_gradients(tf_grads)

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()


fig,ax = plt.subplots(1,1)
ax.set_xlabel('X') ; ax.set_ylabel('Y')
ax.set_xlim(0,total_steps) ; ax.set_ylim(0,200)
pxs, pys = [], []

print 'episode {}: starting up...'.format(episode_number)
while episode_number <= total_steps and running_reward < 225:
#     if episode_number%25==0: env.render()

    # stochastically sample a policy from the network
    x = observation
    feed = {tf_x: np.reshape(x, (1,-1))}
    aprob = sess.run(tf_aprob,feed) ; aprob = aprob[0,:]
    action = np.random.choice(n_actions, p=aprob)
    label = np.zeros_like(aprob) ; label[action] = 1

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    
    # record game history
    xs.append(x) ; ys.append(label) ; rs.append(reward)
    
    if done:
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        epx = np.vstack(xs)
        epr = np.vstack(rs)
        epy = np.vstack(ys)
        xs,rs,ys = [],[],[] # reset game history
        
        feed = {tf_x: epx, tf_epr: epr, tf_y: epy}
        _ = sess.run(train_op,feed) # parameter update

        # visualization
        pxs.append(episode_number)
        pys.append(running_reward)
        if episode_number % 25 == 0:
            print 'ep: {}, reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            plt_dynamic(pxs, pys, ax)
        
        # lame stuff
        episode_number += 1 # the Next Episode
        observation = env.reset() # reset env
        reward_sum = 0
        
plt_dynamic(pxs, pys, ax)
if running_reward > 225:
    print "ep: {}: SOLVED! (running reward hit {} which is greater than 200)".format(
        episode_number, running_reward)


# # Solves Pong with Policy Gradients in Tensorflow
# 

# written October 2016 by Sam Greydanus
# 
# inspired by gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# 

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

import numpy as np
import gym
import tensorflow as tf


n_obs = 80 * 80        # dimensionality of observations
h = 200                # number of hidden layer neurons
n_actions = 3          # number of available actions

learning_rate = 5e-4
gamma = .99            # discount factor for reward
decay = 0.992           # decay rate for RMSProp gradients
save_path='Pong-v0/pong.ckpt'


tf_model = {}
with tf.variable_scope('layer_one',reuse=False):
    xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
    tf_model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
with tf.variable_scope('layer_two',reuse=False):
    xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(h), dtype=tf.float32)
    tf_model['W2'] = tf.get_variable("W2", [h,n_actions], initializer=xavier_l2)


def tf_discount_rewards(tf_r): #tf_r ~ [game_steps,1]
    discount_f = lambda a, v: a*gamma + v;
    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
    tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
    return tf_discounted_r

def tf_policy_forward(x): #x ~ [1,D]
    h = tf.matmul(x, tf_model['W1'])
    h = tf.nn.relu(h)
    logp = tf.matmul(h, tf_model['W2'])
    p = tf.nn.softmax(logp)
    return p

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def plt_dynamic(x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()


#placeholders
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs],name="tf_x")
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="tf_y")
tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")

#gradient processing (PG magic)
tf_discounted_epr = tf_discount_rewards(tf_epr)
tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
tf_discounted_epr -= tf_mean
tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

# initialize tf graph
tf_aprob = tf_policy_forward(tf_x)
loss = tf.nn.l2_loss(tf_y-tf_aprob)
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
train_op = optimizer.apply_gradients(tf_grads)

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs,rs,ys = [],[],[]
running_reward = 2 # usually starts around -20.48 for Pong
reward_sum = 0
episode_number = 0


saver = tf.train.Saver(tf.all_variables())
load_was_success = True # yes, I'm being optimistic
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
except:
    print "no saved model to load. starting new session"
    load_was_success = False
else:
    print "loaded model: {}".format(load_path)
    saver = tf.train.Saver(tf.all_variables())
    episode_number = int(load_path.split('-')[-1])


fig,ax = plt.subplots(1,1)
ax.set_xlabel('steps') ; ax.set_ylabel('reward')
ax.set_xlim(1000,5000) ; ax.set_ylim(-1,10)
pxs, pys = [], []

print 'ep {}: starting up...'.format(episode_number)
count = 0
while count < 100:
#     if True: env.render()
        
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
    prev_x = cur_x

    # stochastically sample a policy from the network
    feed = {tf_x: np.reshape(x, (1,-1))}
    aprob = sess.run(tf_aprob,feed) ; aprob = aprob[0,:]
    action = np.random.choice(n_actions, p=aprob)
    label = np.zeros_like(aprob) ; label[action] = 1

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action+1)
    reward_sum += reward
    
    # record game history
    xs.append(x) ; ys.append(label) ; rs.append(reward)
    
    if done:
        count+=1
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        epx = np.vstack(xs)
        epr = np.vstack(rs)
        epy = np.vstack(ys)
        xs,rs,ys = [],[],[] # reset game history
        
        feed = {tf_x: epx, tf_epr: epr, tf_y: epy}
        _ = sess.run(train_op,feed) # parameter update

        # visualization
        pxs.append(episode_number)
        pys.append(running_reward)
        if episode_number % 10 == 0:
            print 'ep: {}, reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            plt_dynamic(pxs, pys, ax)
        else:
            print '\tep: {}, reward: {}'.format(episode_number, reward_sum)
            
#         if episode_number % 50 == 0:
#             saver.save(sess, save_path, global_step=episode_number)
#             print "SAVED MODEL #{}".format(episode_number)
        
        # lame stuff
        episode_number += 1
        observation = env.reset() # reset env
        reward_sum = 0


# # A modular version
# 

# written October 2016 by Sam Greydanus
# 

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import time
import heapq

import numpy as np
import gym
import tensorflow as tf


# class ActorConv():
#     def __init__(self, n_obs, h, n_actions):
#         self.n_obs = n_obs                  # dimensionality of observations
#         self.h = h                          # number of hidden layer neurons
#         self.n_actions = n_actions          # number of available actions
        
#         self.model = model = {}
#         with tf.variable_scope('actor',reuse=False):
#             # convolutional layer 1
#             self.model['Wc1'] = tf.Variable(tf.truncated_normal([4, 4, 2, 8], stddev=0.1))
#             self.model['bc1'] = tf.Variable(tf.constant(0.1, shape=[8]))

#             # convolutional layer 2
#             self.model['Wc2'] = tf.Variable(tf.truncated_normal([4, 4, 8, 8], stddev=0.1))
#             self.model['bc2'] = tf.Variable(tf.constant(0.1, shape=[8]))

#             # fully connected 1
#             self.model['W3'] = tf.Variable(tf.truncated_normal([14*10*8, self.h], stddev=0.1))
#             self.model['b3'] = tf.Variable(tf.constant(0.1, shape=[self.h]))

#             # fully connected 2
#             self.model['W4'] = tf.Variable(tf.truncated_normal([self.h, n_actions], stddev=0.1))
#             self.model['b4'] = tf.Variable(tf.constant(0.1, shape=[n_actions]))
            
#     def policy_forward(self, x):
#         x_image = tf.reshape(x, [-1, 105, 80, 2])
                                      
#         zc1 = tf.nn.conv2d(x_image, self.model['Wc1'], strides=[1, 1, 1, 1], padding='SAME') + self.model['bc1']
#         hc1 = tf.nn.relu(zc1)
#         hc1 = tf.nn.max_pool(hc1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
#         zc2 = tf.nn.conv2d(hc1, self.model['Wc2'], strides=[1, 1, 1, 1], padding='SAME') + self.model['bc2']
#         hc2 = tf.nn.relu(zc2)
#         hc2 = tf.nn.max_pool(hc2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#         print hc2.get_shape()
        
#         hc2_flat = tf.reshape(hc2, [-1, 14*10*8])
#         h3 = tf.nn.relu(tf.matmul(hc2_flat, self.model['W3']) + self.model['b3'])
#         h3 = tf.nn.dropout(h3, 0.9)
        
#         h4 = tf.matmul(h3, self.model['W4']) + self.model['b4']
#         return tf.nn.softmax(h4)


class Actor():
    def __init__(self, n_obs, h, n_actions):
        self.n_obs = n_obs                  # dimensionality of observations
        self.h = h                          # number of hidden layer neurons
        self.n_actions = n_actions          # number of available actions
        
        self.model = model = {}
        with tf.variable_scope('actor_l1',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
            model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
            model['b1'] = tf.get_variable("b1", [1, h], initializer=xavier_l1)
        with tf.variable_scope('actor_l2',reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(h), dtype=tf.float32)
            model['W2'] = tf.get_variable("W2", [h,n_actions], initializer=xavier_l2)
            model['b2'] = tf.get_variable("b1", [1, n_actions], initializer=xavier_l2)
            
    def policy_forward(self, x): #x ~ [1,D]
        h = tf.matmul(x, self.model['W1']) + self.model['b1']
        h = tf.nn.relu(h)
        logp = tf.matmul(h, self.model['W2']) + self.model['b2']
        p = tf.nn.softmax(logp)
        return p


class Agent():
    def __init__(self, n_obs, n_actions, gamma=0.99, actor_lr = 1e-4, decay=0.95, epsilon = 0.1):
        self.gamma = gamma            # discount factor for reward
        self.epsilon = epsilon
        self.global_step = 0
        self.replay_max = 32 ; self.replay = []
        self.xs, self.rs, self.ys = [],[],[]
        
        self.actor_lr = actor_lr               # learning rate for policy
        self.decay = decay
        self.n_obs = n_obs                     # dimensionality of observations
        self.n_actions = n_actions             # number of available actions
        self.save_path ='models/pong.ckpt'
        
        # make actor part of brain
        self.actor = Actor(n_obs=self.n_obs, h=200, n_actions=self.n_actions)
        
        #placeholders
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs],name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="y")
        self.r = tf.placeholder(dtype=tf.float32, shape=[None,1], name="r")
        
        #gradient processing (PG magic)
        self.discounted_r = self.discount_rewards(self.r, self.gamma)
        mean, variance= tf.nn.moments(self.discounted_r, [0], shift=None, name="reward_moments")
        self.discounted_r -= mean
        self.discounted_r /= tf.sqrt(variance + 1e-6)
        
        # initialize tf graph
        self.aprob = self.actor.policy_forward(self.x)
        self.loss = tf.nn.l2_loss(self.y-self.aprob)
        self.optimizer = tf.train.RMSPropOptimizer(self.actor_lr, decay=self.decay, momentum=0.25)
        self.grads = self.optimizer.compute_gradients(self.loss,                                     var_list=tf.trainable_variables(), grad_loss=self.discounted_r)
        self.train_op = self.optimizer.apply_gradients(self.grads)

        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(tf.all_variables())
    
    def act(self, x):
        aprob = self.sess.run(self.aprob, {self.x: x})
        aprob = aprob[0,:]
        if np.random.rand() > 0.9998: print "\taprob is: ", aprob
        action = np.random.choice(self.n_actions,p=aprob) if np.random.rand() > self.epsilon else np.random.randint(self.n_actions)
        
        label = np.zeros_like(aprob) ; label[action] = 1
        self.xs.append(x)
        self.ys.append(label)
        
        return action
    
    def learn(self, n):
        epx = np.vstack(self.xs)
        epr = np.vstack(self.rs)
        epy = np.vstack(self.ys)
        self.append_replay([epx, epr, epy])
        self.xs, self.rs, self.ys = [],[],[] # reset game history
        
#         self.apply_rewards(epx,epr,epy)
        self.learn_replay(n)
        self.global_step += 1
        
    def apply_rewards(self, epx, epr, epy):
        feed = {self.x: epx, self.r: epr, self.y: epy}
        _ = self.sess.run(self.train_op,feed) # parameter update
        self.global_step += 1
        
    def append_replay(self, ep):
        self.replay.append(ep)
        if len(self.replay) is self.replay_max + 1:
            self.replay = self.replay[1:]
            
    def learn_replay(self, n):
        assert n <= self.replay_max, "requested number of entries exceeds epmax"
        if len(self.replay) < self.replay_max: print "\t\tqueue too small" ; return
        ix = np.random.permutation(self.replay_max)[:n]
        epx = np.vstack([ self.replay[i][0] for i in ix])
        epr = np.vstack([ self.replay[i][1] for i in ix])
        epy = np.vstack([ self.replay[i][2] for i in ix])
        self.apply_rewards(epx, epr, epy)
        
    @staticmethod
    def discount_rewards(r, gamma):
        discount_f = lambda a, v: a*gamma + v;
        r_reverse = tf.scan(discount_f, tf.reverse(r,[True, False]))
        discounted_r = tf.reverse(r_reverse,[True, False])
        return discounted_r


# downsampling
# def prepro(o):
#     rgb = o
#     gray = 0.3*rgb[:,:,0:1] + 0.4*rgb[:,:,1:2] + 0.3*rgb[:,:,2:3]
#     gray = gray[::2,::2,:]
#     gray -= np.mean(gray) ; gray /= 100
#     return gray.astype(np.float)
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def plt_dynamic(x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()


n_obs = 80*80 #2*105*80   # dimensionality of observations
n_actions = 3
agent = Agent(n_obs, n_actions, gamma = 0.99, actor_lr=1e-3, decay=0.95, epsilon = 0.1)

save_path = 'models/model.ckpt'
saver = tf.train.Saver(tf.all_variables())


saver = tf.train.Saver(tf.all_variables())
load_was_success = True # yes, I'm being optimistic
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(agent.sess, load_path)
except:
    print "no saved model to load. starting new session"
    load_was_success = False
else:
    print "loaded model: {}".format(load_path)
    saver = tf.train.Saver(tf.all_variables())
    agent.global_step = int(load_path.split('-')[-1])


env = gym.make("Pong-v0")
observation = env.reset()
cur_x = None
prev_x = None
running_reward = -20.48 # usually starts around 10 for cartpole
reward_sum = 0
episode_number = 0


total_parameters = 0 ; print "Model overview:"
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    print '\tvariable "{}" has {} parameters'         .format(variable.name, variable_parameters)
    total_parameters += variable_parameters
print "Total of {} parameters".format(total_parameters)


fig,ax = plt.subplots(1,1)
ax.set_xlabel('X') ; ax.set_ylabel('Y')
ax.set_xlim(0,500) ; ax.set_ylim(-21,-19)
pxs, pys = [], []

print 'episode {}: starting up...'.format(episode_number)
start = time.time()
while True:
#     if episode_number%25==0: env.render()

    # preprocess the observation, set input to network to be difference image
#     prev_x = cur_x if cur_x is not None else np.zeros((105,80,1))
#     cur_x = prepro(observation)
#     x = np.concatenate((cur_x, prev_x),axis=-1).ravel()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
    prev_x = cur_x

    # stochastically sample a policy from the network
    action = agent.act(np.reshape(x, (1,-1)))

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action + 1)
    agent.rs.append(reward)
    reward_sum += reward
    
    if done:
        eptime = time.time()
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        agent.learn(16)

        # visualization
        pxs.append(episode_number)
        pys.append(running_reward)
        if episode_number % 10 == 0:
            print 'ep: {}, reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            plt_dynamic(pxs, pys, ax)
        else:
            print '\tep: {}, reward: {}'.format(episode_number, reward_sum)
            
#         if episode_number % 50 == 0:
#             saver.save(agent.sess, save_path, global_step=agent.global_step)
#             print "SAVED MODEL #{}".format(agent.global_step)

        stop = time.time()
        if episode_number % 10 == 0:
            print "\t\teptime: {}".format(eptime - start)
            print "\t\tlearntime: {}".format(stop - eptime)
        start = stop
        
        # lame stuff
        cur_x = None
        episode_number += 1 # the Next Episode
        observation = env.reset() # reset env
        reward_sum = 0


saver.save(agent.sess, save_path, global_step=agent.global_step)


def prepro(o):
    rgb = o
    gray = 0.3*rgb[:,:,0:1] + 0.4*rgb[:,:,1:2] + 0.3*rgb[:,:,2:3]
    gray = gray[::2,::2,:]
    gray -= np.mean(gray) ; gray /= 100
    return gray.astype(np.float)


print np.reshape(x, (1,-1)).shape
print agent.x.get_shape()
agent.act(np.reshape(x, (1,-1)))


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = prepro(observation)
observation, reward, done, info = env.step(2)
cur_x = prepro(observation)


get_ipython().magic('matplotlib inline')
print cur_x.shape
plt.imshow(cur_x[:,:,0])


x = np.concatenate((cur_x, prev_x),axis=-1).ravel()
print x.shape


p = np.reshape(x,(-1,105,80,2))
plt.imshow(p[0,:,:,1])


plt.imshow(p[0,:,:,0])





# # A modular version with LSTMs
# 

# written November 2016 by Sam Greydanus
# 

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

import numpy as np
import gym
import tensorflow as tf


class Actor():
    def __init__(self, batch_size, tsteps, xlen, ylen):
        self.sess = tf.InteractiveSession()
        self.batch_size = batch_size
        self.xlen = xlen
        self.ylen = ylen
        self.x = x = tf.placeholder(tf.float32, shape=[None, None, xlen], name="x")
        self.y = y = tf.placeholder(tf.float32, shape=[None, None, ylen], name="y")
        
        self.params = params = {}
        self.fc1_size = fc1_size = 50
        self.rnn_size = rnn_size = 100
        with tf.variable_scope('actor',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(ylen), dtype=tf.float32)
            params['W1'] = tf.get_variable("W1", [xlen, fc1_size], initializer=xavier_l1)

            rnn_init = tf.truncated_normal_initializer(stddev=0.075, dtype=tf.float32)
            params['rnn'] = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True, initializer=rnn_init)

            params['istate_batch'] = params['rnn'].zero_state(batch_size=batch_size, dtype=tf.float32)
            params['istate'] = params['rnn'].zero_state(batch_size=1, dtype=tf.float32)

            xavier_l3 = tf.truncated_normal_initializer(stddev=1./np.sqrt(rnn_size), dtype=tf.float32)
            params['W3'] = tf.get_variable("W3", [rnn_size, ylen], initializer=xavier_l3)
        
        self.reset_state()
            
    def forward(self, x, state, tsteps, reuse=False):
        with tf.variable_scope('actor', reuse=reuse):
            x = tf.reshape(x, [-1, self.xlen])
            h = tf.matmul(x, self.params['W1'])
            h = tf.nn.relu(h) # ReLU nonlinearity
#             h = tf.nn.dropout(h,0.8)

            hs = [tf.squeeze(h_, [1]) for h_ in tf.split(1, tsteps, tf.reshape(h, [-1, tsteps, self.fc1_size]))]
            rnn_outs, state = tf.nn.seq2seq.rnn_decoder(hs, state, self.params['rnn'], scope='actor')
            rnn_out = tf.reshape(tf.concat(1, rnn_outs), [-1, self.rnn_size])
            rnn_out = tf.nn.relu(rnn_out) # ReLU nonlinearity

            logps = tf.matmul(rnn_out, self.params['W3'])
            p = tf.nn.softmax(logps)
            p = tf.reshape(p, [-1, self.ylen])
        return p, state
    
    def reset_state(self):
        self.c, self.h = self.params['istate'].c.eval(), self.params['istate'].h.eval()


class Agent():
    def __init__(self, n_obs, n_actions, gamma=0.99, lr = 1e-4, epsilon = 0.1):
        self.gamma = gamma            # discount factor for reward
        self.epsilon = epsilon
        self.global_step = 0
        self.xs, self.rs, self.ys = [],[],[]
        
        self.lr = lr               # learning rate for policy
        self.n_obs = n_obs                     # dimensionality of observations
        self.n_actions = n_actions             # number of available actions
        
        # make actor part of brain
        self.batch_size = 8
        self.tsteps = 20
        self.actor = Actor(self.batch_size, self.tsteps, self.n_obs, self.n_actions)
        
        #placeholders
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs],name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="y")
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="r")
        
        #gradient processing (PG magic)
        self.discounted_r = self.discount_rewards(self.r, self.gamma)
        mean, variance= tf.nn.moments(self.discounted_r, [0], shift=None, name="reward_moments")
        self.discounted_r -= mean
        self.discounted_r /= tf.sqrt(variance + 1e-6)
        
        # initialize tf graph
        self.y_hat, self.actor.params['fstate'] =                 self.actor.forward(self.x, self.actor.params['istate'], 1, reuse=False)
        self.y_hat_batch, _ = self.actor.forward(self.x, self.actor.params['istate_batch'], self.tsteps, reuse=True)
        
        self.loss = tf.nn.l2_loss(self.y-self.y_hat_batch)
        self.optimizer = tf.train.RMSPropOptimizer(self.lr, decay=0.99)
        self.grads = self.optimizer.compute_gradients(self.loss,                                     var_list=tf.trainable_variables(), grad_loss=self.discounted_r)
        self.train_op = self.optimizer.apply_gradients(self.grads)

        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(tf.all_variables())
        self.actor.reset_state()
    
    def act(self, x):
        feed = {self.x: x, self.actor.params['istate'].c: self.actor.c, self.actor.params['istate'].h: self.actor.h}
        fetch = [self.y_hat, self.actor.params['fstate'].c, self.actor.params['fstate'].h]
        [y_hat, self.actor.c, self.actor.h] = self.sess.run(fetch, feed)
        y_hat = y_hat[0,:]
        if np.random.rand() > 0.99985: print "\ty_hat is: ", y_hat
        action = np.random.choice(self.n_actions,p=y_hat) if np.random.rand() > self.epsilon else np.random.randint(self.n_actions)
        
        label = np.zeros_like(y_hat) ; label[action] = 1
        self.xs.append(x)
        self.ys.append(label)
        return action
    
    def learn(self):
        epx = np.vstack(self.xs)
        epr = np.vstack(self.rs)
        epy = np.vstack(self.ys)
        self.xs, self.rs, self.ys = [],[],[] # reset game history
        
        unit_len = self.batch_size*self.tsteps
        buffer_len = ((unit_len - epx.shape[0]%unit_len)%unit_len)
        epx_buffer = np.zeros((buffer_len,epx.shape[1]))
        epr_buffer = np.zeros((buffer_len,epr.shape[1]))
        epy_buffer = np.zeros((buffer_len,epy.shape[1]))
        
        epx = np.concatenate((epx, epx_buffer),axis=0)
        epr = np.concatenate((epr, epr_buffer),axis=0)
        epy = np.concatenate((epy, epy_buffer),axis=0)
        
        num_batches = epx.shape[0]/unit_len
        for b in range(num_batches):
            start = b*unit_len ; stop = (b+1)*unit_len
            feed = {self.x: epx[start:stop,:], self.r: epr[start:stop,:], self.y: epy[start:stop,:]}
            train_loss, _ = self.sess.run([self.loss, self.train_op],feed) # parameter update
        self.global_step += 1
        return train_loss
        
    @staticmethod
    def discount_rewards(r, gamma):
        discount_f = lambda a, v: (a*gamma + v)*(1-tf.abs(v)) + (v)*tf.abs(v);
        r_reverse = tf.scan(discount_f, tf.reverse(r,[True, False]))
        discounted_r = tf.reverse(r_reverse,[True, False])
        return discounted_r


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def plt_dynamic(x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()


n_obs = 80*80   # dimensionality of observations
n_actions = 3
agent = Agent(n_obs, n_actions, gamma=0.992, lr = 1e-4, epsilon = 0.0)

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
running_reward = -20.48 # usually starts around 10 for cartpole
reward_sum = 0
episode_number = 0

save_path = 'rnn_models/model.ckpt'
saver = tf.train.Saver(tf.all_variables())


saver = tf.train.Saver(tf.all_variables())
load_was_success = True # yes, I'm being optimistic
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(agent.sess, load_path)
except:
    print "no saved model to load. starting new session"
    load_was_success = False
else:
    print "loaded model: {}".format(load_path)
    saver = tf.train.Saver(tf.all_variables())
    agent.global_step = int(load_path.split('-')[-1])


total_parameters = 0 ; print "Model overview:"
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    print '\tvariable "{}" has {} parameters'         .format(variable.name, variable_parameters)
    total_parameters += variable_parameters
print "Total of {} parameters".format(total_parameters)


fig,ax = plt.subplots(1,1)
ax.set_xlabel('X') ; ax.set_ylabel('Y')
ax.set_xlim(0,1000) ; ax.set_ylim(-21,-19)
pxs, pys = [], []

print 'episode {}: starting up...'.format(episode_number)
while True:
#     if True: env.render()

    # preprocess the observation
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
    prev_x = cur_x

    # stochastically sample a policy from the network
    action = agent.act(np.reshape(x, (1,-1)))

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action + 1)
    agent.rs.append(reward)
    reward_sum += reward
    
    if done:
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        agent.learn()

        # visualization
        pxs.append(episode_number)
        pys.append(running_reward)
        if episode_number % 10 == 0:
            print 'ep: {}, reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
            plt_dynamic(pxs, pys, ax)
        else:
            print '\tep: {}, reward: {}'.format(episode_number, reward_sum)
            
#         if episode_number % 50 == 0:
#             saver.save(agent.sess, save_path, global_step=agent.global_step)
#             print "SAVED MODEL #{}".format(agent.global_step)
        
        # lame stuff
        cur_x = None
        episode_number += 1 # the Next Episode
        observation = env.reset() # reset env
        reward_sum = 0


saver.save(agent.sess, save_path, global_step=agent.global_step)


print agent.aprob
print agent.batch_aprob






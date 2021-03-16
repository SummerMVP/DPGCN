# models.py
from DPGCN.layers import *
from DPGCN.metrics import *
# from layers import *
# from metrics import *

noise = 1 



class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        # import tensorflow as tf;
        # print(tf.__version__)
        if(FLAGS.DP==0):
          self.opt_op = self.optimizer.minimize(self.loss)
        

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, sigma,input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.sigma=sigma 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error self.outputs是一个矩阵（2708，7），损失函数计算outputs和label的损失，再mask计算
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        if(FLAGS.DP):
            #这里计算梯度
            print("这里开始差分隐私！")
            gw_W1 = tf.gradients(self.loss,self.layers[0].vars['weights'])[0] # gradient of W1
            # print("self.layers[0].vars['weights_0']",self.layers[0].vars['weights_0'])
            # print("self.layers[0].vars['bias']",self.layers[0].vars['bias'])
            gb1 = tf.gradients(self.loss,self.layers[0].vars['bias'])[0] # gradient of b1

            gw_W2 = tf.gradients(self.loss,self.layers[1].vars['weights'])[0] # gradient of W2
            gb2 = tf.gradients(self.loss,self.layers[1].vars['bias'])[0] # gradient of b2
            #clip gradient  梯度裁剪
            gw_W1 = tf.clip_by_norm(gw_W1,FLAGS.clip)
            gw_W2 = tf.clip_by_norm(gw_W2,FLAGS.clip)
            if(noise):
                # print("这里添加噪音！σ：")
                # print(sigma)
                sensitivity = FLAGS.clip
                #这里在梯度上加入噪音！，这里有四个**2
                gw_W1 += tf.random_normal(shape=tf.shape(gw_W1), mean=0.0, stddev = (self.sigma * sensitivity), dtype=tf.float32)
                gb1 += tf.random_normal(shape=tf.shape(gb1), mean=0.0, stddev = (self.sigma * sensitivity), dtype=tf.float32)
                gw_W2 += tf.random_normal(shape=tf.shape(gw_W2), mean=0.0, stddev = (self.sigma * sensitivity), dtype=tf.float32)
                gb2 += tf.random_normal(shape=tf.shape(gb2), mean=0.0, stddev = (self.sigma * sensitivity), dtype=tf.float32)
            # print("这里更新梯度")
            self.optimizer = self.optimizer.apply_gradients([(gw_W1,self.layers[0].vars['weights']),(gb1,self.layers[0].vars['bias']),
            (gw_W2,self.layers[1].vars['weights']),(gb2,self.layers[1].vars['bias'])]);

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, sigma, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        # print("self.output_dim = placeholders['labels'].get_shape().as_list()[1]",self.output_dim)
        self.placeholders = placeholders
        self.sigma=sigma 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  #修改成梯度下降优化器
        # print("不使用梯度下降了！")
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        if(FLAGS.DP):
            #这里计算梯度
            print("这里进行扰乱！")
            gw_W1 = tf.gradients(self.loss,self.layers[0].vars['weights_0'])[0] # gradient of W1
            # print("self.layers[0].vars['weights_0']",self.layers[0].vars['weights_0'])
            # print("self.layers[0].vars['bias']",self.layers[0].vars['bias'])
            gb1 = tf.gradients(self.loss,self.layers[0].vars['bias'])[0] # gradient of b1

            gw_W2 = tf.gradients(self.loss,self.layers[1].vars['weights_0'])[0] # gradient of W2
            gb2 = tf.gradients(self.loss,self.layers[1].vars['bias'])[0] # gradient of b2
            #clip gradient  梯度裁剪
            gw_W1 = tf.clip_by_norm(gw_W1,FLAGS.clip)
            gw_W2 = tf.clip_by_norm(gw_W2,FLAGS.clip)
            if(noise):
                # print("这里添加噪音！σ：")
                # print(sigma)
                sensitivity = FLAGS.clip
                #这里在梯度上加入噪音！
                gw_W1 += tf.random_normal(shape=tf.shape(gw_W1), mean=0.0, stddev = (self.sigma * sensitivity)**2, dtype=tf.float32)
                gb1 += tf.random_normal(shape=tf.shape(gb1), mean=0.0, stddev = (self.sigma * sensitivity)**2, dtype=tf.float32)
                gw_W2 += tf.random_normal(shape=tf.shape(gw_W2), mean=0.0, stddev = (self.sigma * sensitivity)**2, dtype=tf.float32)
                gb2 += tf.random_normal(shape=tf.shape(gb2), mean=0.0, stddev = (self.sigma * sensitivity)**2, dtype=tf.float32)
            # print("这里更新梯度")
            self.optimizer = self.optimizer.apply_gradients([(gw_W1,self.layers[0].vars['weights_0']),(gb1,self.layers[0].vars['bias']),
            (gw_W2,self.layers[1].vars['weights_0']),(gb2,self.layers[1].vars['bias'])]);


    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))
    
    def predict(self):
        # print("self.outputs",self.outputs[0])
        return tf.nn.softmax(self.outputs)

class ChebNet(Model):
    def __init__(self, placeholders, sigma, input_dim, **kwargs):
        super(ChebNet, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        # print("self.output_dim = placeholders['labels'].get_shape().as_list()[1]",self.output_dim)
        self.placeholders = placeholders
        self.sigma=sigma 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  #修改成梯度下降优化器
        # print("不使用梯度下降了！")
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        if(FLAGS.DP):
            #这里计算梯度
            print("这里进行扰乱！")
            gw_W1 = tf.gradients(self.loss,self.layers[0].vars['weights_0'])[0] # gradient of W1
            # print("self.layers[0].vars['weights_0']",self.layers[0].vars['weights_0'])
            # print("self.layers[0].vars['bias']",self.layers[0].vars['bias'])
            gb1 = tf.gradients(self.loss,self.layers[0].vars['bias'])[0] # gradient of b1

            gw_W2 = tf.gradients(self.loss,self.layers[1].vars['weights_0'])[0] # gradient of W2
            gb2 = tf.gradients(self.loss,self.layers[1].vars['bias'])[0] # gradient of b2
            #clip gradient  梯度裁剪
            gw_W1 = tf.clip_by_norm(gw_W1,FLAGS.clip)
            gw_W2 = tf.clip_by_norm(gw_W2,FLAGS.clip)
            if(noise):
                # print("这里添加噪音！σ：")
                # print(sigma)
                sensitivity = FLAGS.clip
                #这里在梯度上加入噪音！
                gw_W1 += tf.random_normal(shape=tf.shape(gw_W1), mean=0.0, stddev = (self.sigma * sensitivity), dtype=tf.float32)
                gb1 += tf.random_normal(shape=tf.shape(gb1), mean=0.0, stddev = (self.sigma * sensitivity), dtype=tf.float32)
                gw_W2 += tf.random_normal(shape=tf.shape(gw_W2), mean=0.0, stddev = (self.sigma * sensitivity), dtype=tf.float32)
                gb2 += tf.random_normal(shape=tf.shape(gb2), mean=0.0, stddev = (self.sigma * sensitivity), dtype=tf.float32)
            # print("这里更新梯度")
            self.optimizer = self.optimizer.apply_gradients([(gw_W1,self.layers[0].vars['weights_0']),(gb1,self.layers[0].vars['bias']),
            (gw_W2,self.layers[1].vars['weights_0']),(gb2,self.layers[1].vars['bias'])]);


    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))
    
    def predict(self):
        # print("self.outputs",self.outputs[0])
        return tf.nn.softmax(self.outputs)

class shadow_GCN(object):
    def __init__(self, placeholders, learning_rate,hidden1,weight_decay,input_dim, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}
        self.layers = []
        self.activations = []
        self.loss = 0
        self.accuracy = 0
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  #修改成梯度下降优化器
        self.opt_op =None
        self.hidden1=hidden1
        self.weight_decay=weight_decay
        self.build()

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        # import tensorflow as tf;
        # print(tf.__version__)
        self.opt_op = self.optimizer.minimize(self.loss)
        
    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
       

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))
    
    def predict(self):
        # print("self.outputs",self.outputs[0])
        return tf.nn.softmax(self.outputs)

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class shadow_MLP():
    def __init__(self, placeholders,learning_rate,hidden1,weight_decay,input_dim, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}
        self.layers = []
        self.activations = []
        self.loss = 0
        self.accuracy = 0
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  #修改成梯度下降优化器
        self.opt_op =None
        self.hidden1=hidden1
        self.weight_decay=weight_decay
        self.build()

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        # import tensorflow as tf;
        # print(tf.__version__)
        self.opt_op = self.optimizer.minimize(self.loss)
    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error self.outputs是一个矩阵（2708，7），损失函数计算outputs和label的损失，再mask计算
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=self.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=self.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)



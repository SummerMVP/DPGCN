# train.py
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
# from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import numpy as np
import math
# import tensorflow.compat.v1 as tf

from DPGCN.utils import *
from DPGCN.models import GCN, MLP,ChebNet
from DPGCN.RDP import RDP_run 
import DPGCN.accountant as accountant

# from utils import *
# from models import GCN, MLP,ChebNet
# import accountant
# from RDP import RDP_run 

#没有下面这个可能会报错
# tf.disable_eager_execution()
dataPath="work/Myproject/data"

def train(dataset,DP=0,learning_rate=0.01,model='gcn',max_degree=3,hidden1=16,dropout=0.5,weight_decay=5e-4,target_eps=6,
sigma=2.4,delta=[1e-5],epochs=120,clip=1e-2):
  flags = tf.app.flags
  FLAGS = flags.FLAGS
  flags.DEFINE_string('dataset', dataset, 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
  flags.DEFINE_string('model', model, 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
  flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.')
  # flags.DEFINE_integer('epochs', epochs, 'Number of epochs to train.')
  flags.DEFINE_integer('hidden1', hidden1, 'Number of units in hidden layer 1.')#卷积层第一层的output_dim，第二层的input_dim
  flags.DEFINE_float('dropout', dropout, 'Dropout rate (1 - keep probability).')
  flags.DEFINE_float('clip', clip, 'clip bound')
  flags.DEFINE_float('weight_decay',weight_decay, 'Weight for L2 loss on embedding matrix.')
  flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
  flags.DEFINE_integer('max_degree', max_degree, 'Maximum Chebyshev polynomial degree.')
  flags.DEFINE_string('f', '', 'kernel')
  flags.DEFINE_integer('DP',DP, 'DP method')
  seed = 128
  np.random.seed(seed)
  tf.set_random_seed(seed)
  #差分隐私有关参数,和计算隐私成本有关的参数 
  target_eps = target_eps; 
  sigma = sigma # 'sigma' 
  delta = delta # 'delta' 
  # lambd = 1e3 # 'exponential distribution parameter'指数分布的参数 
  if(dataset=='cora'):
    # D = 2708 #这个应该是样本总数 140
    D=1208
    batch_size=140
    N= 2708
    k=np.sqrt(140/2708)*7/pow(2,2)*5
  elif(dataset=='citeseer'):
    # D = 3327  #这个应该是训练样本总数 120
    D=2827
    batch_size=120
    N= 3327
    k=np.sqrt(120/3327)*6/pow(2,2)*5
  elif(dataset=='pubmed'):
    # D = 19717;  #这个应该是训练样本总数  60
    D=18217
    batch_size=60
    N= 19717
    k=np.sqrt(60/19717)*3/pow(2,2)*100

  pri_acc = DP; #1---DP，2---RDP,0--None
  S=0.001  #整个是自适应噪声系数的精度阈值
  k1=0.99 #这个是衰减率

  #差分隐私参数结束
  # Load data 加载训练数据
 
  adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask ,y = load_data(dataset)

  # print("feayures",features.shape)
  # Some preprocessing选择模型

  # if(dataset!='mnist'):
  features = preprocess_features(features)
    # print("features[2][1]",features[2][1])
  if model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
  elif model == 'gcn_cheby':
      support = chebyshev_polynomials(adj, max_degree)
      num_supports = 1 + max_degree
      model_func = ChebNet
  elif model == 'dense':
      support = [preprocess_adj(adj)]  # Not used
      num_supports = 1
      model_func = MLP
  else:
      raise ValueError('Invalid argument for model: ' + str(model))

  placeholders = {
      'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
      'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
      'labels': tf.placeholder(tf.int32, shape=(None, y_train.shape[1])),
      'labels_mask': tf.placeholder(tf.int32),
      'dropout': tf.placeholder_with_default(0., shape=()),
      'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
  }

  # Create model 创建模型
  if(dataset=="mnist"):
    model = model_func(placeholders, sigma,input_dim=features.shape[1], logging=True)
  else:
    model = model_func(placeholders, sigma,input_dim=features[2][1], logging=True)

  # Initialize session
  sess = tf.Session()


  # Define model evaluation function 定义模型评估函数
  def evaluate(features, support, labels, mask, placeholders):
      t_test = time.time()
      feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
      outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
      # print("len",len(outs_val[2]))
      return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]

  if (pri_acc==1):
    priv_accountant = accountant.GaussianMomentsAccountant(D)
    privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None],sigma, batch_size)
    # privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None],sigma, 1000)
    print("DP accountant!")
  elif(pri_acc==2):
    print("RDP accuntant!")

  # Init variables
  sess.run(tf.global_variables_initializer())
  cost_val = []
  acc_arry=[]
  # Train model
  for epoch in range(epochs):
      # privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None],sigma, batch_size)
      t = time.time()
      # Construct feed dictionary
      feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
      feed_dict.update({placeholders['dropout']: dropout})

      # Training step
      if(pri_acc):
        outs = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed_dict)
      else:
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

      # Validation 验证集参数
      cost, acc, duration ,predic = evaluate(features, support, y_val, val_mask, placeholders)
      cost_val.append(cost)
      
      if(pri_acc==1):
        #隐私预算
        # print("DP account begin!")
        sess.run([privacy_accum_op])
        spent_privacy = priv_accountant.get_privacy_spent(sess,target_deltas=delta)
        # print("eps_delta:",spent_privacy[0]) 
        eps=spent_privacy[0].spent_eps
        # print("delta",delta) 
      if(pri_acc==2):
        # print("RDP account begin!")
        spent_privacy=RDP_run(dataset_size=D,batch_size=batch_size,noise_multiplier=sigma,epochs=epoch,delta=delta)
        # delta1=1e-5
        # spent_privacy=compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=D, batch_size=batch_size, noise_multiplier=sigma, epochs=epoch, delta=delta1) 
        # print("eps:",spent_privacy[0])
        eps=spent_privacy[0]
        # print("opt_alpha:",spent_privacy[1])  
        # print("delta",delta) 
        
      # Print results
      # if(epoch%10==0):
      print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
            "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
      # print("eps:",eps)
      # print("acc:",)
      # acc_arry.append(acc)
      # 下面那行代码是防止梯度爆炸的，如果损失函数没有下降反而上升了就提前停止
          # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
      #     print("Early stopping...")
      #这里开始自适应：
      # if(pri_acc!=0 and epoch>1):
      #   if(acc_arry[-1]-acc_arry[-2]<S):
      #     print("这里进行了衰减！")
      #     # sigma*=(k/math.log(epoch,math.e))
      #     print(sigma)
      #     sigma*=k1
          # privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None],sigma, batch_size)
    
      
      if(pri_acc):
        _break = False
        if eps > target_eps:        
          _break = True;                 
          break;
        if _break == True:
            break;
  print("Optimization Finished!")
  # 测试集
  test_cost, test_acc, test_duration ,te_predic= evaluate(features, support, y_test, test_mask, placeholders)
  print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
  
  #保存训练集预测值，用以成员推理攻击
  #只有训练集的部分有结果[原来的预测是所有点都预测了的]
  tr_cost, tr_acc, tr_duration ,tr_predic = evaluate(features, support, y_train, train_mask, placeholders)
  _, _, _ ,val_predic = evaluate(features, support, y_val, val_mask, placeholders)    
  attack_x, attack_y = [], []
  for i in range(len(train_mask)):
        if(train_mask[i]):
              attack_x.append(tr_predic[i])
              attack_y.append([int(1),int(0)])
              #这里是新添加的
        elif(val_mask[i]):
              attack_x.append(val_predic[i])
              attack_y.append([int(0),int(1)])
              # 新的
        elif(test_mask[i]):
              attack_x.append(te_predic[i])              
              attack_y.append([int(0),int(1)])
        # else:
        #   attack_x.append(tr_predic[i])
        #   attack_y.append([int(1),int(0)])
  attack_x = np.vstack(attack_x)
  attack_y = np.vstack(attack_y)
  attack_x=np.array(attack_x)
  attack_y=np.array(attack_y)
  attack_x = attack_x.astype('float32')
  attack_y = attack_y.astype('int32') 
  # 这里进行全监督转换
  np.savez(dataPath+"/shadow_labels.npz",tr_predic,te_predic,val_predic)
  # np.save(dataPath+"/shadow_labels.npy",te_predic)
  print("finished!")

  return attack_x,attack_y,y


if __name__ == '__main__':
  train(model='gcn_cheby',dataset="pubmed",DP=1,target_eps=0.186,epochs=200,sigma=2.4,delta=[1e-5],clip=1e-4)
  # testsigma()
from sklearn.metrics import classification_report, accuracy_score
# import theano.tensor as T
import numpy as np
# import lasagne
# import theano
import argparse
#n ew
from DPGCN.models import *
from DPGCN.utils import *
from DPGCN.train import *
import tensorflow as tf
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.metrics import classification_report
# SVM训练，有问题 效果0.90
def classifier1():
    clf = SVC(C=0.5,                         #误差项惩罚系数,默认值是1
                  kernel='linear',               #线性核 kenrel="rbf":高斯核
                  decision_function_shape='ovo')
    return clf
#随机森林，效果0.56
def classifier2():     
    clf = RandomForestClassifier(n_estimators=50)     
    return clf
#AdaBoostClassifier ，一种集成算法，效果0.64895
def classifier3():     
    #迭代100次 ,学习率为0.01     
    clf = AdaBoostClassifier(n_estimators=100,learning_rate=0.01)     
    return clf 
# GradientBoosting，一种集成算法 0.87
def classifier4():    
    #迭代100次 ,学习率为0.01     
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.01)    
    return clf
def trainclf(dataset):     
    train_x, train_y, test_x, test_y = dataset
    np.savez("clfdata.npz",train_x, train_y, test_x, test_y)
    clf1=classifier1()
    # print("train_y.shape:",train_y.shape)
    # print("train_x.shape:",train_x.shape)     
    # train_y.ravel()
    y=[]
    for i in range(len(train_y)):
        if(train_y[i][0]==1):
            y.append(int(1))
        else:
            y.append(int(0))
    y=np.vstack(y)
    clf1.fit(train_x,y.ravel())#ravel()将多维数据降成一维
    target_names=['out','in']    
    y1=[]
    for i in range(len(test_y)):
        if(test_y[i][0]==1):             
            y1.append(int(1))        
        else:             
            y1.append(int(0))     
    y1=np.vstack(y1)
    print("train reporter:")
    print(classification_report(y, clf1.predict(train_x), target_names=target_names))
    print("test reporter:")
    print(classification_report(y1, clf1.predict(test_x), target_names=target_names))
    return clf1.score(test_x, y1)



def shadow_evaluate(features, support, labels, mask, placeholders,sess,model):
      t_test = time.time()
      feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
      outs_val = sess.run([model.loss, model.accuracy,model.predict()], feed_dict=feed_dict_val)
      # print("len",len(outs_val[2]))
      return outs_val[0], outs_val[1], (time.time() - t_test),outs_val[2]


def add_layer(inputs,in_size,out_size,activation_function=None):
    dropout=0.5
    inputs = tf.nn.dropout(inputs, 1-dropout)
    w = tf.Variable(tf.random_normal([in_size,out_size]))
    b = tf.Variable(tf.zeros([1,out_size])+0.1)
    f = tf.matmul(inputs,w)+b
    if activation_function is None:
        outputs = f
    else:
        outputs = activation_function(f)
    return outputs

def get_tensor_net(placeholder1,in_size,out_size,hidden_size):
    # Create the model
    l1 = add_layer(placeholder1,in_size,hidden_size,activation_function=tf.nn.relu)
    prediction = add_layer(l1,hidden_size,out_size,activation_function=tf.nn.relu)
    return prediction


# 全连接神经 0.69
def train(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01):
    train_x, train_y, test_x, test_y = dataset
    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))
    # n_out=1

    if batch_size > len(train_y):
        batch_size = len(train_y)

    print ('Building model with {} training data, {} classes...'.format(len(train_x), n_out))
    xs = tf.placeholder(tf.float32,[None,n_in])
    ys = tf.placeholder(tf.float32,[None,n_out])
    prediction=tf.nn.softmax(get_tensor_net(xs,n_in,n_out,10))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) 
    # 初始化
    init = tf.global_variables_initializer()
    sess = tf.Session()

    graph = tf.get_default_graph()
    with graph.as_default():
    # sess.run()
        sess.run(init)
    # 开始迭代训练
    for i in range(epochs):
        sess.run(train_step,feed_dict={xs:train_x,ys:train_y})
        if(i%5==0):
            print("train epoch "+str(i)+":")
            print(sess.run([cross_entropy,accuracy],feed_dict={xs:train_x,ys:train_y}))
    if test_x is not None:
        print ('Testing...')
        # if batch_size > len(test_y):
        #     batch_size = len(test_y)
        # print(sess.run([cross_entropy,accuracy],feed_dict={xs:test_x,ys:test_y}))
        print("accuracy:",sess.run(accuracy,feed_dict={xs:test_x,ys:test_y}))
        accuracy=sess.run(accuracy,feed_dict={xs:test_x,ys:test_y})
        return accuracy

def shadow_train(dataset,epochs,dropout=0.1,learning_rate=0.01,hidden1=25,weight_decay=1e-4,model='gcn',num=0):
    # Load data
    max_degree=3
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask ,lab= shadow_load_data(dataset)
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask ,lab= shadow_load_data1(dataset)
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask ,lab= load_data(dataset)
    # Some preprocessing
    features = preprocess_features(features)
    if model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = shadow_GCN
    elif model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, max_degree)
        num_supports = 1 + max_degree
        model_func = shadow_GCN
    elif model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = shadow_MLP
    else:
      raise ValueError('Invalid argument for model: ' + str(model))
   
    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    # Create model
    # if(model=='gcn'):
    model = model_func(placeholders,learning_rate,hidden1,weight_decay,input_dim=features[2][1],logging=True)
    # else:
    #     model = model_func(placeholders,learning_rate,hidden1,weight_decayinput_dim=features[2][1], logging=True)
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())
    # cost_val = []
    # Train model
   
    for epoch in range(epochs):
        # print("epoch begin:",epoch)
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: dropout})
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        cost, acc, duration ,p= shadow_evaluate(features, support, y_val, val_mask, placeholders,sess,model)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
        # print("epoch end:",epoch)
    print(str(num)+"  Optimization Finished!")

    tr_cost, tr_acc, tr_duration ,tr_predic = shadow_evaluate(features, support, y_train, train_mask, placeholders,sess,model)
    _, _, _ ,val_predic = shadow_evaluate(features, support, y_val, val_mask, placeholders,sess,model)     
    # 测试集
    test_cost, test_acc, test_duration ,te_predic= shadow_evaluate(features, support, y_test, test_mask, placeholders,sess,model)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    #保存测试集集预测值，用以成员推理攻击
    # test_predic=np.zeros(te_predic.shape)
    # test_predic[test_mask, :] = tr_preic[test_mask, :]
    # np.save("Myproject\data\shadow_core_test_predic"+str(num)+".npy",test_predic)
    # a=0
    p=0
    attack_x, attack_y = [], []
    # for i in range(len(train_mask)):
    #         if(train_mask[i]):
    #             attack_x.append(tr_predic[i])
    #             attack_y.append([int(1),int(0)])
    #         elif(test_mask[i] and p<500):
    #             attack_x.append(te_predic[i])
    #             attack_y.append([int(0),int(1)])
    #             p+=1
    #          # 新的         
    #         elif(val_mask[i] and p<500):               
    #             attack_x.append(val_predic[i])                            
    #             attack_y.append([int(0),int(1)])
    #             p+=1

    for i in range(len(train_mask)):
            if(train_mask[i]):
                attack_x.append(tr_predic[i])
                attack_y.append([int(1),int(0)])
            elif(test_mask[i]):
                attack_x.append(te_predic[i])
                attack_y.append([int(0),int(1)])
                # p+=1
             # 新的         
            elif(val_mask[i]):               
                attack_x.append(val_predic[i])                            
                attack_y.append([int(0),int(1)])
                # p+=1
          
            # else:
            #     attack_x.append(tr_predic[i]) 
            #     attack_y.append([int(1),int(0)])
                # a+=1
                # print("test:",a)
    attack_x = np.vstack(attack_x)
    attack_y = np.vstack(attack_y)
    attack_x=np.array(attack_x)
    attack_y=np.array(attack_y)
    # attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    print(str(num)+"  finished!")
    return attack_x,attack_y,lab,test_acc


import os
import tensorflow as tf
import re
import numpy as np
import math
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DOC_NAME = os.listdir("Document")  # Document file name

DOCUMENT = []
VOC_DICT = {}   # change 51252 words to 123xx id

WINDOW_SIZE = 1
WORD_COUNT = 0


def readfile():
    global DOCUMENT, DOC_NAME, WORD_COUNT, VOC_DICT
    voc_id = 0
    # read document , create word id dictionary
    for doc_id in DOC_NAME:
        with open("Document\\" + doc_id) as doc_file:
            doc_file_content = doc_file.read()
            doc_voc = re.split(' |\n', doc_file_content)
            doc_voc = list(filter('-1'.__ne__, doc_voc))
            doc_voc.remove('')
            del doc_voc[0:5]
            doc_voc = list(map(int, doc_voc))
            # word's id create
            for voc in doc_voc:
                if str(voc) not in VOC_DICT:
                    VOC_DICT[str(voc)] = voc_id
                    voc_id += 1

        DOCUMENT.append(doc_voc)

    WORD_COUNT = len(VOC_DICT)

    print('read file down')


def add_layer(inputs, input_tensors, output_tensors, activation_function=None):
    w = tf.Variable(tf.random_normal([input_tensors, output_tensors], stddev=1.0 / math.sqrt(WORD_COUNT)), name='w')
    b = tf.Variable(tf.zeros([output_tensors]), name='b')
    formula = tf.add(tf.matmul(inputs, w), b)  # matmul = dot
    if activation_function is None:
        outputs = formula
    else:
        outputs = activation_function(formula)
    return outputs


readfile()

train_inputL = tf.placeholder(tf.int64, shape=[1])
train_inputR = tf.placeholder(tf.int64, shape=[1])
# y_hat = tf.placeholder(tf.float32, shape=[1, WORD_COUNT])
y_hat = tf.placeholder(tf.int64, shape=[1, 1])


cbow = tf.Variable(tf.random_uniform([WORD_COUNT, 100], -1.0, 1.0), name="cbow")
cbowL = tf.nn.embedding_lookup(cbow, ids=train_inputL)
cbowR = tf.nn.embedding_lookup(cbow, ids=train_inputR)
average = tf.reduce_mean([cbowL, cbowR], 0, keep_dims=False)
# prediction = add_layer(average, input_tensors=100, output_tensors=WORD_COUNT, activation_function=None)
#
# y_hat_onehot = tf.one_hot(y_hat, WORD_COUNT)
# loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y_hat_onehot)
# loss = tf.reduce_sum(loss, axis=1)


nce_weights = tf.Variable(
        tf.truncated_normal([WORD_COUNT, 100],
                            stddev=1.0 / math.sqrt(100)))
nce_biases = tf.Variable(tf.zeros([WORD_COUNT]))

nce_loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=y_hat,        # ans
                       inputs=average,      # embeding
                       num_sampled=128,       # negative sampled num
                       num_classes=WORD_COUNT))     # ?

optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(nce_loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print(time.strftime("%D,%H:%M:%S"))
for iteration in range(9999999):
    print(iteration)
    cb_save = sess.run(cbow)

    total_loss = 0
    for did, doc in enumerate(DOCUMENT):

        for wid, word in enumerate(doc):
            if wid < WINDOW_SIZE or wid > (len(doc)-WINDOW_SIZE-1):
                continue
            # l_onehot = onehot(doc[wid - WINDOW_SIZE])
            # r_onehot = onehot(doc[wid + WINDOW_SIZE])
            il = VOC_DICT[str(doc[wid - WINDOW_SIZE])]
            ir = VOC_DICT[str(doc[wid + WINDOW_SIZE])]
            ans = VOC_DICT[str(word)]

            _, loss = sess.run([optimizer, nce_loss], feed_dict={train_inputL: [il], train_inputR: [ir], y_hat: [[ans]]})
            # print(loss)
            total_loss += loss
        if did % 100 == 0:
            print(str(did) + '/2265')

    print(total_loss)
    if iteration % 10 == 0:
        np.savetxt('./embedding/cbowADG_dict' + str(iteration) + '.txt', cb_save, delimiter=',')
        print('save down')

    print(time.strftime("%D,%H:%M:%S"))

    # saver = tf.train.Saver()
    # save_path = saver.save(sess, "/jupyter/IR/IR-HW5-CBOW/save/test.ckpt")
    # print("Model saved in file: %s" % save_path)
#


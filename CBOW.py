import os
import tensorflow as tf
import re
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DOC_NAME = os.listdir("Document")  # Document file name
QUERY_NAME = os.listdir("Query")  # Query file name

QUERY = []
DOCUMENT = []
BG = []

WINDOW_SIZE = 1
WORD_COUNT = 51252


def readfile():
    global QUERY, DOCUMENT, BG
    global QUERY_NAME, DOC_NAME

    # read document , create dictionary
    for doc_id in DOC_NAME:
        doc_dict = {}
        with open("Document\\" + doc_id) as doc_file:
            doc_file_content = doc_file.read()
            doc_voc = re.split(' |\n', doc_file_content)
            doc_voc = list(filter('-1'.__ne__, doc_voc))
            doc_voc.remove('')
            del doc_voc[0:5]
            doc_voc = list(map(int, doc_voc))
            # for dv_id, dv_voc in enumerate(doc_voc):
            #     if dv_id < 5:
            #         continue
            #     if dv_voc in doc_dict:
            #         doc_dict[dv_voc] += 1
            #     else:
            #         doc_dict[dv_voc] = 1
            # if '' in doc_dict:  # ? error
            #     doc_dict.pop('')
        DOCUMENT.append(doc_voc)

    for query_id in QUERY_NAME:
        query_dict = {}
        with open("Query\\" + query_id) as query_file:
            query_file_content = query_file.read()
            query_voc = re.split(' |\n', query_file_content)
            query_voc = list(filter('-1'.__ne__, query_voc))
            for qv_id, qv_voc in enumerate(query_voc):
                if qv_voc in query_dict:
                    query_dict[qv_voc] += 1
                else:
                    query_dict[qv_voc] = 1
            if '' in query_dict:  # ? error
                query_dict.pop('')
        QUERY.append(query_dict)
    print('read file down')


def add_layer(inputs, input_tensors, output_tensors, activation_function=None):
    w = tf.Variable(tf.random_normal([input_tensors, output_tensors]))
    b = tf.Variable(tf.truncated_normal([output_tensors]))
    formula = tf.add(tf.matmul(inputs, w), b)  # matmul = dot
    if activation_function is None:
        outputs = formula
    else:
        outputs = activation_function(formula)
    return outputs


def onehot(word):
    code = [0] * WORD_COUNT
    code[word] = 1
    return code


readfile()

train_inputL = tf.placeholder(tf.int64, shape=[1])
train_inputR = tf.placeholder(tf.int64, shape=[1])
y_hat = tf.placeholder(tf.int64, shape=[1, WORD_COUNT])


cbow = tf.Variable(tf.random_uniform([50000, 100], -1.0, 1.0), name="cbow")
cbowL = tf.nn.embedding_lookup(cbow, ids=train_inputL)
cbowR = tf.nn.embedding_lookup(cbow, ids=train_inputR)

average = tf.reduce_mean([cbowL, cbowR], 0)
prediction = add_layer(average, input_tensors=100, output_tensors=WORD_COUNT, activation_function=tf.nn.softmax)

loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_hat)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)



for iteration in range(1):
    print(iteration)
    for did, doc in enumerate(DOCUMENT):
        for wid, word in enumerate(doc):
            if wid < WINDOW_SIZE or wid > (len(doc)-WINDOW_SIZE-1):
                continue
            # l_onehot = onehot(doc[wid - WINDOW_SIZE])
            # r_onehot = onehot(doc[wid + WINDOW_SIZE])
            ans = onehot(word)
            sess.run(train, feed_dict={train_inputL: [doc[wid - WINDOW_SIZE]], train_inputR: [doc[wid + WINDOW_SIZE]], y_hat: [ans]})
        print(str(did) + '/2256')
        # print(sess.run(prediction, feed_dict={train_inputL: [doc[wid - WINDOW_SIZE]], train_inputR: [doc[wid + WINDOW_SIZE]], y_hat: [ans]}))
        break

saver = tf.train.Saver({"cbow":cbow})
save_path = saver.save(sess, "/work/IR/IR-HW5-CBOW/save/test.ckpt")
print("Model saved in file: %s" % save_path)
#


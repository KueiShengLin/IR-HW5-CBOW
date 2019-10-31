import os
import re
import math
import numpy as np
from Vector_Space_Model import VSM

DOC_NAME = os.listdir("Document")  # Document file name
QUERY_NAME = os.listdir("Query")  # Query file name

QUERY = []
DOCUMENT = []
VOC_DICT = {}
ALL_WORD = {}
OLD = 0.4
NEW = 0.6
# RANK = []


def readfile():
    global QUERY, DOCUMENT,ALL_WORD
    global QUERY_NAME, DOC_NAME
    voc_id = 0
    # read document , create dictionary
    for doc_id in DOC_NAME:
        doc_dict = {}
        with open("Document\\" + doc_id) as doc_file:
            doc_file_content = doc_file.read()
            doc_voc = re.split(' |\n', doc_file_content)
            doc_voc = list(filter('-1'.__ne__, doc_voc))
            doc_voc.remove('')
            for dv_id, dv_voc in enumerate(doc_voc):
                if dv_id < 5:
                    continue
                if dv_voc in doc_dict:
                    doc_dict[dv_voc] += 1
                else:
                    doc_dict[dv_voc] = 1
            if '' in doc_dict:  # ? error
                doc_dict.pop('')

            for voc in doc_dict:
                if str(voc) not in VOC_DICT:
                    VOC_DICT[str(voc)] = voc_id
                    voc_id += 1
                    ALL_WORD[str(voc)] = 1
                else:
                    ALL_WORD[str(voc)] += 1

        DOCUMENT.append(doc_dict)

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


def ans_read(ans):

    ans_list = []
    with open(ans) as ans_file:
        for line in ans_file:
            if line == 'Query,RetrievedDocuments\n':
                continue
            ans_name = re.split(',| ', line)
            ans_name.remove('\n')
            ans_name.pop(0)
            ans_list.append(ans_name)
    return ans_list


def old_vsm():
    global VOC_DICT
    global d_tfidf, q_tfidf
    old_q, old_d = [], []
    old_q_dis, old_d_dis = [], []
    for q in q_tfidf:
        q_seq = [0] * len(VOC_DICT)
        for qw in q:
            if qw not in VOC_DICT:
                continue
            q_seq[VOC_DICT[qw]] = q[qw]
        old_q.append(q_seq)
        a = np.sum(pow(qv, 2) for qv in q_seq)
        old_q_dis.append(math.sqrt(a))

    for d in d_tfidf:
        d_seq = [0] * len(VOC_DICT)
        for dw in d:
            d_seq[VOC_DICT[dw]] = d[dw]
        old_d.append(d_seq)
        b = np.sum(pow(dv, 2) for dv in d_seq)
        old_d_dis.append(math.sqrt(b))

    print('old_vs, down')
    return old_q, old_d, old_q_dis, old_d_dis


def VSMcos():
    global QUERY, DOCUMENT, EMBEDDING, VOC_DICT, ALL_WORD, NEW, OLD
    q_vector_list = []
    q_vectordis_list = []
    d_vector_list = []
    d_vectordis_list = []
    old_q, old_d, old_q_dis, old_d_dis = old_vsm()

    # query vector calculate
    for qid, q in enumerate(QUERY):
        q_vector = np.zeros([100])
        q_len = sum(q.values())
        for qw in q:
            if str(qw) not in VOC_DICT:
                continue
            ew = VOC_DICT[str(qw)]
            q_vector += (q[qw] / q_len) * EMBEDDING[ew]

        # add pseudo relevant document
        for sudo in sudo_relevant[qid]:
            sudo_doc = DOCUMENT[DOC_NAME.index(sudo)]
            sudo_doc_len = sum(sudo_doc.values())
            for sudo_id, sudo_doc_w in enumerate(sudo_doc):
                sew = VOC_DICT[str(sudo_doc_w)]
                q_vector += (sudo_doc[sudo_doc_w] / sudo_doc_len) * EMBEDDING[sew]

        # if q_vector doesn't have word in word embedding, give it the most frequency word
        if q_vector[0] == 0:
            inverse = [(value, key) for key, value in ALL_WORD.items()]
            top = max(inverse)[1]
            q_vector = EMBEDDING[VOC_DICT[top]]

        a = np.sum(pow(qv, 2) for qv in q_vector)
        q_vector_dis = math.sqrt(a)

        q_vector_list.append(q_vector)
        q_vectordis_list.append(q_vector_dis)
    print('q_vector down')

    # document vector calculate
    for did, doc in enumerate(DOCUMENT):
        d_vector = np.zeros([100])
        d_len = sum(doc.values())
        for dw in doc:
            ew = VOC_DICT[str(dw)]
            d_vector += (doc[dw] / d_len) * EMBEDDING[int(ew)]
        b = np.sum(pow(dv, 2) for dv in d_vector)
        d_vector_dis = math.sqrt(b)

        d_vector_list.append(d_vector)
        d_vectordis_list.append(d_vector_dis)
    print('d_vector down')

    # calculate cos (dsim = embedding, old_sim = hw1 ) NEW, OLD are weights, sim is final score
    for qvid, qv in enumerate(q_vector_list):
        sim = []
        for dvid, dv in enumerate(d_vector_list):
            dsim = dot(qv, dv) / (q_vectordis_list[qvid] * d_vectordis_list[dvid])
            k = (old_q_dis[qvid] * old_d_dis[dvid])
            if k == 0:
                k = 1
            old_dsim = dot(old_d[dvid], old_q[qvid]) / k
            sim.append(NEW * dsim + OLD * old_dsim)
        if qvid % 100 == 0:
            print(qvid)
        RANK.append(sim)


def dot(K, L):
    if len(K) != len(L):
        return 0

    return sum(i[0] * i[1] for i in zip(K, L))


def write_file(name):
    global QUERY_NAME, DOC_NAME, RANK
    with open('./relevant/' + name + '.txt', 'w') as retrieval_file:
        retrieval_file.write("Query,RetrievedDocuments\n")

        for retrieval_id, retrieval_list in enumerate(RANK):
                retrieval_file.write(QUERY_NAME[retrieval_id] + ',')
                sort = sorted(retrieval_list, reverse=True)
                for sort_list in sort[0:100]:
                    retrieval_file.write(DOC_NAME[retrieval_list.index(sort_list)] + ' ')
                if retrieval_id != len(QUERY_NAME) - 1:
                    retrieval_file.write('\n')


readfile()
# tf-tid vector spcae model
VSM_re = VSM(doc_name=DOC_NAME, query_name=QUERY_NAME, document=DOCUMENT, query=QUERY, rank_amount=5)
VSM_re.df_measure()
d_tfidf, q_tfidf = VSM_re.tf_idf_LNIF_RF()
old_vsm()

sudo_relevant = ans_read('VSM5.txt')

# tf-idf + cbow embedding + pseudo relevant doc
for cbow_id in range(240,241):
    RANK = []
    EMBEDDING = np.loadtxt('./embedding/cbowADG_dict' + str(cbow_id) + '.txt', delimiter=',')
    VSMcos()
    print('VSM down')
    write_file('cbowADG_dict' + str(cbow_id))
    print('write_file:' + 'cbowADG_dict' + str(cbow_id))

print('process end')
#

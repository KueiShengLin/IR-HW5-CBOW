import os
import re
import math
import numpy as np
from Vector_Space_Model import VSM
import copy
DOC_NAME = os.listdir("Document")  # Document file name
QUERY_NAME = os.listdir("Query")  # Query file name

QUERY = []
DOCUMENT = []
VOC_DICT = {}
ALL_WORD = {}
OLD = 0.6
NEW = 0.4
NEW_SUDO = 0.2
ROC_ALPHA = 0.7
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



def VSMcos():
    global QUERY, DOCUMENT, EMBEDDING, VOC_DICT, ALL_WORD
    global NEW, OLD, old_cos

    q_vector_list = []
    q_vectordis_list = []
    d_vector_list = []
    d_vectordis_list = []

    for qid, q in enumerate(QUERY):
        q_vector = np.zeros([100])
        q_len = sum(q.values())
        for qw in q:
            if str(qw) not in VOC_DICT:
                continue
            ew = VOC_DICT[str(qw)]
            q_vector += ((1-NEW_SUDO) * (q[qw] / q_len) * EMBEDDING[ew])

        # pseudo relevant
        for sudo in sudo_relevant[qid]:
            sudo_doc = DOCUMENT[DOC_NAME.index(sudo)]
            sudo_doc_len = sum(sudo_doc.values())
            for sudo_id, sudo_doc_w in enumerate(sudo_doc):
                sew = VOC_DICT[str(sudo_doc_w)]
                q_vector += (NEW_SUDO * (sudo_doc[sudo_doc_w] / sudo_doc_len) * EMBEDDING[sew])

        if q_vector[0] == 0:
            inverse = [(value, key) for key, value in ALL_WORD.items()]
            top = max(inverse)[1]
            q_vector = EMBEDDING[VOC_DICT[top]]

        a = np.sum(pow(qv, 2) for qv in q_vector)
        q_vector_dis = math.sqrt(a)

        q_vector_list.append(q_vector)
        q_vectordis_list.append(q_vector_dis)
    print('q_vector down')

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

    for qvid, qv in enumerate(q_vector_list):
        sim = []
        for dvid, dv in enumerate(d_vector_list):
            dsim = dot(qv, dv) / (q_vectordis_list[qvid] * d_vectordis_list[dvid])
            sim.append(NEW * dsim + OLD * old_cos[qvid][dvid])
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


def relevant_doc(first):
    global QUERY
    relevant_doc = []
    relevant_doc_word = []
    new_q = []

    for d_list in first:
        doc_dict = []
        total_voc = {}
        for d_name in d_list:
            temp = copy.deepcopy(DOCUMENT[DOC_NAME.index(d_name)])
            doc_dict.append(temp)
            for temp_word in temp:
                if temp_word not in total_voc:
                    total_voc[temp_word] = temp[temp_word]
                else:
                    total_voc[temp_word] += temp[temp_word]

        relevant_doc.append(doc_dict)     # query k 的 relevant doc
        relevant_doc_word.append(total_voc)     # query k relevant doc all word

    # for q_id, q in enumerate(QUERY):
    #     pseudo = q.copy()
    #
    #     for doc in relevant_doc[q_id]:
    #         for relevant_word in doc:
    #             if relevant_word not in pseudo:
    #                 pseudo[relevant_word] = doc[relevant_word]
    #             else:
    #                 pseudo[relevant_word] += doc[relevant_word]
    #     new_q.append(pseudo)
    return relevant_doc_word


readfile()
# VSM_re = VSM(doc_name=DOC_NAME, query_name=QUERY_NAME, document=DOCUMENT, query=QUERY, rank_amount=10)
# VSM_re.calculate()
# VSM_re.writeAns('VSM10')
sudo_relevant = ans_read('VSM//VSM10.txt')

print('pseudo relevant down')

# tf-tid vector spcae model
rel_doc = relevant_doc(sudo_relevant)
VSM_re = VSM(doc_name=DOC_NAME, query_name=QUERY_NAME, document=DOCUMENT, query=QUERY)
VSM_re.df_measure()
d_tfidf, q_tfidf, rel_tfidf = VSM_re.tf_idf_with_rel(rel_doc)
ans, old_cos = VSM_re.ROCCHIO(d_tfidf, q_tfidf, rel_tfidf, 0.6, 0.4, 1)
# VSM_re.ans = ans
# VSM_re.writeAns('qqqqq')

# tf-idf + cbow embedding + pseudo relevant doc
for cbow_id in range(2740, 2741):
    RANK = []
    EMBEDDING = np.loadtxt('./embedding/cbowADG_dict' + str(cbow_id) + '.txt', delimiter=',')
    VSMcos()
    print('VSM down')
    write_file('no1' + str(15))
    print('write_file:' + 'no1' + str(cbow_id))

print('process end')
#

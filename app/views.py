# 根据__name__，分别发挥不同的作用
# 作为模块时是用来搭建检索系统 
# 作为脚本时是用来计算nDCG

from django.shortcuts import render
from elasticsearch import Elasticsearch
if __name__ == '__main__':
    from utils import *
else:
    from app.utils import *
import random
import os
import numpy as np
from tqdm import tqdm

# 确认你已在本地启动了Elasticsearch
# https://www.elastic.co/guide/en/elasticsearch/reference/current/starting-elasticsearch.html
es = Elasticsearch(hosts="http://localhost:9200")
quotes = load_json('data/corpus.json')

# 建立法律文书检索系统需要的索引
sys_name = "law_query"
print("starting system")
if not es.indices.exists(index=sys_name):
    print("start indexing")
    index_body = {
        "settings": {
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "qw": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart"
                },
                "ajName": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart"
                },
                "cpfxgc": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart"
                },
                "pjjg": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart"
                },
                "ajjbqk": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart"
                },
            }
        }
    }
    es.indices.create(index=sys_name, body=index_body)
    docu_path = "dataset/data/data/documents"
    docus = os.listdir(docu_path)
    for docu in tqdm(docus, desc="Indexing for first start"):
        if docu.endswith(".json"):
            docu_id = docu[:-5]
            docu_body = load_json(os.path.join(docu_path, docu))
            es.index(index=sys_name, body=docu_body, id=docu_id)


def index(request):
    if request.method == 'POST' and 'query' in request.POST:
        query = request.POST['query']
        # 调用选定的方法
        results = improved_api(query)

        return render(request, 'index.html', {'results': results})
    else:
        return render(request, 'index.html', {'results': []})


def random_api(query):
    return random.sample(quotes, 100)


def bm25_api(query):
    # baseline实现：
    #   建立索引时，使用es默认的BM25相似度，对空格分词的名句使用whitespace作为analyzer；
    #   查询时，对query进行空格分词，然后作为match的参数进行查询。
    # TODO: 实现BM25相似度查询，在此处调用es的api（已建好表）

    # 基本的查询，只使用全文进行查询
    print("当前的query: "+query)
    query_body = {
        "query": {
            "match": {
                "qw": {
                    "query": query,
                    "analyzer": "ik_smart"
                }
            }
        },
        "size": 30
    }
    res = es.search(index=sys_name, body=query_body)
    print(res)
    res = [{"content": hit['_source']['ajName'],'id':hit['_id']}
           for hit in res['hits']['hits']]
    return res


def improved_api(query):
    # TODO: 实现你的改进算法，使之在测试集上性能超过baseline。
    # 例如:
    #   算法挑选查询中的重要词，移除不重要的词（包括去停用词）；
    #   对BM25的前k个结果进行rerank，在BM25公式中使用调和平均，使与query具有更多共同词的名言排在更前面；
    #   结合字和词的混合检索策略；
    #   更多其他改进策略。

    # 改进的查询，按照不同的字段进行查询，然后对结果进行加权求和
    print("当前的query: "+query)
    query_body = {
        "query": {
            "function_score": {
                "query": {
                    "match": {
                        "qw": {
                            "query": query,
                            "analyzer": "ik_smart"
                        }
                    }
                },
                "functions": [
                    {
                        "filter": {"match": {"ajName": query}},
                        "weight": 4
                    },
                    {
                        "filter": {"match": {"cpfxgc": query}},
                        "weight": 3
                    },
                    {
                        "filter": {"match": {"pjjg": query}},
                        "weight": 3
                    },
                    {
                        "filter": {"match": {"ajjbqk": query}},
                        "weight": 1
                    }
                ],
                "score_mode": "sum",
                "boost_mode": "multiply"
            }
        },
        "size": 30
    }
    res = es.search(index=sys_name, body=query_body)
    print(res)
    res = [{"content": hit['_source']['ajName'],'id':hit['_id']}
           for hit in res['hits']['hits']]
    return res

# 用于展示全文内容
def detail(request, result_id):
    res=es.get(index=sys_name,id=result_id)['_source']['qw']
    res={"content":res}
    return render(request=request,template_name='detail.html',context={'result':res})


# 以下用于计算nDCG
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # 是否改进方法
    parser.add_argument('--mode', type=str, default='baseline',choices=['baseline', 'improved'],
                        help='baseline/improved')
    
    args=parser.parse_args()

    def get_top30_golden_labels(processed: bool = False):
        # 得到前30的标注值，按照标注值的降序排列
        # 返回id_to_label以及golden_labels
        if processed:
            golden_path = "dataset/data/data/top30_golden_labels.json"
            label_path = "dataset/data/data/label_top30_dict.json"
            with open(label_path, 'r') as f:
                label_dict = json.load(f)
            with open(golden_path, 'r') as g:
                golden_labels = json.load(g)
            return label_dict, golden_labels
        else:
            label_path = "dataset/data/data/label_top30_dict.json"
            save_path = "dataset/data/data/top30_golden_labels.json"
            with open(label_path, 'r') as f:
                label_dict = json.load(f)
                golden_labels = {key: sorted(value.values(), reverse=True)
                                 for key, value in label_dict.items()}
            with open(save_path, 'w') as f:
                json.dump(golden_labels, f)
            return label_dict, golden_labels

    def compute_nDCG(res, labels, id_to_label):
        # 计算nDCG@k
        # res: 检索结果，id值，list
        # labels: 标注值，按照降序排列，list
        # id_to_label: id到标注值的映射，没有的项表示0，dict

        idcg_values = [labels[i]/np.log2(i+2) for i in range(30)]
        idcg_5 = sum(idcg_values[:5])
        idcg_10 = sum(idcg_values[:10])
        idcg_30 = sum(idcg_values)

        dcg_values = [id_to_label[res[i]] /
                      np.log2(i+2) if res[i] in id_to_label.keys() else 0 for i in range(30)]

        dcg_5 = sum(dcg_values[:5])
        dcg_10 = sum(dcg_values[:10])
        dcg_30 = sum(dcg_values)

        ndcg_5 = dcg_5/idcg_5
        ndcg_10 = dcg_10/idcg_10
        ndcg_30 = dcg_30/idcg_30

        return (ndcg_5, ndcg_10, ndcg_30)

    candidates_path = "dataset/data/data/candidates"
    query_path = "dataset/data/data/query.json"
    ndcg_path = "dataset/data/data/ndcg_baseline.txt" if args.mode=='baseline' else "dataset/data/data/ndcg_improved.txt"

    query_ids = os.listdir(candidates_path)
    query_ids = [query_id for query_id in query_ids if os.path.isdir(
        os.path.join(candidates_path, query_id))]
    index_body = {
        "settings": {
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "qw": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart"
                },
                "ajName": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart"
                },
                "cpfxgc": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart"
                },
                "pjjg": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart"
                },
                "ajjbqk": {
                    "type": "text",
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart"
                },
            }
        }
    }

    # 为每个query建立索引
    for query_id in tqdm(query_ids, desc="Indexing"):
        index_name = "query_"+query_id
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, body=index_body)
            candidates = os.listdir(os.path.join(candidates_path, query_id))
            candidates = [os.path.join(candidates_path, query_id, candidate)
                          for candidate in candidates]
            for candidate in candidates:
                body = load_json(candidate)
                candidate_id = candidate.split("/")[-1][:-5]
                es.index(index=index_name, body=body, id=candidate_id)

    id_to_label, golden_labels = get_top30_golden_labels()
    querys = load_json(query_path)

    ndcg_5 = []
    ndcg_10 = []
    ndcg_30 = []

    # 下面可以用于计算不同参数下的nDCG
    # for a in range(4,6):
    #     for b in range(0,4):
    #         for c in range(0,4):
    #             for d in range(0,4):
    #                 ndcg_path = f"dataset/data/data/ndcg_improved{a}{b}{c}{d}.txt"

    # 进行查询
    for query in tqdm(querys, desc="Querying"):
        if args.mode=='baseline':
            query_body = {
                "query": {
                    "match": {
                        "qw": {
                            "query": query['q'],
                            "analyzer": "ik_smart"
                        }
                    }
                },
                "size": 30
            }
        else:
            query_body = {
                "query": {
                    "function_score": {
                        "query": {
                            "match": {
                                "qw": {
                                    "query": query['q'],
                                    "analyzer": "ik_smart"
                                }
                            }
                        },
                        "functions": [
                            {
                                "filter": {"match": {"ajName": query['q']}},
                                "weight": 4 # a
                            },
                            {
                                "filter": {"match": {"cpfxgc": query['q']}},
                                "weight": 3 #b
                            },
                            {
                                "filter": {"match": {"pjjg": query['q']}},
                                "weight": 3 #c,
                            },
                            {
                                "filter": {"match": {"ajjbqk": query['q']}},
                                "weight": 1 #d
                            }
                        ],
                        "score_mode": "sum",
                        "boost_mode": "sum"
                    }
                },
                "size": 30
            }

        res = es.search(index="query_"+str(query['ridx']), body=query_body)
        res = [hit['_id'] for hit in res['hits']['hits']]

        ndcg = compute_nDCG(
            res, golden_labels[str(query['ridx'])], id_to_label[str(query['ridx'])])
        ndcg_5.append(ndcg[0])
        ndcg_10.append(ndcg[1])
        ndcg_30.append(ndcg[2])

    # 记录nDCG
    with open(ndcg_path, 'w') as w:
        w.write("nDCG@5: "+str(sum(ndcg_5)/len(ndcg_5))+"\n")
        w.write("nDCG@10: "+str(sum(ndcg_10)/len(ndcg_10))+"\n")
        w.write("nDCG@30: "+str(sum(ndcg_30)/len(ndcg_30))+"\n")
        w.write("\n")

        w.write("nDCG@5: "+str(ndcg_5)+"\n")
        w.write("nDCG@10: "+str(ndcg_10)+"\n")
        w.write("nDCG@30: "+str(ndcg_30)+"\n")

from django.shortcuts import render
from elasticsearch import Elasticsearch
from app.utils import *
import random

# 确认你已在本地启动了Elasticsearch
# https://www.elastic.co/guide/en/elasticsearch/reference/current/starting-elasticsearch.html
es = Elasticsearch(hosts="http://localhost:9200")
quotes = load_json('data/corpus.json')


def index(request):
    if request.method == 'POST' and 'query' in request.POST:
        query = request.POST['query']
        # 调用选定的方法
        results = bm25_api(query)

        return render(request, 'index.html', {'results': results})
    else:
        return render(request, 'index.html', {'results': []})


def random_api(query):
    return random.sample(quotes, 20)


def bm25_api(query):
    # baseline实现：
    #   建立索引时，使用es默认的BM25相似度，对空格分词的名句使用whitespace作为analyzer；
    #   查询时，对query进行空格分词，然后作为match的参数进行查询。
    # TODO: 实现BM25相似度查询，在此处调用es的api（已建好表）
    
    return random_api(query)


def improved_api(query):
    # TODO: 实现你的改进算法，使之在测试集上性能超过baseline。
    # 例如:
    #   算法挑选查询中的重要词，移除不重要的词（包括去停用词）；
    #   对BM25的前k个结果进行rerank，在BM25公式中使用调和平均，使与query具有更多共同词的名言排在更前面；
    #   结合字和词的混合检索策略；
    #   更多其他改进策略。
    return random_api(query)

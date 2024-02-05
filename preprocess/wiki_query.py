from SPARQLWrapper import SPARQLWrapper,JSON
import pandas as pd
import time
import global_var
import numpy as np
import pandas as pd
import re
import requests
import json
from bs4 import BeautifulSoup

global_var._init()

def label2id(label):
    if type(label) == list:
        label = label[0]
    url = 'https://www.wikidata.org/w/api.php?action=wbsearchentities&search='+label+'&language=en&limit=10&format=json'
    r=requests.get(url)
    # soup = BeautifulSoup(r.text)
    if not len(json.loads(r.text)['search']):
        return False
    qid = json.loads(r.text)['search'][0]['id']
    return qid


def get_tail_entity(eid,rid):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # From https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples#Cats
    sparql.setQuery("""
            SELECT ?item ?itemLabel
            WHERE
            {{
                wd:{} wdt:{} ?item .
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
            }}
            LIMIT 500
                    """.format(eid,rid))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.json_normalize(results['results']['bindings'])
    if len(results_df) == 0:
        return np.array([])
    return results_df[['item.value','itemLabel.value']].values

def get_head_entity(eid,rid):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # From https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples#Cats
    sparql.setQuery("""
            SELECT ?item ?itemLabel
            WHERE
            {{
                ?item wdt:{} wd:{} .
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
            }}
            LIMIT 500
                    """.format(rid,eid))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.json_normalize(results['results']['bindings'])
    return results_df[['item.value','itemLabel.value']].values

def get_rela_entity(eid,head):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # From https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples#Cats
    if head:
        sparql.setQuery("""
            SELECT ?rela ?item
            WHERE
            {{
                wd:{} ?rela ?item .
            }}
            LIMIT 500
                    """.format(eid))
    else:
        sparql.setQuery("""
            SELECT ?rela ?item
            WHERE
            {{
                ?rela  ?item wd:{}.
            }}
            LIMIT 500
                    """.format(eid))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.json_normalize(results['results']['bindings'])
    # time.sleep(1)
    if len(results_df):
        return results_df[['rela.value','item.value']]
    else:
        return [False]

def get_rela_only(eid):
    res = {}
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # From https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples#Cats
    sparql.setQuery("""
            SELECT ?item
            WHERE
            {{
                ?head ?item wd:{} .
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en"}}
            }}
            LIMIT 500
                    """.format(eid))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.json_normalize(results['results']['bindings'])
    tail_rela = []
    for itemlabel in results_df[['item.value']].values:
        pid = itemlabel[0][itemlabel[0].find('P'):]
        if len(pid) < 2:
            continue
        label = qid2label(pid)
        if label != None:
            tail_rela.append(label)
        time.sleep(0.5)
    res['tail'] = tail_rela
    sparql.setQuery("""
            SELECT ?item
            WHERE
            {{
                wd:{} ?item ?tail .
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
            }}
            LIMIT 500
                    """.format(eid))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.json_normalize(results['results']['bindings'])
    head_rela = []
    for itemlabel in results_df[['item.value']].values:
        pid = itemlabel[0][itemlabel[0].find('P'):]
        if len(pid) < 2:
            continue
        label = qid2label(pid)
        time.sleep(0.5)
        if label != None:
            tail_rela.append(label)
    res['head'] = head_rela
    return res

def qid2label(eid):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # From https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples#Cats
    sparql.setQuery("""
            SELECT *
            WHERE
            {{
                wd:{} rdfs:label ?label .
                FILTER (langMatches( lang(?label), "EN" ) ) 
            }}
                    """.format(eid))
    sparql.setReturnFormat(JSON)
    while True:
        try:
            results = sparql.query().convert()
            break
        except BaseException:
            print('wait for request')
            time.sleep(60)
 
    results_df = pd.json_normalize(results['results']['bindings'])
    # time.sleep(5)
    if len(results_df) != 0:
        global_var.set_value(results_df[['label.value']].values[0][0],eid)
        return results_df[['label.value']].values[0][0]
    
def my_entity_search(entity, relation, head):

    rid = global_var.get_value(relation)
    if not rid or rid == "Not Found!":
        return [], []
    rid_str = rid
    # rid_str = rid.pop()
    if head:
        entities_set = get_head_entity(entity,rid_str)
    else:
        entities_set = get_tail_entity(entity,rid_str)

    if len(entities_set) == 0:
        return [],[]
    id_list = [entities_set[0][0][entities_set[0][0].find('Q'):]]
    name_list = [entities_set[0][1]]

    return id_list, name_list

def search_rela_entity(eid,head):
    entity_set = get_rela_entity(eid,head)
    if type(entity_set) != pd.DataFrame:
        return [],[]
    entities = []
    triples = []
    for itemlabel in entity_set.values:
        pid_loc = re.search(r'P\d+',itemlabel[0])
        if pid_loc == None:
            continue
        pid = pid_loc.group()

        qid_loc = re.search(r'Q\d+',itemlabel[1])
        if qid_loc == None:
            continue
        qid = qid_loc.group()
        
        # qid = qid[:itemlabel[1].find('/')]
        if len(pid) < 2 or len(qid) < 2:
            continue
        entities.append(qid)
        triples.append([pid,qid])

    return entities,triples



if __name__ == "__main__":
    get_rela_entity("Q146")
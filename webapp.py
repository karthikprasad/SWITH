###
# Technopedia kappa
#   - a web application to access Technopedia
#
###

from urllib import quote as encode_uri
from urllib import unquote as decode_uri
import feedparser as fp

from technopedia.apps import kappa

from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/about/")
def about_page():
    return render_template("about.html")

def get_bing_results(q):
    q = encode_uri(q)
    feed_url = "http://www.bing.com/search?format=rss&count=5&q="+q
    feed = fp.parse(feed_url)
    return feed["items"]

def get_so_results(q):
    q = encode_uri(q)
    feed_url = "http://www.bing.com/search?format=rss&count=5&q=site%3Astackoverflow.com%20"+q
    feed = fp.parse(feed_url)
    return feed["items"]

def get_optimized_facts(q):
    conj_query = None
    query_words = q.split(" ")

    if len(query_words) == 1:
        conj_query = "?var1 ?var2 ?var3 . ?var1 <http://www.w3.org/2000/01/rdf-schema#label> \""+q+"\""

    elif len(query_words) == 2:
        q1 = query_words[0]
        q2 = query_words[1]
        if q1 == "type":
            conj_query = "?var1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?var2 . ?var1 <http://www.w3.org/2000/01/rdf-schema#label> \""+q2+"\""
        elif q2 == "type":
            conj_query = "?var1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?var2 . ?var1 <http://www.w3.org/2000/01/rdf-schema#label> \""+q1+"\""

        elif q1 == "returns":
            conj_query = "?var1 <http://www.pes.edu/predicate/returns> ?var2 . ?var2 <http://www.w3.org/2000/01/rdf-schema#label> \""+q2+"\""
    
    return conj_query


def get_facts(q):
    conj_query = get_optimized_facts(q)
    if conj_query is not None:
        sparql_query = kappa._get_sparql_query(conj_query)
        facts = kappa.sparql_to_facts(sparql_query)
        if facts is None:
            facts = [{"sub":{"label":"NO", "uri":""}, "pred":{"label":"RESULT", "uri":""}, "obj":{"label":"FOUND", "uri":""}}]
    else:
        sparql_queries = kappa.topk(q,3)
        facts = kappa.sparql_to_facts(sparql_queries[0])
        if facts is None:
            facts = kappa.sparql_to_facts(sparql_queries[1])
        if facts is None:
            facts = kappa.sparql_to_facts(sparql_queries[2])
        if facts is None:
            facts = [{"sub":{"label":"NO", "uri":""}, "pred":{"label":"RESULT", "uri":""}, "obj":{"label":"FOUND", "uri":""}}]
    return facts

def get_conj_facts(conj):
    facts.kappa.conj_to_facts(conj)
    return facts, conj

@app.route("/results")
def results():
    q = request.args.get("q", "")
    res_type = request.args.get("type", "all")

    if res_type == "facts":
        facts = get_facts(q)
        return render_template("results.html", 
            facts=facts, bing_results=None, so_results=None)

    elif res_type == "bing":
        bings_results = get_bing_results(q)
        return render_template("results.html", 
            facts=None, bing_results=bings_results, so_results=None)

    elif res_type == "stackoverflow":
        so_results = get_so_results(q)
        return render_template("results.html", 
            facts=None, bing_results=None, so_results=so_results)

    elif res_type == "all":
        facts,inter = get_facts(q)
        bings_results = get_bing_results(q)
        so_results = get_so_results(q)
        return render_template("results.html", 
            facts=facts, bing_results=bings_results, so_results=so_results)


if __name__ == "__main__":
    app.run(debug=True)
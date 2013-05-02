###
# Technopedia
#   - a REST API access to Technopedia
#
###

from technopedia import data

from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

@app.route("/")
def main():
    res_type = request.args.get("responsetype", None)
    
    if res_type == "subjects":
        pred = request.args.get("predicate", None)
        obj = request.args.get("object", None)
        return data.subjects(predicate=pred, object=obj, format="json")

    elif res_type == "objects":
        sub = request.args.get("subject", None)
        pred = request.args.get("predicate", None)
        return data.objects(subject=sub, predicate=pred, format="json")

    elif res_type == "predicates":
        sub = request.args.get("subject", None)
        obj = request.args.get("object", None)
        return data.predicates(subject=sub, object=obj, format="json")

    elif res_type == "triples":
        sub = request.args.get("subject", None)
        pred = request.args.get("predicate", None)
        obj = request.args.get("object", None)
        return data.triples(subject=sub, predicate=pred, object=obj, format="json")

    elif res_type == "literals":
        sub = request.args.get("subject", None)
        pred = request.args.get("predicate", None)
        return data.literals(subject=sub, predicate=pred, format="json")

    elif res_type == "sparql":
        q = request.args.get("q", None)
        return data.query(q)


if __name__ == "__main__":
    app.run(port=8000,debug=True)
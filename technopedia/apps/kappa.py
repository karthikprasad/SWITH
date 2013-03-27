import networkx as _nx
from technopedia import data

# INDEXING GRAPH DATA

def _make_keyword_index():
	"""
	function which makes a keyword index - a mapping from
	keyword to keyword-element via predicate-label/aedge.

	@return:
		dictionary of form {"keyword":[(predicate-labeli, [Ci1, Ci2, Ci3,...]),
									   (predicate-labelj, [Cj1, Cj2, Cj3,...]),
									   (...)
									  ]
							}

	"""
	keyword_index = {}

	# get a list of Value nodes(literals)
	vvertices = data.literals()["response"]

	# for each keyword
	for vvertex in vvertices:
		value = []

		# get list of predicate labels for a given literal
		aedges = data.predicates(object=vvertex)["response"]

		# for each predicate labels
		for aedge in aedges:
			# get a list of entity nodes associated with literal and predicate
			enodes = data.subjects(object=vvertex, predicate=aedge)

			# for each of the entity nodes, obtain all the class nodes
			cnodes = []
			for enode in enodes:
				cnodes += class_type(enode)
			cnodes = set(cnodes)

			# aedge=predicate-labeli, cnodes=[Ci1, Ci2,...]
			value.append((aedge, cnodes))

		keyword_index[vvertex] = value

	return keyword_index



###########################################
# PREPROCESSED DATASTRUCTURES
_keyword_index = _make_keyword_index()
_summary_graph = _make_summary_graph()

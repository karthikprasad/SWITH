import networkx as _nx
import collections as _coll

from technopedia import data

#### PRE-PROCESSING FUNCTIONS ####
# INDEXING GRAPH DATA

def _get_vertex_keyword_index():
	"""
	function which makes a keyword index - a mapping from
	keyword to keyword-element_vertex(V-vertex) via predicate-label/aedge.

	@return:
		dictionary of form {"keyword":[(predicate-labeli, [Ci1, Ci2, Ci3,...]),
									   (predicate-labelj, [Cj1, Cj2, Cj3,...]),
									   (...)
									  ]
							}

	"""
	keyword_index = _coll.defaultdict(list)

	# get a list of V-vertices(possible keywords)
	vnodes = data.literals()["response"]

	for vnode in vnodes:
		# get list of predicate labels(A-edges) for a given keyword
		aedges = data.predicates(object=vvertex)["response"]

		for aedge in aedges:
			# get a list of entity nodes associated with keyword and predicate
			enodes = data.subjects(object=vvertex, predicate=aedge)

			# for each of the entity nodes, obtain all the class nodes
			cnodes = []
			for enode in enodes:
				cnodes += _class_type(enode)
			cnodes = list(set(cnodes))

			# aedge=predicate-labeli, cnodes=[Ci1, Ci2,...]
			tuple_i = (aedge, cnodes)
			# append this tuple to the keyword index list
			keyword_index[vvertex].append(tuple_i)

	return keyword_index



def _get_edge_keyword_index():
	"""
	function which makes a keyword index - a mapping from
	keyword to keyword-element-edge(A-edge).

	@return:
		dictionary of form {"keyword":[(predicate-labeli, [Ci1, Ci2, Ci3,...]),
									   (predicate-labelj, [Cj1, Cj2, Cj3,...]),
									   (...)
									  ]
							}

	"""
	# to be implemented by aparna
	return {}



def _make_keyword_index():
	"""
	function to obtain a keyword index with both V-vertices and A-edges considered

	@return:
		dictionary of form {"keyword":[(predicate-labeli, [Ci1, Ci2, Ci3,...]),
									   (predicate-labelj, [Cj1, Cj2, Cj3,...]),
									   (...)
									  ]
							}

	"""
	nodes_dict = _get_vertex_keyword_index()
	edges_dict = _get_edge_keyword_index()
	# return the merged dictionaries
	return dict(edges_dict, **nodes_dict)



def _make_summary_graph():
	summary_graph = nx.MultiDiGraph(label="summary graph")

	cnodes = _get_all_class_nodes()
	for cnode in cnodes:
		summary_graph.add_node(cnode, cost=_cost(cnode), cursors=[])
	# add BNode and Thing Class
	summary_graph.add_node("BNode", cost=_cosr("Bnode"), cursor=[])
	summary_graph.add_node("Thing", cost=_cosr("Thing"), cursor=[])

	




def _get_all_class_nodes():
	return _class_type()



def _cost(node):
	"""
	function which returns the cost associated with the node

	one of the scoring(cost) functions from section 5 of the paper

	"""
	# to be implemented by aparna
	return 1



def _class_type(enode=None):
	"""
	function to obtain types of given entity node(E-vertex)

	@param:
		enode :: as string
	If None given, all class nodes will be returned

	@return:
		list of class nodes to which given enode belongs.

	"""
	type_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
	class_type = data.objects(subject=enode, predicate=type_predicate)["response"]

	if len(class_type) <= 0:
		if data.Term.type(enode) == "BNode":
			class_type = ["BNode"]
		else:
			class_type = ["Thing"]

	return class_type



###########################################
# PREPROCESSED DATASTRUCTURES
_keyword_index = _make_keyword_index()
_summary_graph = _make_summary_graph()

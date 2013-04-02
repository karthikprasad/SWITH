import networkx as _nx
import collections as _coll
import itertools as _it

from technopedia import data

######## GRAPH ELEMENT ########

class _GE:
    """
    a node is of form ("node", key)
    an edge is of form ("edge", node1, node2, key)
    """
    graph = None
    cnodes = None  # independently obtained
    vnodes = None  # independently obtained
    redges = None  # dependent on cnodes
    aedges = None  # dependent on vnodes

    ### functions asscoiated with GE object
    def __init__(self,ele_type,key,n1=None,n2=None,sub_type=None):
        self.type = ele_type  # Grpah element type: node or edge
        self.key = key  # Grpah element label (URI or literal)
        if self.type == "edge":
            self.n1 = n1  # label of one vertex of edge
            self.n2 = n2  # label of another vertex of edge
        self.sub_type = sub_type  # "c" or "v" for node; "a" or "r" for edge


    @property
    def cost(self):
        """
        returns the cost associated with the graph element

        """
        if self.type == "node":
            return self.graph.node[self.key]["cost"]
        elif self.type == "edge":
            return self.graph.edge[self.n1][self.n2][self.key]["cost"]
        return 0


    @property
    def cursors(self):
        """
        returns the dictionary of cursors associated with the graph element

        """
        if self.type == "node":
            return self.graph.node[self.key]["cursors"]
        elif self.type == "edge":
            return self.graph.edge[self.n1][self.n2][self.key]["cursors"]


    @property
    def neighbours(self):
        """
        returs the list of neighbour Graph elements.
        returns list of connected edges if self is a node.
        returns list of adjacent nodes (two nodes) if self is an edge.
        """
        neighbours = []
        if self.type == "node":
            connected_edges = self.graph.in_edges([self.key], keys=True) + 
                self.graph.out_edges([self.key], keys=True)
            # GEfy edges
            for edge in connected_edges:
                n1 = edge[0]
                n2 = edge[1]
                key = edge[2]
                neighbours.append(_GE("edge",key,n1,n2))
        elif self.type == "edge":
            neighbours.append(_GE("node",self.n1))
            neighbours.append(_GE("node",self.n2))
        return neighbours


    def add_cursor(self,c):
        """
        add the given cursor to an appropriate slot in the elements cursors dict

        """
        if self.type == "node":
            self.graph.node[self.key]["cursors"][c.i].append(c)
        elif self.type == "edge":
            self.graph.edge[self.n1][self.n2][self.key]["cursors"][c.i].append(c)


    ### static functions
    @staticmethod
    def class_types(enode=None):
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


    @staticmethod
    def get_all_class_nodes():
        return _GE.class_types()


    @staticmethod
    def get_all_value_nodes():
        return data.literals()["reponse"]


    @staticmethod
    def entity_nodes(cnode=None):
        """
        function to obtain entity nodes of given class node(C-vertex)
        @param:
            cnode::as string

        @return:
            list of entity nodes to which given cnode belongs.

        """
        type_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        entity_nodes = data.subjects(object=cnode, predicate=type_predicate)["response"]
        return list(set(entity_nodes))


    @staticmethod
    def get_all_edges():
        redges = set()
        aedges = set()
        triples = data.triples()
        for row in triples:
            edge = (row[0], row[2])  #edge = (subj, obj)
            if _GE.is_aedge(edge):
                n2 = row[2]
                key = row[1]
                # obtain a list of class nodes of the entity node(subject)
                for n1 in _GE.class_types(row[0]):
                    aedges.append((n1,n2,key))

            elif _GE.is_redge(edge):
                # make bunch of node pairs
                # by taking the cartesian product of each type of subj and obj
                node_pairs = _it.product(_GE.class_types(row[0]), 
                    _GE.class_types(row[2]))
                for pair in node_pairs:
                    regdes.append((pair[0], pair[1], row[1]))

        return list(redges), list(aedges)


    @staticmethod
    def is_enode(node):
        """
        function to determine if the node is an enode
        @param:
        node : a node whose type should be checked to be E-node
        
        @return:
            boolean value

        depends on _GE.cnodes
        """
        if data.Term.type(node) == "BNode":
            return True
        elif data.Term.type(node) == "URI" and node not in _GE.cnodes
            return True
        return False


    @staticmethod
    def is_cnode(node):
        """
        function to determine if the node is an cnode
        @param:
        node : a node whose type should be checked to be C-node
        
        @return:
            boolean value

        depends on _GE.cnodes
        """
        return node in _GE.cnodes


    @staticmethod
    def is_vnode(node):
        """
        function to determine if the node is an vnode
        @param:
        node : a node whose type should be checked to be V-node
        
        @return:
            boolean value

        depends on _GE.vnodes
        """
        return node in _GE.vnodes


    @staticmethod
    def is_redge(edge):
        """
        function to determine if the edge is an R-edge
        @param:
            edge : an edge whose type should be checked to be R-edge
        
        @return:
            boolean value

        depends on is_enode and hence _GE.cnodes
        """
        return _GE.is_enode(edge[0]) and _GE.is_enode(edge[1])


    @staticmethod
    def is_aedge(edge):
        """
        function to determine if the edge is an A-edge
        @param:
            edge : an edge whose type should be checked to be A-edge
        
        @return:
            boolean value

        depends on self.vnodes
        can be made independent
        """
        return edge[1] in _GE.vnodes


## Graph Element Initilaization
_GE.cnodes = _GE.get_all_class_nodes()
_GE.vnodes = _GE.get_all_value_nodes()
_GE.redges, _GE.aedges = _GE.get_all_edges()

#############################################################################################################################
#############################################################################################################################


######## SECTION 4 - INDEXING GRAPH DATA ########
### KEYWORD INDEXING ###

def get_keyword_index():
    """
    function which maps keyword to Graph Elements (_GE objects)
    the graph elements are cnodes, vnodes, redges and aedges

    @return
        index dictionary of form {"keyword":[list of _GE objects],...}

    """
    index = _coll.defaultdict(list)

    for cnode in _GE.cnodes:
        keywords = _extract_keywords(cnode)
        for keyword in keywords:
            index[keyword].append(_GE("node",cnode,sub_type="c"))

    for vnode in _GE.vnodes:
        keywords = _extract_keywords(vnode)
        for keyword in keywords:
            index[keyword].append(_GE("node",vnode,sub_type="v"))

    for redge in _GE.redges:
        n1 = redge[0]
        n2 = redge[1]
        key = redge[2]
        keywords = _extract_keywords(redge)
        for keyword in keywords:
            index[keyword].append(_GE("edge",key,n1,n2,sub_type="r"))

    for aedge in _GE.aedges:
        n1 = aedge[0]
        n2 = aedge[1]
        key = aedge[2]
        keywords = _extract_keywords(aedge)
        for keyword in keywords:
            index[keyword].append(_GE("edge",key,n1,n2,sub_type="a"))

    return index


def _extract_keywords(uri):

    """ 
    function that fetches keywords from URI
    @param:
        uri : the uri that needs to be processed to fetch keywords
    @return:
        returns a list of keywords keyword_list = [k1,k2,k3,..kn]

    """
            
    # fetching the last keywords part of the predicate URI
    aedge_split = aedge.split("/")
    keywords_token_aedge = aedge_split[len(aedge_split)-1]
        
    # seperating the keywords and cleaning them
    keyword_list = keywords_token_aedge.split(".")
    keyword_list_length = len(keyword_list)
    for keyword_list_index in range(0,keyword_list_length):
        keyword_list[keyword_list_index] = keyword_list[keyword_list_index].replace("_"," ")
        #checking for the hash part of the URI 
        if re.search(r'[a-z]+#[a-z]+',keyword_list[keyword_list_index]):
            hash_position = keyword_list[keyword_list_index].index("#")
            if hash_position > 0 :
                cap_words_positions = re.search(r'[A-Z]+[a-z]*',keyword_list[keyword_list_index][hash_position+1:])
                if cap_words_positions != None:
                    #print cap_words_positions.span()
                    if len(cap_words_positions.span()) >1:
                        sub_keyword = keyword_list[keyword_list_index][hash_position+1:cap_words_positions.span()[0]+hash_position+1]
                        for cap_word_position in range(0,len(cap_words_positions.span())-1):
                            sub_keyword  = sub_keyword +  " " + keyword_list[keyword_list_index][(hash_position+1+cap_words_positions.span()[cap_word_position]):(hash_position+1+cap_words_positions.span()[cap_word_position+1])].lower()
                            keyword_list.append(sub_keyword)
                else:
                    sub_keyword = keyword_list[keyword_list_index][hash_position+1:]
                    keyword_list.append(sub_keyword)

                keyword_list[keyword_list_index] = keyword_list[keyword_list_index][:hash_position]

        else:
            cap_words_positions = re.search(r'[A-Z]+[a-z]*',keyword_list[keyword_list_index])
            if cap_words_positions != None:
                #print cap_words_positions.span()
                if len(cap_words_positions.span()) >1:
                    sub_keyword = keyword_list[keyword_list_index][:cap_words_positions.span()[0]]
                    for cap_word_position in range(0,len(cap_words_positions.span())-1):
                        sub_keyword  = sub_keyword +  " " + keyword_list[keyword_list_index][(cap_words_positions.span()[cap_word_position]):(cap_words_positions.span()[cap_word_position+1])].lower()
                    
                    keyword_list[keyword_list_index] = sub_keyword
                                                
        #print keyword_list
        return keyword_list


### GRAPH SCHEMA INDEXING ###

def _get_summary_graph():
    summary_graph = _nx.MultiDiGraph(label="summary graph")

    for cnode in _GE.cnodes:
        summary_graph.add_node(cnode, cost=None, cursors=_coll.defaultdict(list))
    # add BNode and Thing Class
    summary_graph.add_node("BNode", cost=None, cursors=_coll.defaultdict(list))
    summary_graph.add_node("Thing", cost=None, cursors=_coll.defaultdict(list))
    
    for redge in _GE.redges:
        summary_graph.add_edge(redge[0], redge[1], key=redge[2],
                    cost=None, cursors=_coll.defaultdict(list))

    return summary_graph


#############################################################################################################################
#############################################################################################################################


######## SECTION 5 - SCORING ########
def _attach_costs(graph):
    """
    function which attaches the cost to every node and edge of the graph
    @return graph with costs attached

    """
    graph = _attach_node_costs(graph)
    graph = _attach_edge_costs(graph)
    return graph


def _attach_node_costs(graph):
    """
    function which attaches the cost to every node of the graph
    @return graph with node costs attached

    """
    total_number_of_nodes = graph.number_of_nodes()
    for n in graph.nodes_iter():
        graph.node[n]["cost"] = _get_node_cost(n, graph, total_number_of_nodes)
    return graph


def _get_node_cost(node, graph, total_number_of_nodes):
    """
    function which returns the cost associated with the node
    popularity score: section 5
    @param:
        node: node in the summary graph
        graph: a graph to which the edge belongs to
        total_number_of_nodes: total_number_of_nodes in the graph
    @return:
        a score in the range 0-1
    """
    return 1 - len(_get_all_entity_nodes(node)/(total_number_of_nodes +0.0)


def _attach_edge_costs(graph):
    """
    function which attaches the cost to every edge of the graph
    @return graph with edge costs attached

    """
    total_number_of_edges = graph.number_of_edges()
    for e in graph.edges_iter(keys=True):
        n1 = e[0]
        n2 = e[1]
        key = e[2]
        graph.edge[n1][n2][key]["cost"] = 
            _get_edge_cost(e, graph, total_number_of_edges)
    return graph
    
    
def _get_edge_cost(edge, graph, total_number_of_edges):
    """
    function which returns the cost associated with the node
    popularity score :section 5

    @param:
    edge: edge from the summary graph
    graph: a graph to which the edge belongs to
    total_number_of_edges: total_number_of_edges in the graph
    @return:
    a score in the range 0-1
    """
    eedge_count = 0
    n1 = edge[0]
    n2 = edge[1]
    adjacent_edges = graph.edges([n1,n2], keys=True)
    adjacent_edges.remove(edge)

    for neighbour_edge in adjacent_edges:
        if _is_redge(neighbour_edge):
            eedge_count + = 1
    return 1 - eedge_count/(total_number_of_edges+0.0)


def _get_subgraph_cost(subgraph,summary_graph):
    """
    function which returns the cost of the subgraph
    @param:
        subgraph: the subgraph for which cost needs to be computed
        summary_graph: the main summary graph
    @return:
        a cumilative score
    """
    cumilative_cost = 0.0
    total_number_of_nodes = summary_graph.number_of_nodes()
    total_number_of_edges = summary_graph.number_of_edges()
    for node in subgraph.nodes():
        cumilative_cost += _get_node_cost(node, summary_graph, total_number_of_nodes)
    for edge in subgraph.edges(keys=True):
        cumilative_cost += _get_edge_cost(edge, summary_graph, total_number_of_edges)
    return cumilative_cost


#############################################################################################################################
#############################################################################################################################


######## SECTION 6 - QUERY INTERPRETATION ########

def _get_keyword_elements(keyword_list):
    """
    function returns a list of list of keyword elements given
    a list of keywords.

    uses the _keyword_index data structure

    @param
        keyword_list :: list of keywords(strings)
        [keyword1, keyword2, ...]

    @return
        K :: list of list of _GE objects (keyword elements)
        [[_GE objects for keyword1], [_GE objects for keyword2], ...]

    """
    K = []
    for keyword in keyword_list:
        K.append(_keyword_index[keyword])
    return K


def _make_augmented_graph(K):
    """
    function which makes an augmented summary graph.

    takes a copy of the _summary_graph and adds vnodes and aedges to it.
    attaches the augmented graph to _GE.graph

    @param
        K :: list of list of _GE objects (keyword elements)

    @return
        aug_graph :: networkx graph with keyword elements attached

    """
    aug_graph = <get summry graph copy>
    for Ki in K:
        for ele in Ki:
            # if element is a V-Node
            if ele.type == "node" and ele.sub_type == "v":
                aug_graph.add_node(ele.key, cost=1, cursors=_coll.defaultdict(list))

                # get list of aedges associated with the given vnode(literal)
                aedges = [edge for edge in _GE.aedges if edge[1] == ele.key]
                for aedge in aedges:
                    aug_graph.add_edge(aedge[0], aedge[1], key=aedge[2],
                        cost=1, cursors=_coll.defaultdict(list))
                        # NOTE: aedge[1] is same as ele.key
            
            # else if element is A-edge        
            elif ele.type == "edge" and ele.sub_type == "a":
                aug_graph.add_edge(ele.n1, ele.n2, key=key,
                    cost=1, cursors=_coll.defaultdict(list))

    return aug_graph


class _Cursor:
    def __init__(self,n,k,i,p,c,d):
        self.graph_element = n  # _GE
        self.keyword_element = k  # _GE
        self.i = i  # keyword number
        self.parent = p  # _Cursor
        self.cost = c  # int
        self.distance = d  # int

    def __cmp__(self,other):
        if self.cost < other.cost:
            return -1
        elif self.cost > other.cost:
            return 1
        else:
            return 0

    @property
    def ancestors(self):
        ancestor_list = []
        p = self.parent
        while p is not None:
            ancestor_list.append(p)
            p = p.parent
        return ancestor_list


def _alg1(num, dmax, aug_graph, K):
    m = len(K)
    LQ = []
    LG = [] # global var from paper
    R = []

    i = 0
    for Ki in K:
        i += 1
        for k in Ki:
            heapq.heappush(LQ, _Cursor(k,k,i,None,k.cost,0))

    # while LQ not empty
    while len(LQ) > 0:
        c = heapq.heappop(LQ)
        n = c.graph_element
        if c.distance < dmax:
            n.add_cursor(c)
            neighbours = n.neighbours
            neighbours.remove(c.parent) # reomove the parent from list
            # if neighbours not empty
            if len(neighbours) > 0:
                for neighbour in neighbours:
                    # take care of cyclic paths
                    if neighbour not in c.ancestors:
                        # add new cursor to LQ
                        heapq.heappush(LQ, _Cursor(neighbour, c.keyword_element,
                            c.i, n, c.cost+neighbour.cost, c.distance+1))
            R,LG = top_k(n,LG,LQ,num,R)

    return R




#############################################################################################################################
#algo2
##############################################################################################################################	
def is_connected(self,m):
        """
        checks if the graph element is connected to all the keywords along the
        path already visited which is tracked by the cursors.

        self is a connecting element if all self.Ci are not empty, i.e. 
        for every keyword i, there is at least one cursor

        @param
            m :: no of keywords as int
        @return
            boolean value

        """
        # obtain a list of list of cursors
        # each list within a list represents a list of cursors from keyword i.
        list_of_lists = self.cursors.values()

        # first check if self has been visited from all m keywords
        if len(list_of_lists) == m:
            # check if self is still connected to all the m keywords
            # check if there is atleast one cursor in the path to each keyword
            # i.e., length of every list within the list is >0
            num_keywords_connected = sum([1 for i in list_of_lists if len(i)>0])

            # self is connected if it is connected to all the m keywords
            if num_keywords_connected == m:
                return True
        # self is not connected if it has not even been visited by all keywords
        return False



def _cursor_combinations(n):
    """
    function to obtain a combination of paths to graph element - n,
    originating from different keywords.

    each element has a list of list of cursors.
    each element will have m lists; one list for each keyword.
    each of these lists may have more than one cursor to the keyword i.
    each cursor represents a path from element to keyword i.

    this function therefore returns a list of tuples where each tuple has a 
    one cursor to each keyword. the length of every tuple is therefore m.

    the reurned list represents all possible subgraphs from node 
    to each of the keywords along all combinations of paths
    
    @param:
        n :: graph element
    @return:
        list of list of m-cusor paths 
        [[c11,c12,..c1m],[c21,c22,..c2m],...[ck1,ck2,..ckm]]

    """
    list_of_lists = n.cursors.values()
    return _it.product(*list_of_lists)
	
def _build_all_paths(all_combinations_list):
	"""
	A function that builds a set of paths for all the combinatios
	
	@param: 
		all_combinations_list:: A list of list of m-cusor paths 
	@return:
		A list of list of m-graph paths 
	"""
	subgraphs_path_collection = []
	for m_cursor_set in all_combinations_list:
		subgraph_paths =[]
		for cursor in m_cursor_set:
			subgraph_paths.append(_build_path_from_cursor(cursor))
		subgraphs_path_collection.append(subgraph_paths)
	return subgraphs_path_collection

def _build_path_from_cursor(cursor):
	"""
	function to obtain a path from a cursor
	
	@param:
		cursor : A cursor to the node from were we need to create a path
	
	@return:
		A path from keyword element to the end node, it is of type mulitDiGraph
	"""
	
	destination = cursor.keyword
	source = cursor.graph_element
	path = nx.MultiDiGraph(label="path_"+source.key+"_to_"+destination.key)
	current_cursor = cursor
	while current_cursor != None:
		if current_cursor.type == "node":
			path.add_node(current_cursor.graph_element.key, cost = current_cursor.graph_element.cost)
		else:
			path.add_edge(current_cursor.graph_element.n1.key,current_cursor.graph_element.n2.key,current_cursor.graph_element.key,current_node_cursor.graph_element.cost)
		current_cursor = current_cursor.parent
		
	return path
	
def _merge_paths_to_graph(paths):
	"""
	function to merge paths to a subgraph
	@param:
		A list of paths
	@return:
		A merged multiDiGraph
	"""
	subgraph = _nx.MultiGraph()
	for path in paths:
		subgraph = _nx.compose(path,subgraph)
	return subgraph

def _update_cost_of_subgraphs(subgraph_list):

	cost_agumented_subgraph_list = []
	for subgraph in subgraph_list:
		cost = _get_subgraph_cost(subgraph,_summary_graph)
		agumented_tuple = (subgraph,cost)
		cost_agumented_subgraph_list.append(agumented_tuple)
	return cost_agumented_subgraph_list
		
def _choose_top_k_sub_graphs(cost_agumented_subgraph_list,k):
	cost_agumented_subgraph_list.sort(key=ret_cost)
	return cost_agumented_subgraph_list[len(cost_agumented_subgraph_list)-k-1:]
	
def _ret_cost(a):
	return a[1]
	


#############################################################################################################################
#############################################################################################################################


# PREPROCESSED DATASTRUCTURES
#_keyword_index = _get_keyword_index()
#_summary_graph = _get_summary_graph()
#_summary_graph = _attach_costs(_summary_graph)


if __name__ == "__main__":
        import matplotlib.pyplot as plt
        g = _get_summary_graph()
        _nx.draw(g)
        plt.show()




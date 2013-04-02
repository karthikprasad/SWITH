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
        self.type = ele_type
        self.key = key
        if self.type == "edge":
            self.n1 = n1
            self.n2 = n2
        self.sub_type = sub_type


    @property
    def cost(self):
        if self.type == "node":
            return self.graph.node[self.key]["cost"]
        elif self.type == "edge":
            return self.graph.edge[self.n1][self.n2][self.key]["cost"]
        return 0


    @property
    def neighbours(self):
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
        if self.type == "node":
            self.graph.node[self.key]["cursors"].append(c)
        elif self.type == "edge":
            self.graph.edge[self.n1][self.n2][self.key]["cursors"].append(c)


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



### GRAPH SCHEMA INDEXING ###

def _get_summary_graph():
    summary_graph = _nx.MultiDiGraph(label="summary graph")

    for cnode in _GE.cnodes:
        summary_graph.add_node(cnode, cost=None, cursors=[])
    # add BNode and Thing Class
    summary_graph.add_node("BNode", cost=None, cursors=[])
    summary_graph.add_node("Thing", cost=None, cursors=[])
    
    for redge in _GE.redges:
        summary_graph.add_edge(redge[0], redge[1], key=redge[2],
                    cost=None, cursors=[])

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
                aug_graph.add_node(ele.key, cost=None, cursors=[])

                # get list of aedges associated with the given vnode(literal)
                aedges = [edge for edge in _GE.aedges if edge[1] == ele.key]
                for aedge in aedges:
                    aug_graph.add_edge(aedge[0], aedge[1], key=aedge[2],
                        cost=None, cursors=[])
                        # NOTE: aedge[1] is same as ele.key
            
            # else if element is A-edge        
            elif ele.type == "edge" and ele.sub_type == "a":
                aug_graph.add_edge(ele.n1, ele.n2, key=key, 
                    cost=None, cursors=[])

    return aug_graph


class _Cursor:
    def __init__(self,n,k,p,c,d):
        self.graph_element = n  # _GE
        self.keyword = k  # _GE
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
    for Ki in K:
        for k in Ki:
            heapq.heappush(LQ, _Cursor(k,k,None,k.cost,0))

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
                        heapq.heappush(LQ, _Cursor(neighbour,c.keyword,n,
                            c.cost+neighbour.cost, c.distance+1))
            R,LG = top_k(n,LG,LQ,num,R)

    return R


#############################################################################################################################
#############################################################################################################################


# PREPROCESSED DATASTRUCTURES
#_keyword_index = _get_keyword_index()
#_summary_graph = _get_summary_graph()
#_summary_graph = _attach_costs(_summary_graph)


if __name__ == "__main__":
        import matplotlib.pyplot as plt
        g = _make_summary_graph()
        _nx.draw(g)
        plt.show()




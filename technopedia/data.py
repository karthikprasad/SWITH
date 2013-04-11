"""
    An interface to access the technopedia dataset which is loaded 
    on a mysql database.
    It creates an rdflib graph and store and initialises them.
    Effectively wrapping itself around the rdflib objects.

"""
import re as _re

import rdflib as _rb
import rfc3987 as _rfc
import simplejson as _simplejson


# database name and connection string
_NAME = "javadocs"
_CONN_STR = "host=localhost,user=root,password=root"

# global variable (module-level)
_graph = None


# initialization function
def __init__(name=_NAME, connection_str=_CONN_STR):
    """
    Initialises the graph by loading the database onto it.
    @param: 
        id :: identifier for the Technopedia object
        conn_str :: To connect to the database in format 
                    host=address,user=name,password=pass,db=tech_fb

        """
    conn_str = connection_str+",db="+name
    
    # create store
    store = _rb.plugin.get("MySQL", _rb.store.Store)(name)

    # connect store to the database
    store.open(conn_str)

    # graph will be accesed by all the functions
    global _graph
    # load technopedia to rdflib graph
    _graph = _rb.ConjunctiveGraph(store)



try:
    # Initiliaze (by calling function)
    __init__()
except:
    raise ImportError("Unable to make connection to database!")
else:

    def _sparql_query(query_string, initNs={}, initBindings={}):
        """
        Private method to execute a SPARQL query on technopedia.
        Calls rdflib graph.query() internally.

        Handles errors and exceptions without exposing rdflib.

        @param: 
            query_string :: sparql query string
        
        @return:
            SPARQL result as rdflib_sparql.processor.SPARQLResult object.

        """
        #try:
        sparql_result = _graph.query(query_string, initNs=initNs, 
                                                initBindings=initBindings)
        #except:
        #    # handle No query or wrong query syntax errors
        #    raise SparqlError("query execution failed")

        return sparql_result



    def _stringify_sparql_result(res, format):
        """
        Prefix bnode value with _:
        Example: N69bdff2a33874675b1b02 => _:N69bdff2a33874675b1b02

        """
        if format == "json":
            prefixed_res = res.replace('"type": "bnode", "value": "', 
                                        '"type": "bnode", "value": "_:')
        elif format == "xml":
            prefixed_res = res.replace('<sparql:bnode>', '<sparql:bnode>_:')

        return prefixed_res



    def query(query_string, format="json"):
        """
        Method to execute a SPARQL query on technopedia.
        @param: 
            query_string :: sparql query string
            format :: format of the spqrql reslut; json or xml
        
        @return:
            SPARQL result in xml or json format as required. Default is json.

        """
        # get rdflib_sparql.processor.SPARQLResult object
        result = _sparql_query(query_string)
        
        try:
            response = result.serialize(format=format)
        except:
            # handle unregistered format Exception
            raise SparqlError("format not supported")
        else:
            #stringify the bnode of sparql result
            response = _stringify_sparql_result(response, format)

        return response



    def subjects(predicate=None, object=None, format="python"):
        """
        List of subjects for given predicate and/or object.

        @param:
            predicate :: as string
            object :: as string
        If none given, all subjects will be returned

        @return:
            a python dictionary when format="python" (default)
            JSON object when format="json"

            The return value is of the form:
                {
                  "responsetype": "subjects",
                  "object": "object_value",
                  "predicate": "predicate_value",
                  "response": [list of subjects in str format]
                }

        """
        ##convert the arguments to suitable rdflib terms
        t_predicate = Term._termify_predicate(predicate)
        t_object = Term._termify_object(object)

        # obtain subjects generator.
        gen = _graph.subjects(predicate=t_predicate, object=t_object)

        # make a list of string subjects
        subjects_list = list(set([Term._stringify(s) for s in gen]))
        #create a response object
        dic = {
                "responsetype": "subjects",
                "object": object,
                "predicate": predicate,
                "response": subjects_list
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return _simplejson.dumps(dic)



    def predicates(subject=None, object=None, format="python"):
        """
        List of predicates for given subject and/or object.

        @param:
            subject :: as string
            object :: as string
        If none given, all predicates will be returned

        @return:
            a python dictionary when format="python" (default)
            JSON object when format="json"

            The return value is of the form:
                {
                  "responsetype": "predicates",
                  "subject": "subject_value",
                  "object": "object_value",
                  "response": [list of predicates in str format]
                }

        """
        ##convert the arguments to suitable rdflib terms
        t_subject = Term._termify_subject(subject)
        t_object = Term._termify_object(object)

        # obtain predicate generator.
        gen = _graph.predicates(subject=t_subject, object=t_object)

        # make a list of string predicates
        predicates_list = list(set([Term._stringify(p) for p in gen]))
        #create a response object
        dic = {
                "responsetype": "predicates",
                "subject": subject,
                "object": object,
                "response": predicates_list
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return _simplejson.dumps(dic)



    def objects(subject=None, predicate=None, format="python"):
        """
        List of objects for given predicate and/or subject.

        @param:
            subject :: as string
            predicate :: as string
        If none given, all objects will be returned

        @return:
            a python dictionary when format="python" (default)
            JSON object when format="json"

            The return value is of the form:
                {
                  "responsetype": "objects",
                  "subject": "subject_value",
                  "predicate": "predicate_value",
                  "response": [list of objects in str format]
                }

        """
        ##convert the arguments to suitable rdflib terms
        t_subject = Term._termify_object(subject)
        t_predicate = Term._termify_predicate(predicate)

        # obtain object generator. 
        gen = _graph.objects(predicate=t_predicate, subject=t_subject)

        # make a list of string objects
        objects_list = list(set([Term._stringify(o) for o in gen]))
        #create a response object
        dic = {
                "responsetype": "objects",
                "subject": subject,
                "predicate": predicate,
                "response": objects_list
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return _simplejson.dumps(dic)



    def contexts(subject=None, predicate=None, object=None, format="python"):
        """
        List of contexts in which given subject and/or predicate and/or object appear

        @param:
            subject :: as string
            predicate :: as string
            object :: as string
        If none given, all contexts will be returned

        @return:
            a python dictionary when format="python" (default)
            JSON object when format="json"

            The return value is of the form:
                {
                  "responsetype": "contexts",
                  "subject": "subject_value",
                  "object": "object_value",
                  "predicate": "predicate_value",
                  "response": [list of contexts in str format]
                }

        """
        ##convert the arguments to suitable rdflib terms
        t_subject = Term._termify_object(subject)
        t_predicate = Term._termify_predicate(predicate)
        t_object = Term._termify_object(object)

        # obtain context generator.
        gen = _graph.contexts(triple=(t_subject,t_predicate,t_object))

        # make a list of string contexts
        # context is a minigraph, so use c.identifier to get the context label
        contexts_list = list(set([Term._stringify(c.identifier) for c in gen]))
        #create a response object
        dic = {
                "responsetype": "contexts",
                "subject": subject,
                "object": object,
                "predicate": predicate,
                "response": contexts_list
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return _simplejson.dumps(dic)



    def triples(subject=None, predicate=None, object=None, context=None, format="python"):
        """
        List of triples in which given subject and/or predicate and/or \
        object and/or context appear.

        @param:
            subject :: as string
            predicate :: as string
            object :: as string
            context :: as string
        If none given, all triples will be returned

        @return:
            a python dictionary when format="python" (default)
            JSON object when format="json"

            The return value is of the form:
                {
                  "responsetype": "triples",
                  "subject": "subject_value",
                  "object": "object_value",
                  "predicate": "predicate_value",
                  "context": "context_value",
                  "response": [list of triples] each triple is a list of string
                }

        """
        ##convert the arguments to suitable rdflib terms
        t_subject = Term._termify_object(subject)
        t_predicate = Term._termify_predicate(predicate)
        t_object = Term._termify_object(object)
        t_context = Term._termify_context(context)

        # obtain context generator.
        gen = _graph.triples((t_subject,t_predicate,t_object), context=t_context)

        # make a list of string triples
        triples_list = [[Term._stringify(row[0]), Term._stringify(row[1]),
                            Term._stringify(row[2])] for row in gen]
        #create a response object
        dic = {
                "responsetype": "triples",
                "subject": subject,
                "object": object,
                "predicate": predicate,
                "context": context,
                "response": triples_list
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return _simplejson.dumps(dic)



    def literals(subject=None, predicate=None, format="python"):
        """
        List of literals for given predicate and/or subject.

        @param:
            subject :: as string
            predicate :: as string
        If none given, all literals will be returned

        @return:
            a python dictionary when format="python" (default)
            JSON object when format="json"

            The return value is of the form:
                {
                  "responsetype": "literals",
                  "subject": "subject_value",
                  "predicate": "predicate_value",
                  "response": [list of literals in str format]
                }

        """
        ##convert the arguments to suitable rdflib terms
        t_subject = Term._termify_object(subject)
        t_predicate = Term._termify_predicate(predicate)

        # obtain object generator. 
        gen = _graph.objects(predicate=t_predicate, subject=t_subject)

        # make a list of literal objects
        literals_list = list(set([Term._stringify(o) for o in gen \
                                            if type(o) is _rb.term.Literal]))
        #create a response object
        dic = {
                "responsetype": "literals",
                "subject": subject,
                "predicate": predicate,
                "response": literals_list
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return _simplejson.dumps(dic)



class Term:
    """
    A class which defines static functions to manipulate term information.

    """
    @staticmethod
    def type(str_term):
        """
        function determine the type of the term
        """
        # get an rdflib term
        obj_term = Term._termify_object(str_term)

        if Term._is_uriref(obj_term):
            type_val = "URI"
        elif Term._is_bnode(obj_term):
            type_val = "BNode"
        elif Term._is_literal(obj_term):
            type_val = "Literal"

        return type_val



    @staticmethod
    def _termify_subject(subject_str):
        """
        internal function which returns an appropriate rdflib term 
        for a given string subject

        """
        if subject_str is None:
            t_subject = None
        else:  # subject is URI or Blank Node
            t_subject = (Term._make_uriref(subject_str) or 
                         Term._make_bnode(subject_str)
                        )
            if not t_subject:
                raise ParseError("Subject must be uri or blank node")
        return t_subject



    @staticmethod
    def _termify_object(object_str):
        """
        internal function which returns an appropriate rdflib term
        for a given string object

        """
        if object_str is None:
            t_object = None
        else:  # object is uri or blank node or literal
            t_object = (Term._make_uriref(object_str) or 
                        Term._make_bnode(object_str) or 
                        Term._make_literal(object_str)
                       )
            if not t_object:
                raise ParseError("Unrecognised object type")
        return t_object



    @staticmethod
    def _termify_predicate(predicate_str):
        """
        internal function which returns an appropriate rdflib term 
        for a given string predicate

        """
        if predicate_str is None:
            t_predicate = None
        else:  # prediacte is URI
            t_predicate = Term._make_uriref(predicate_str)
            if not t_predicate:
                 raise ParseError("Predicate must be uri")
        return t_predicate



    @staticmethod
    def _termify_context(context_str):
        """
        internal function which returns an appropriate rdflib term 
        for a given string context

        """
        if context_str is None:
            t_context = None
        else:  # context is URI
            t_context_id = Term._make_uriref(predicate_str)
            if not t_context_id:
                 raise ParseError("Context is not a URI")
            t_context = _rb.Graph(identifier=t_context_id)
        return t_context



    @staticmethod
    def _make_uriref(string):
        """
        internal function which returns rdflib.term.URIRef if successful;
        returns False otherwise

        """
        uri_pattern = _rfc.format_patterns()["URI"]
        match = _re.compile(uri_pattern).match(string)
        if not match:
            return False
        return _rb.term.URIRef(string.decode("unicode-escape"))        



    @staticmethod
    def _make_bnode(string):
        """
        internal function which returns rdflib.term.BNode if successful;
        returns False otherwise

        """
        bnode_pattern = ur'_:([A-Za-z][A-Za-z0-9]*)'
        match = _re.compile(bnode_pattern).match(string)
        if not match:
            return False
        # else if match occurs,
        string = string[2:]  # remove _: from the string
        return _rb.term.BNode(string.decode("unicode-escape"))       



    @staticmethod
    def _make_literal(string):
        """
        internal function which returns rdflib.term.Literal if successful;
        returns False otherwise

        """
        # for literal string without other info
        lit_pattern = ur'([^"\\]+(?:\\.[^"\\]*)*)'

        # for literal string with other info
        litinfo_name_pattern = ur'"([^"\\]*(?:\\.[^"\\]*)*)"'  # has double quote
        litinfo_info_pattern = ur'(?:@([a-z]+(?:-[a-z0-9]+)*)|\^\^(' + \
                                _rfc.format_patterns()["URI"] + ur'))?'
        litinfo_pattern = litinfo_name_pattern + litinfo_info_pattern

        # try matching both patterns
        match_lit = _re.compile(lit_pattern).match(string)
        match_litinfo = _re.compile(litinfo_pattern).match(string)

        if match_lit:
            lit = match_lit.groups()[0]
            lit = lit.decode("unicode-escape")  # encoding is unicode
            lang = None
            dtype = None

        elif match_litinfo:
            lit, lang, dtype = match_litinfo.groups()
            lit = lit.decode("unicode-escape")  # encoding is unicode
            
            if lang:
                lang = lang.decode("unicode-escape")
            else:
                lang = None
            
            if dtype:
                dtype = dtype.decode("unicode-escape")
            else:
                dtype = None

        else: # if no match
            return False

        return _rb.term.Literal(lit, lang, dtype)



    @staticmethod
    def _is_literal(term):
        return type(term) is _rb.term.Literal


    @staticmethod
    def _is_bnode(term):
        return type(term) is _rb.term.BNode

    
    @staticmethod
    def _is_uriref(term):
        return type(term) is _rb.term.URIRef


    @staticmethod
    def _stringify(term):
        opstring = ""
        if Term._is_uriref(term):
            opstring = term.encode("unicode-escape")
        elif Term._is_bnode(term):
            opstring = "_:"+term.encode("unicode-escape")
        elif Term._is_literal(term):
            opstring = term.encode("unicode-escape")
            if term.language is not None:
                opstring = "\""+opstring+"\"" + "@" + term.language
            if term.datatype is not None:
                opstring += opstring + "^^" + term.datatype
        return opstring



class ParseError(Exception):
    pass



class SparqlError(Exception):
    pass



if __name__ == "__main__":

    '''
    print
    print
    print "SUBJECTS"
    print "========="
    print subjects(object="Driver")
    print
    print
    print "PREDICATES"
    print "========="
    print predicates(object="Driver")
    print
    print
    print "OBJECTS"
    print "========="
    objs = objects(subject="http://docs.oracle.com/javase/7/docs/api/java/applet/AppletContext.html")["response"]
    for obj in objs:
        print obj
    print
    print
    print "LITERALS"
    print "========="
    print literals(subject="http://docs.oracle.com/javase/7/docs/api/java/applet/AppletContext.html")
    print
    print
    print "CONTEXTS"
    print "========="
    print contexts(subject="http://docs.oracle.com/javase/7/docs/api/java/applet/AppletContext.html")
    print
    print
    #print "TRIPLES"
    #print "========="
    #t = triples(predicate="http://www.w3.org/2000/01/rdf-schema#member")["response"]
    #for row in t:
    #    print row
    '''

    print "SPARQL QYERY"
    print "============="
    # doesnt work
    q='''select distinct ?var0 ?var1 where {graph ?g {
        {?var0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.pes.edu/type/package>}.
        {?var0 <http://www.w3.org/2000/01/rdf-schema#member> ?var1}}}'''

    # works
    q='''select distinct ?var0 ?var1 where {graph ?g {
        {?var1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.pes.edu/type/class>}.
        {?var0 <http://www.w3.org/2000/01/rdf-schema#member> ?var1}}}'''

    # doesnt work
    q='''select distinct ?var0 ?var1 where {graph ?g {
        {?var1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.pes.edu/type/class>}.
        {?var1 <http://www.w3.org/2000/01/rdf-schema#label> "Driver"}.
        {?var0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.pes.edu/type/package>}.
        {?var0 <http://www.w3.org/2000/01/rdf-schema#member> ?var1}}}'''

    # works
    q='''select distinct ?var0 ?var1 where {graph ?g {
        {?var1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.pes.edu/type/class>}.
        {?var1 <http://www.w3.org/2000/01/rdf-schema#label> "Driver"}.
        {?var0 <http://www.w3.org/2000/01/rdf-schema#member> ?var1}}}'''

    print query(q)

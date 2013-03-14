import re

import rdflib as rb
import rfc3987 as rfc
import simplejson


class Technopedia:
    """
    An interface to access the technopedia dataset which is loaded 
    on a mysql database.
    It creates an rdflib graph and store and initialises them.
    Effectively wrapping itself around the rdflib objects.

    """

    def __init__(self, id=None, connection_str=None):
        """
        The constructor initialises the graph by loading the database onto it.
        @param: 
            id :: identifier for the Technopedia object
            conn_str :: To connect to the database in format 
                        host=address,user=name,password=pass,db=tech_fb

        """
        self._id = None
        self._name = "tech_fb"
        self._store = None
        self._graph = None
        self._conn_str = None

        #create store
        self._store = rb.plugin.get("MySQL", rb.store.Store)(self._name)

        #set database connection parameters
        if connection_str is None:
            self._conn_str = "host=localhost,user=root,password=root,db=tech_fb"
        else:
            self._conn_str = connection_str

        #connect store to the database
        self._store.open(self._conn_str)

        #load technopedia to rdflib graph
        self._graph = rb.ConjunctiveGraph(store, identifier=self._id)



    def _sparql_query(self, query_string, initNs={}, initBindings={}):
        """
        Private method to execute a SPARQL query on technopedia.
        Calls rdflib graph.query() internally.

        Handles errors and exceptions without exposing rdflib.

        @param: 
            query_string :: sparql query string
        
        @return:
            SPARQL result as rdflib_sparql.processor.SPARQLResult object.

        """
         try:
            sparql_result = self._graph.query(query_string, initNs, initBindings)
        except:
            # handle No query or wrong query syntax errors
            sparql_result = "ERROR: query execution failed"

        return sparql_result



    def query(self, query_string, format="json"):
        """
        Method to execute a SPARQL query on technopedia.
        @param: 
            query_string :: sparql query string
            format :: format of the spqrql reslut; json or xml
        
        @return:
            SPARQL result in xml or json format as required. Default is json.

        """
        # get rdflib_sparql.processor.SPARQLResult object
        result = self._sparql_query(query_string)
        
        try:
            response = result.serialize(format=format)
        except:
            # handle unregistered format Exception
            response = "ERROR: format not supported"
        else:
            #stringify the bnode of sparql result
            response = Technopedia._stringify_sparql_result(response, format)

        return response



    def subjects(self, predicate=None, object=None, format="python"):
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
        t_predicate = Technopedia._termify_predicate(predicate)
        t_object = Technopedia._termify_object(object)

        if Technopedia._is_literal(t_object):
            # sparql query
            ## takes care when the user doesnt give language info is not given
            q = ('select distinct ?s'
                    'where {graph ?g {?s ?p ?o . '
                        'filter regex(?o, "'+ str(t_object) +'")}}')
            bindings = {"?p": t_predicate}

        else:  # when object is bnode or uri
            q = ('select distinct ?s'
                    'where {graph ?g {?s ?p ?o .}}')
            bindings = {"?p": t_predicate, "?o": t_object}

        # obtain tuple generator. first element of each tuple is a subj
        gen = self._sparql_query(q, initBindings=bindings)
        # make a list of string subjects
        subjects_list = [Technopedia._stringify(s[0]) for s in gen]
        
        #create a response object
        dic = {
                "responsetype": "subjects",
                "object": object,
                "predicate": prediacte,
                "response": subjects_list
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return simplejson.dumps(dic)




#########################################################################################
########################################################################################


    @staticmethod
    def _termify_subject(subject_str):
        """
        internal function which returns an appropriate rdflib term 
        for a given string subject

        """
        # subject is URI or Blank Node
        t_subject = (Technopedia._make_uriref(subject_str) or 
                     Technopedia._make_bnode(subject_str
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
        # object is uri or blank node or literal
        t_object = (Technopedia._make_uriref(object_str) or 
                    Technopedia._make_bnode(object_str) or 
                    Technopedia._make_literal(object_str)
                   )
        if t_object is False:
            raise ParseError("Unrecognised object type")
        return t_object



    @staticmethod
    def _termify_predicate(predicate_str):
        """
        internal function which returns an appropriate rdflib term 
        for a given string predicate

        """
        # prediacte is URI
        t_predicate = Technopedia._make_uriref(predicate_str)
        if not t_predicate:
             raise ParseError("Predicate must be uri")
        return t_predicate



    @staticmethod
    def _termify_context(context_str):
        """
        internal function which returns an appropriate rdflib term 
        for a given string context

        """
        # context is URI
        t_context = Technopedia._make_uriref(predicate_str)
        if not t_context:
             raise ParseError("Context must be uri")
        return t_context



    @staticmethod
    def _make_uriref(string):
        """
        internal function which returns rdflib.term.URIRef if successful;
        returns False otherwise

        """
        uri_pattern = rfc.format_patterns()["URI"]
        match = re.compile(uri_pattern).match(string)
        if not match:
            return False
        return rb.term.URIRef(string.decode("unicode-escape"))        



    @staticmethod
    def _make_bnode(string):
        """
        internal function which returns rdflib.term.BNode if successful;
        returns False otherwise

        """
        bnode_pattern = r'_:([A-Za-z][A-Za-z0-9]*)'
        match = re.compile(bnode_pattern).match(string)
        if not match:
            return False
        # else if match occurs,
        string = string[2:]  # remove _: from the string
        return rb.term.BNode(string.decode())  # in unicode encoding        



    @staticmethod
    def _make_literal(string):
        """
        internal function which returns rdflib.term.Literal if successful;
        returns False otherwise

        """
        # for literal string without other info
        lit_pattern = r'([^"\\]*(?:\\.[^"\\]*)*)()()'

        # for literal string with other info
        litinfo_name_pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'  # has double quote
        litinfo_info_pattern = r'(?:@([a-z]+(?:-[a-z0-9]+)*)|\^\^(' + \
                                rfc.format_patterns()["URI"] + r'))?'
        litinfo_pattern = litinfo_name_pattern + litinfo_info_pattern

        # try matching both patterns
        match_lit = re.compile(lit_pattern).match(string)
        match_litinfo = re.compile(litinfo_pattern).match(string)
        
        if match_lit:
            lit, lang, dtype = match_lit.groups()
            lit = lit.decode("unicode-escape")  # encoding is unicode
            lang = None
            dtype = None

        elif match_litinfo:
            lit, lang, dtype = match.groups()
            lit = lit.decode("unicode-escape")  # encoding is unicode
            
            if lang:
                lang = lang.decode()
            else:
                lang = None
            
            if dtype:
                dtype = dtype.decode()
            else:
                dtype = None

        else:
            return False

        return rb.term.Literal(lit, lang, dtype)



    @staticmethod
    def _is_literal(term):
        return type(term) is rb.term.Literal


    @staticmethod
    def _is_bnode(term):
        return type(term) is rb.term.BNode

    
    @staticmethod
    def _is_uriref(term):
        return type(term) is rb.term.URIRef


    @staticmethod
    def _stringify(term):
        if Technopedia._is_uriref(term):
            opstring = str(term)
        elif Technopedia._is_bnode(term):
            opstring = "_:"+str(term)
        elif Technopedia._is_literal(term):
            opstring = str(term)
            if term.language is not None:
                opstring += opstring + "@" + term.language
            if term.datatype is not None:
                opstring += opstring + "^^" + term.language
        return opstring



    @staticmethod
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

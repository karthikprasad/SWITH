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

    def __init__(self, name="tech_fb", connection_str="host=localhost,user=root,password=root"):
        """
        The constructor initialises the graph by loading the database onto it.
        @param: 
            id :: identifier for the Technopedia object
            conn_str :: To connect to the database in format 
                        host=address,user=name,password=pass,db=tech_fb

        """
        self._name = name
        self._conn_str = connection_str+",db="+name
        self._store = None
        self._graph = None

        #create store
        self._store = rb.plugin.get("MySQL", rb.store.Store)(self._name)

        #connect store to the database
        self._store.open(self._conn_str)

        #load technopedia to rdflib graph
        self._graph = rb.ConjunctiveGraph(self._store)



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
            sparql_result = self._graph.query(query_string, initNs=initNs, 
                                                initBindings=initBindings)
        except:
            # handle No query or wrong query syntax errors
            raise SparqlError("query execution failed")

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
            raise SparqlError("format not supported")
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

        bindings = {}

        # if object is a literal (and hence not None)
        ## takes care when the user doesnt give language info is not given
        if Technopedia._is_literal(t_object) and \
            t_object.language is None and t_object.datatype is None:

                if t_predicate is not None:
                    bindings["?p"] = t_predicate
                # get only the name part of the literal  in unicode
                str_t_object = t_object.encode("unicode-escape", "ignore")
                # sparql query
                # use regex to find pattern when lang is unknown
                q = ('select distinct ?s'
                        ' where {graph ?g {?s ?p ?o .'
                        ' filter regex(?o, "'+ str_t_object +'")}}')

        else:  # when object is bnode or uri or complete litearl info
            if t_predicate is not None:
                bindings["?p"] = t_predicate
            if t_object is not None:
                bindings["?o"] = t_object
            # sparql query
            q = ('select distinct ?s'
                    ' where {graph ?g {?s ?p ?o .}}')

        # obtain tuple generator. first element of each tuple is a subj
        gen = self._sparql_query(q, initBindings=bindings)

        # make a list of string subjects
        subjects_list = [Technopedia._stringify(s[0]) for s in gen]
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
            return simplejson.dumps(dic)



    def predicates(self, subject=None, object=None, format="python"):
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
        t_subject = Technopedia._termify_subject(subject)
        t_object = Technopedia._termify_object(object)

        bindings = {}

        # if object is a literal (and hence not None)
        ## takes care when the user doesnt give language info is not given
        if Technopedia._is_literal(t_object) and \
            t_object.language is None and t_object.datatype is None:

                if t_subject is not None:
                    bindings["?s"] = t_subject
                # get only the name part of the literal  in unicode
                str_t_object = t_object.encode("unicode-escape", "ignore")
                # sparql query
                # use regex to find pattern when lang is unknown
                q = ('select distinct ?p'
                        ' where {graph ?g {?s ?p ?o .'
                        ' filter regex(?o, "'+ str_t_object +'")}}')

        else:  # when object is bnode or uri or complete litearl info
            if t_subject is not None:
                bindings["?s"] = t_predicate
            if t_object is not None:
                bindings["?o"] = t_object
            # sparql query
            q = ('select distinct ?p'
                    ' where {graph ?g {?s ?p ?o .}}')

        # obtain tuple generator. first element of each tuple is a predicate
        gen = self._sparql_query(q, initBindings=bindings)

        # make a list of string subjects
        predicates_list = [Technopedia._stringify(p[0]) for p in gen]
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
            return simplejson.dumps(dic)



    def objects(self, subject=None, predicate=None, format="python"):
        """
        List of subjects for given predicate and/or object.

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
        t_subject = Technopedia._termify_object(subject)
        t_predicate = Technopedia._termify_predicate(predicate)

        bindings = {}
        if t_subject is not None:
            bindings["?s"] = t_subject
        if t_predicate is not None:
            bindings["?p"] = t_predicate
        # sparql query
        q = ('select distinct ?o'
                ' where {graph ?g {?s ?p ?o .}}')

        # obtain tuple generator. first element of each tuple is an object
        gen = self._sparql_query(q, initBindings=bindings)

        # make a list of string subjects
        objects_list = [Technopedia._stringify(o[0]) for o in gen]
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
            return simplejson.dumps(dic)



    def contexts(self, subject=None, predicate=None, object=None, format="python"):
        """
        List of contexts in which given subject and/or predicate and/or object. appear

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
        t_subject = Technopedia._termify_object(subject)
        t_predicate = Technopedia._termify_predicate(predicate)
        t_object = Technopedia._termify_object(object)

        bindings = {}

        # if object is a literal (and hence not None)
        ## takes care when the user doesnt give language info is not given
        if Technopedia._is_literal(t_object) and \
            t_object.language is None and t_object.datatype is None:
                if t_subject is not None:
                    bindings["?s"] = t_subject
                if t_predicate is not None:
                    bindings["?p"] = t_predicate
                # get only the name part of the literal  in unicode
                str_t_object = t_object.encode("unicode-escape", "ignore")
                # sparql query
                # use regex to find pattern when lang is unknown
                q = ('select distinct ?g'
                        ' where {graph ?g {?s ?p ?o .'
                        ' filter regex(?o, "'+ str_t_object +'")}}')

        else:  # when object is bnode or uri or complete litearl info
            if t_subject is not None:
                bindings["?s"] = t_subject
            if t_predicate is not None:
                bindings["?p"] = t_predicate
            if t_object is not None:
                bindings["?o"] = t_object
            # sparql query
            q = ('select distinct ?g'
                    ' where {graph ?g {?s ?p ?o .}}')

        # obtain tuple generator. first element of each tuple is a subj
        gen = self._sparql_query(q, initBindings=bindings)

        # make a list of string subjects
        contexts_list = [Technopedia._stringify(c[0]) for c in gen]
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
            return simplejson.dumps(dic)



#############################################################################
############################################################################


    @staticmethod
    def _termify_subject(subject_str):
        """
        internal function which returns an appropriate rdflib term 
        for a given string subject

        """
        if subject_str is None:
            t_subject = None
        else:  # subject is URI or Blank Node
            t_subject = (Technopedia._make_uriref(subject_str) or 
                         Technopedia._make_bnode(subject_str)
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
            t_object = (Technopedia._make_uriref(object_str) or 
                        Technopedia._make_bnode(object_str) or 
                        Technopedia._make_literal(object_str)
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
        if context_str is None:
            t_context = None
        else:  # context is URI
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
        bnode_pattern = ur'_:([A-Za-z][A-Za-z0-9]*)'
        match = re.compile(bnode_pattern).match(string)
        if not match:
            return False
        # else if match occurs,
        string = string[2:]  # remove _: from the string
        return rb.term.BNode(string.decode("unicode-escape"))       



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
                                rfc.format_patterns()["URI"] + ur'))?'
        litinfo_pattern = litinfo_name_pattern + litinfo_info_pattern

        # try matching both patterns
        match_lit = re.compile(lit_pattern).match(string)
        match_litinfo = re.compile(litinfo_pattern).match(string)

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
        opstring = ""
        if Technopedia._is_uriref(term):
            opstring = term.encode("unicode-escape")
        elif Technopedia._is_bnode(term):
            opstring = "_:"+term.encode("unicode-escape")
        elif Technopedia._is_literal(term):
            opstring = term.encode("unicode-escape")
            if term.language is not None:
                opstring = "\""+opstring+"\"" + "@" + term.language
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




class ParseError(Exception):
    pass



class SparqlError(Exception):
    pass



if __name__ == "__main__":
    g = Technopedia("test_nq")
    '''
    print g.subjects()
    print
    print
    s = "Alice"
    print g.subjects(object=s)
    print
    print
    s = '"Alice"@en'
    print g.subjects(object=s)
    print
    print
    s = '_:N54080b9b88ce4d52be67be8d0bfbb008'
    print g.subjects(object=s)
    print
    print
    s = '"\u0E0B\u0E34\u0E01\u0E27\u0E34\u0E19"@th'
    print g.subjects(object=s)
    print
    print
    s = '"\\u0E0B\\u0E34\\u0E01\\u0E27\\u0E34\\u0E19"@th'
    print g.subjects(object=s)
    print
    print
    s = "Bob"
    print g.subjects(object=s)
    print
    print
    '''

    '''
    print g.predicates()
    print
    print
    s = "Alice"
    print g.predicates(object=s)
    print
    print
    s = '"Alice"@en'
    print g.predicates(object=s)
    print
    print
    s = '_:N54080b9b88ce4d52be67be8d0bfbb008'
    print g.predicates(object=s)
    print
    print
    s = '"\u0E0B\u0E34\u0E01\u0E27\u0E34\u0E19"@th'
    print g.predicates(object=s)
    print
    print
    s = '"\\u0E0B\\u0E34\\u0E01\\u0E27\\u0E34\\u0E19"@th'
    print g.predicates(object=s)
    print
    print
    s = "Bob"
    print g.predicates(object=s)
    print
    print
    '''

    '''
    print g.objects()
    print g.objects(subject="http://example.org/alice/foaf.rdf#me")
    print g.objects(predicate="http://xmlns.com/foaf/0.1/name")
    '''

    '''
    print g.contexts()
    print g.contexts(subject="http://example.org/alice/foaf.rdf#me")
    print g.contexts(predicate="http://xmlns.com/foaf/0.1/name")
    print g.contexts(object="Bob")
    print g.contexts(object="Alic")
    print g.contexts(object='"Alice"@fr')
    '''

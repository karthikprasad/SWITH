import simplejson
import rfc3987 as iri

import rdflib as rb

class Technopedia:
    """
    A interface to access the technopedia dataset which is loaded on a mysql database.
    Provides the results in python format for use by other python applications.
    Also provides the results in JSON format.
    """

    def __init__(self, id=None, conn_str=None):
        """
        The constructor creates an rdflib graph and store and initialises them.
        Effectively wrapping itself around the rdflib objects.
        """
        self._id = None
        self._store = None
        self._graph = None
        self._connection_str = None

        #create store
        self._store = rb.plugin.get("MySQL", rb.store.Store)("tech_fb")

        #set database connection parameters
        if conn_str is None:
            self._connection_str = "host=localhost,user=root,password=root,db=tech_fb"
        else:
            self._connection_str = conn_str

        #connect store to the database
        self._store.open(self._connection_str)

        #load technopedia to rdflib graph
        self._graph = rb.ConjunctiveGraph(store, identifier=self._id)



    def triples(self, format="python", subject=None, predicate=None, object_=None, lang="", context=None):
        """
        List of triples for given(any or all) subject, object, predicate and/or context.
        Specify lang if the object is literal, default is empty language.
        If none given, all triples will be returned.

        Function returns a python dictionary when format="python" and JSON object when format="json"

        The return value is of the form:
            {
              "subject":"subject_value",
              "predicate":"predicate_value",
              "object":"object_value",
              "lang": "language",
              "context":"context_value",
              "triples":[list of triples in str format]
            }
        """

        ##convert the arguments to suitable rdflib terms
        t_subject = Technopedia._termify_subject(subject)
        t_predicate = Technopedia._termify_predicate(predicate)
        t_object = Technopedia._termify_object(object_, lang)
        t_context = Technopedia._termify_context(context)

        #obtain a triples generator
        gen = self._graph.triples((t_subject, t_predicate, t_object), t_context)

        #create a response object
        dic = {
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "lang": lang,
                "context": context,
                "triples": [str(i[0])+" "+str(i[1])+" "+str(i[2])+" ." for i in gen]
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return simplejson.dumps(dic)



    def quads(self, format="python", subject=None, predicate=None, object_=None, lang=""):
        """
        List of quads for given(any or all) subject, object and/or predicate.
        Specify lang if the object is literal, default is empty language
        If none given, all quads will be returned.

        Function returns a python dictionary when format="python" and JSON object when format="json"

        The return value is of the form:
            {
              "subject":"subject_value",
              "predicate":"predicate_value",
              "object":"object_value",
              "quads":[list of quads in str format]
            }
        """

        ##convert the arguments to suitable rdflib terms
        t_subject = Technopedia._termify_subject(subject)
        t_predicate = Technopedia._termify_predicate(predicate)
        t_object = Technopedia._termify_object(object_, lang)

        #obtain a quads generator
        gen = self._graph.quads((t_subject, t_predicate, t_object))

        #create a response object
        dic = {
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "lang": lang,
                "quads": [str(i[0])+" "+str(i[1])+" "+str(i[2])+" "+str(i[3])+" ." for i in gen]
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return simplejson.dumps(dic)



    def subjects(self, format="python", predicate=None, object_=None, lang=""):
        """
        List of subjects for given object and/or predicate.
        Specify lang if the object is literal, default is empty language
        If none given, all subjects will be returned

        Function returns a python dictionary when format="python" and JSON object when format="json"

        The return value is of the form:
            {
              "object":"object_value",
              "lang":"language",
              "predicate":"predicate_value",
              "subjects":[list of subjects in str format]
            }
        """

        ##convert the arguments to suitable rdflib terms
        t_predicate = Technopedia._termify_predicate(predicate)
        t_object = Technopedia._termify_object(object_, lang)

        #obtain a subjects generator
        gen = self._graph.subjects(predicate=t_predicate, object=t_object)

        #create a response object
        dic = {
                "object": object_,
                "lang": lang,
                "predicate": predicate,
                "subjects": [str(i) for i in gen]
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return simplejson.dumps(dic)



    def predicates(self, format="python", subject=None, object_=None, lang=""):
        """
        List of predicates for given subject and/or object.
        Specify lang if the object is literal, default is empty language
        If none given, all predicates will be returned

        Function returns a python dictionary when format="python" and JSON object when format="json"

        The return value is of the form:
            {
              "subject":"subject_value",
              "object":"object_value",
              "lang": "language"
              "predicates":[list of predicates in str format]
            }
        """

        ##convert the arguments to suitable rdflib terms
        t_subject = Technopedia._termify_subject(subject)
        t_object = Technopedia._termify_object(object_, lang)

        #obtain a predicates generator
        gen = self._graph.predicates(subject=t_subject, object=t_object)

        #create a response object
        dic = {
                "subject": subject,
                "object": object_,
                "lang": lang,
                "predicates": [str(i) for i in gen]
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return simplejson.dumps(dic)


    def objects(self, format="python", subject=None, predicate=None):
        """
        List of objects for given subject and/or predicate.
        If none given, all objects will be returned.

        Function returns a python dictionary when format="python" and JSON object when format="json"

        The return value is of the form:
            {
              "subject":"subject_value",
              "predicate":"predicate_value",
              "objects":[list of objects in str format]
            }
        """

        ##convert the arguments to suitable rdflib terms
        t_subject = Technopedia._termify_subject(subject)
        t_predicate = Technopedia._termify_predicate(predicate)

        #obtain a objects generator
        gen = self._graph.objects(subject=t_subject, predicate=t_predicate)

        # make object list
        obj_list = []
        for i in gen:
            lang = ""
            if type(i) is rb.term.Literal and i.language != "":
                lang = "@"+i.language
            obj_list.append(str(i)+lang)

        #create a response object
        dic = {
                "subject": subject,
                "predicate": predicate,
                "objects": obj_list
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return simplejson.dumps(dic)



    def contexts(self, format="python"):
        """
        List of all contexts

        Function returns a python dictionary when format="python" and JSON object when format="json"

        The return value is of the form:
            {
              "contexts":[list of contexts in str format]
            }
        """
        dic = {
                "contexts": [str(i) for i in self._graph.contexts()]
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return simplejson.dumps(dic)



    def literals(self, format="python"):
        """
        List of all literals(not uris or metanodes) in technopedia
        
        Function returns a python dictionary when format="python" and JSON object when format="json"

        The return value is of the form:
            {
              "literals":[list of literals in str format]
            }
        """
        # obtain objects generator
        gen = self._graph.objects()

        # make literal list
        lit_list = []
        for i in gen:
            lang = ""
            if type(i) is rb.term.Literal:
                if i.language != "":
                    lang = "@"+i.language
                lit_list.append(str(i)+lang)

        dic = {
                "literals": lit_list
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return simplejson.dumps(dic)



    def get_uri(self, name, lang="", any_one=True, name_predicate="http://rdf.freebase.com/ns/type.object.name", format="python"):
        """
        Get the subject uri of the given name(literal) in language 'lang'(default is no lang)
        Default name_predicate is http://rdf.freebase.com/ns/type.object.name
        
        Return one uri if any_one is True, returns a list of uris otherwise; in python or json format as desired.
        """

        t_name = Technopedia._termify_object(name, lang)
        t_name_predicate = Technopedia._termify_predicate(name_predicate)

        # obtain the generator of the subject uris
        gen = self._graph.subjects(predicate=t_name_predicate, object=t_name)

        # only keep the URI subjects
        uri_subj = [str(i) for i in gen if type(i) is rb.term.URIRef]

        # decide to return one or more uris
        if any_one:
            response = set(uri_subj).pop()
        else:
            response = uri_subj

        #return in expected format
        if format == "python":
            return response
        elif format == "json":
            return simplejson.dumps(response)



    def get_name(self, uri, lang="", name_predicate="http://rdf.freebase.com/ns/type.object.name", format="python"):
        """
        Get the name of the given uri(subject).
        Default name_predicate is http://rdf.freebase.com/ns/type.object.name

        Return name in given language(lang) - en is default; in python or json format as specified.
        """

        t_uri = Technopedia._termify_subject(uri)
        t_name_predicate = Technopedia._termify_predicate(name_predicate)

        # obtain the generator of the object uris
        gen = self._graph.objects(subject=t_uri, predicate=t_name_predicate)

        # only keep the literal objects
        name_list = [str(i) for i in gen if type(i) is rb.term.Literal]

        response = name_list.pop()

        #return in expected format
        if format == "python":
            return response
        elif format == "json":
            return simplejson.dumps(response)



    @staticmethod
    def _termify_subject(subject):
        """
        internal function which returns an appropriate rdflib term for a given string subject

        """
        #check if subject is URI or Blank Node
        try:
            iri.parse(subject, rule="URI")  # returns error if subject is not URI => subject is Blank node
            t_subject = rb.term.URIRef(subject)
        except:
            t_subject = rb.term.BNode(subject)

        return t_subject



    @staticmethod
    def _termify_object(object_, lang=""):
        """
        internal function which returns an appropriate rdflib term for a given string object

        """
        #check if object is URI or Literal Node
        try:
            iri.parse(object_, rule="URI")  # returns error if object is not URI => object is Literal node
            t_object = rb.term.URIRef(object_)
        except:
            t_object = rb.term.Literal(object_, lang=lang)

        return t_object



    @staticmethod
    def _termify_predicate(predicate):
        """
        internal function which returns an appropriate rdflib term for a given string predicate

        """
        return rb.term.URIRef(predicate)



    @staticmethod
    def _termify_context(context):
        """
        internal function which returns an appropriate rdflib term for a given string context

        """
        return rb.term.URIRef(context)

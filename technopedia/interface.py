import simplejson
import rfc3987 as iri

import rdflib as rb


class Technopedia:
    """
    A interface to access the technopedia dataset which is loaded on a mysql database.
    Provides the results in python format for use by other python applications.
    Also provides the results in JSON format

    """

    def __init__(self, id =None, conn_str=None):
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



    def triples(self, format="python", subject=None, predicate=None, object_=None, context=None):
        """
        List of triples for given(any or all) subject, object, predicate and/or context.
        The default value is None in which case all the triples are returned.

        Function returns a python dictionary when format="python" and JSON object when format="json"

        The return value is of the form:
            { "subject":"subject_value",
              "object":"object_value",
              "predicate":"predicate_value",
              "context":"context_value",
              "triples":[list of triples in str format]
            }
        """

        ##convert the arguments to suitable rdflib terms
        t_subject = Technopedia._termify_subject(subject)
        t_predicate = Technopedia._termify_predicate(predicate)
        t_object = Technopedia._termify_object(object_)
        t_context = Technopedia._termify_context(context)


        #obtain a triples generator
        gen = self._graph.triples((t_subject, t_predicate, t_object), t_context)

        #create a response object
        dic = {
                "subject":subject,
                "object":object_,
                "predicate":predicate,
                "context":context,
                "triples":[ str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + " ." for i in gen ]
              }

        #return in expected format
        if format == "python":
            return dic
        elif format == "json":
            return simplejson.dumps(dic)


            
    def get_subjects():
        pass


    
    def get_predicates():
        pass


    def get_objects():
        pass



    def get_contexts():
        pass



    def get_literals():
        pass



    def get_uri():
        pass



    def get_name():
        pass



    @staticmethod
    def _termify_subject(subject):
        """
        internal function which returns an appropriate rdflib term for a given string subject

        """
        #check if subject is URI or Blank Node
        try:
            iri.parse(subject, rule="URI") #returns error if subject is not URI => subject is Blank node
            t_subject = rb.term.URIRef(subject)
        except:
            t_subject = rb.term.BNode(subject)

        return t_subject


    
    @staticmethod
    def _termify_object(object_):
        """
        internal function which returns an appropriate rdflib term for a given string object

        """
        #check if object is URI or Literal Node
        try:
            iri.parse(object_, rule="URI") #returns error if object is not URI => object is Literal node
            t_object = rb.term.URIRef(object_)
        except:
            t_object = rb.term.Literal(object_)

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




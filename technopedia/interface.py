import simplejson

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



	def get_triples():
		pass



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
	








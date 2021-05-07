'''
This file is for bioconcept annotation Pubtator retrieval (official file)
This file takes dima_159_figures_summary.xlsx as input txt file.
'''

import requests
import numpy as np
import os
import pandas as pd
import sys
from xml.etree import ElementTree as ET

def SubmitPMCIDList(pmcid_list, Format, Bioconcept, output_pmc_folder_path):
	'''
	:param pmcid_list: list, pmcid used to retrieve BioC xml file
	:param Format: str, pubtator (PubTator), biocxml (BioC-XML), and biocjson (JSON-XML)
	:param Bioconcept: str, gene, disease, chemical, species, proteinmutation, dnamutation, snp, and cellline. Default includes all
	:param output_pmc_folder_path: str, xml file save folder path
	:return: none, save BioC xml results as xml file
	'''
	json = {}
	json = {"pmcids": pmcid_list}

	#
	# load bioconcepts
	#
	if Bioconcept != "":
		json["concepts"] = Bioconcept.split(",")

	#
	# request
	#
	r = requests.post("https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/" + Format, json=json)
	if r.status_code != 200:
		print("[Error]: HTTP code " + str(r.status_code))
	else:
		tree = ET.XML(r.text.encode("utf-8"))
		for document in tree.findall('document'):
			with open(output_pmc_folder_path + 'PMC' + document.find('id').text + '.xml', "wb") as f:
				collection = ET.Element("collection")
				collection.insert(0, document)
				f.write(ET.tostring(collection))



if __name__ == "__main__":

	# arg_count=0
	# for arg in sys.argv:
	# 	arg_count+=1
	# if arg_count<2 or (sys.argv[2]!= "pubtator" and sys.argv[2]!= "biocxml" and sys.argv[2]!= "biocjson"):
	# 	print("\npython SubmitPMCIDList.py [InputFile] [Format] [BioConcept]\n\n")
	# 	print("\t[Inputfile]: a file with a pmcid list\n")
	# 	print("\t[Format]: pubtator (PubTator), biocxml (BioC-XML), and biocjson (JSON-XML)\n")
	# 	print("\t[Bioconcept]: gene, disease, chemical, species, proteinmutation, dnamutation, snp, and cellline. Default includes all.\n")
	# 	print("\t* All input are case sensitive.\n\n")
	# 	print("Eg., python SubmitPMCIDList.py examples/ex.pmcid pubtator gene\n\n")
	# else:
	# 	Inputfile = sys.argv[1]
	# 	Format = sys.argv[2]
	# 	Bioconcept=""
	# 	if arg_count>=4:
	# 		Bioconcept = sys.argv[3]

	df = pd.read_csv('./validation_set_meta.csv', usecols=['pmcid'])
	# df = df.fillna('')
	Format = 'biocxml'
	Bioconcept = 'gene'
	output_folder = './pmcid_pubtator_gene_annotation_retrieval/'
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	for index, group in df.groupby(np.arange(len(df))//30):
		SubmitPMCIDList(group['pmcid'].to_list(), Format, Bioconcept, output_folder)

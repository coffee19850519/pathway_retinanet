[Directory]
A. Summary of folders & files
B. Retrieving PubTator annotations
C. Processing raw text online
D. Using the PubTator API efficiently

======================================================================================

A. [Summary of folders & files]
	
	virtual environments:
		[venv]
	Input & output:
		[input]
		[output]
		[tmp]
		[input_pmid]
	Resource codes:
		SubmitPMIDList.py
		SubmitText_request.py
		SubmitText_retrieve.py
	
B. [Retrieving PubTator annotations]

	The code to retreive pubtator annotations requires three arguments 1) the name of the file containing the list of pmids, 2) the output file format, 3) the specific concept to retrieve (optional).
	
		$ python SubmitPMIDList.py [InputFile] [Format] [BioConcept]
	
		[Inputfile]: a file with a pmid list
		[Format]: 1) pubtator (PubTator)
				  2) biocxml (BioC-XML)
				  3) biocjson (JSON-XML)
				  * Reference for format descriptions: https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/format/
		[Bioconcept]: Default (leave it blank) includes all bioconcepts. Otherwise, user can choose gene, disease, chemical, species, proteinmutation, dnamutation, snp, and cellline.
		
		* All arguments are case sensitive.
	
		Eg., python SubmitPMIDList.py input_pmid/ex.pmid pubtator

C. [Processing raw text online] 
	
	To set up the environment:
	
		$ virtualenv -p python3.6 venv
		$ source venv/bin/activate
		$ pip install -r requirements.txt

	The process consists of two primary steps 1) submitting requests and 2) retrieving results.
	
	1) Submitting requests : Three parameters are required, which include the name of the target folder containing files to process, the specific concept to retreive, and the output file to save the session numbers for later retrieval. Note that each session number represents the submission of one file.
	
		$ python SubmitText_request.py [Inputfolder] [Bioconcept:Mutation|Chemical|Disease|Gene|All] [outputfile_SessionNumber]
        
		[Inputfolder]: a folder with files to submit
        [Bioconcept]: Gene, Disease, Chemical, Species, Mutation and All.
        [outputfile_SessionNumber]: output file to save the session numbers.

		Eg., python SubmitText_request.py input All SessionNumber.txt
	
	2) Retrieving results : Three parameters are required, which includes the folder of input files, the filename containing session numbers, and the folder to store results. 
	
		$ python SubmitText_retrieve.py [InputFolder] [Inputfile_SessionNumber] [outputfolder]
        
		[InputFolder]: Input folder
		[Inputfile_SessionNumber]: a file with a list of session numbers
        [outputfolder]: Output folder

		Eg., python SubmitText_retrieve.py SessionNumber.txt output

D. [Using the PubTator API efficiently]

	Each file in the input folder will be submitted for processing separately. After submission, each file may be queued for 10 to 20 minutes, depending on the computer cluster workload. Files then wait several additional minutes loading the trained models before processing can begin. System throughput is therefore significantly reduced if each file only contains a small amount of text. To improve efficiency, we suggest that each file contain roughly 100 abstracts or 5 full-text articles (100,000-200,000 characters). Note that some files may complete earlier than others; the estimated time to complete (ETC) is an estimate of the processing time for all files.
		

References
---------------------------------------------------------------------------
[1] Wei, C.H., Allot, A., Leaman, R., & Lu, Z. (2019). PubTator central: automated concept annotation for biomedical full text articles. Nucleic acids research, 47(W1), W587-W593. doi: 10.1093/nar/gkz389
[2] Comeau DC et al. (2019) PMC text mining subset in BioC: about three million full-text articles and growing, 2019, Bioinformatics, 10.1093/bioinformatics/btz070
[3] Wei, C.H., Leaman, R., and Lu, Z. (2016) Beyond accuracy: Creating interoperable and scalable text mining web services. Bioinformatics, 2016. 32(12): p. 1907-1910.

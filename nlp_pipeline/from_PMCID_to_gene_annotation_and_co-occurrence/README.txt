This folder aims to use PMCID to retrieve gene annotations and full text from Pubtator, and get the gene co-occurrence.

Step1: SubmitPMCIDList.py.py
       Use PMCID in dima_159_figures_summary.xlsx to retrieve all BioC xml files from Pubtator, which contain gene annotations and full text, if applicable. 
       BioC xml results are stored in folder named pmcid_pubtator_gene_annotation_retrieval.
       
Step 2: gene_co-occurrence_from_pubtator_results.py
        Use BioC xml resuls to get gene annotation and full text, then count gene co-occurrence for each article.
        Full text files are stored in txt format under folder named pmcid_full_text_txt_files.
        Gene co-occurrence results are stored in csv format under folder named gene_co_occurrence

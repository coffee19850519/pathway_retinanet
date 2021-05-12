This folder aims to use pdf file to retrieve gene annotations from Pubtator, and get the gene co-occurrence.

Step 1: extract_text_from_pdf_then_xml_whole_article.py

Step 2: SubmitText_request.py

Step 3: SubmitText_retrieve.py
              
Step 4: gene_co-occurrence_from_pubtator_results.py
        Use BioC xml resuls to get gene annotation and full text, then count gene co-occurrence for each article.
        Full text files are stored in txt format under folder named pmcid_full_text_txt_files.
        Gene co-occurrence results are stored in csv format under folder named gene_co_occurrence

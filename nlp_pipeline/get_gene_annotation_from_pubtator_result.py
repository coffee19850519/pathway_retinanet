'''
This file is to extract gene annotations from Pubtator retrieved BioC xml file. The final result for each xml file is
a list of gene annotations.
'''

import xml.etree.ElementTree as ET

def extract_gene_annotation(input_file_path):
    '''
    :param input_file_path: str, xml file retrived from Pubtator.
    :return: list, contains all gene annotations of an article
    '''

    tree = ET.parse(input_file_path)
    root = tree.getroot()

    gene_annotation = []
    for document in root.findall('document'):
        gene = set()

        for passage in document.iter('passage'):
            if passage.find('annotation') == None:
                continue
            for annotation in passage.findall('annotation'):
                gene.add(annotation.find('text').text)

        gene_annotation.append(', '.join(gene))

    return gene_annotation



if __name__ == "__main__":
    source = './pmid_sample_2.xml'
    gene_list = extract_gene_annotation(source)
    print(gene_list)

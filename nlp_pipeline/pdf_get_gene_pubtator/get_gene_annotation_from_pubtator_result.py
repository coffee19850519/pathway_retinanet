'''
This file is to extract gene annotations of an article from Pubtator retrieved BioC xml files. The final result for each
article is a list of gene annotations.
'''

import xml.etree.ElementTree as ET
import glob

def extract_gene_annotation(input_file_folder):
    '''
    :param input_file_folder: str, folder path of xml files retrived from Pubtator.
    :return: list, contains all gene annotations of an article
    '''

    gene = set()
    for xml_file in glob.glob(input_file_folder + "*.xml"):

        tree = ET.parse(xml_file)
        root = tree.getroot()

        for document in root.findall('document'):
            for passage in document.iter('passage'):
                if passage.find('annotation') == None:
                    continue
                for annotation in passage.findall('annotation'):
                    if annotation.find('.//infon[@key="type"]').text == 'Gene':
                        gene.add(annotation.find('text').text)

    return list(gene)

if __name__ == "__main__":
    source = './xml_file_folder/'
    gene_list = extract_gene_annotation(source)
    print(gene_list)

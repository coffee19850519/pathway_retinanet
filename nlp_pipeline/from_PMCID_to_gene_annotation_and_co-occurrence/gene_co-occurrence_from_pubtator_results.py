'''
This file is to utilize the BioC xml file which is retrieved from Pubtator by PMCID and extract gene annotations & full
text, then get gene co-occurrence
* step 1: extract both gene annotations and full text from Pubtator-retrieved BioC xml file
* step 2: count gene co-occurrence for each gene annotation pair
'''

import pandas as pd
import numpy as np
import glob
import os
import xml.etree.ElementTree as ET
import nltk
from collections import OrderedDict

def extract_gene_annotation_and_full_text(input_file_path, output_txt_file_folder):
    '''
    :param input_file_path: str, BioC xml file retrieved from Pubtator thru PMCID.
    :param output_txt_file_folder: str, a folder path to store txt file extracted from BioC xml file
    :return: gene_list: list, contains all gene annotations in the BioC xml file
             full_text: str, full text of the PMCID article
             pmcid: str, pmcid
             none, save full_text to a txt file
    '''

    tree = ET.parse(input_file_path)
    root = tree.getroot()

    full_text = ''
    gene_annotation = []
    pmcid = ''

    if len(root.findall('document')) == 0:
        print('This BioC xml file has no full text: ' + input_file_path)
        return gene_annotation, full_text, pmcid


    for document in root.findall('document'):
        gene = set()

        for passage in document.iter('passage'):

            if passage.find('text') == None:
                continue

            if passage.find('text').text[-1] != '.':
                full_text = full_text + ' ' + passage.find('text').text + '.'
            else:
                full_text = full_text + ' ' + passage.find('text').text

            if passage.find('annotation') == None:
                continue
            for annotation in passage.findall('annotation'):
                gene.add(annotation.find('text').text)

        gene_annotation = gene_annotation + list(gene)

        pmcid = pmcid + 'PMC' + document.find('id').text
        with open(output_txt_file_folder + pmcid + ".txt", "w") as text_file:
            text_file.write(full_text)

    return gene_annotation, full_text, pmcid


def gene_co_occurrence_in_sentence(sentence_list, gene_list, csv_folder_path, pmcid):
    '''
    :param sentence_list: list, a list of sentences
    :param gene_list: list, a list of gene names
    :param csv_folder_path: str, a folder path to store gene co-occurrence csv file
    :param pmcid: str, pmcid, refer to the output gene co-occurrence csv file name
    :return: none, a csv file contains gene co-occurrence information
    '''

    occurrences = OrderedDict((name, OrderedDict((name, 0) for name in gene_list)) for name in gene_list)

    # Find the co-occurrences:
    gene_set = set(gene_list)
    for sent in sentence_list:
        intersection = gene_set.intersection(set(sent))
        if len(intersection) > 1:
            name_list = list(intersection)
            for i in range(len(name_list)):
                for item in name_list[:i] + name_list[i + 1:]:
                    occurrences[name_list[i]][item] += 1

    # print(' ', ' '.join(occurrences.keys()))
    gene_name_1 = []
    gene_name_2 = []
    gene_co_occurrence = []
    for name_1, values in occurrences.items():
        for name_2, co_occurrence in values.items():
            if (co_occurrence != 0) and not (name_2 in gene_name_1 and name_1 in gene_name_2):
                gene_name_1.append(name_1)
                gene_name_2.append(name_2)
                gene_co_occurrence.append(str(co_occurrence))

    gene_name_1 = [name.replace('_', ' ') for name in gene_name_1]
    gene_name_2 = [name.replace('_', ' ') for name in gene_name_2]

    dict = {'gene_name_1': gene_name_1,
            'gene_name_2': gene_name_2,
            'co_occurrence': gene_co_occurrence}

    df = pd.DataFrame(dict)
    df.to_csv(csv_folder_path + pmcid + '.csv', index=False)


# use underline '_' to replace space ' ' for each gene, to make each gene name consisting of 'one word', then tokenize
# each sentence in full text
def preprocess_sent_list_and_gene_list(full_text, gene_list):
    '''
    :param full_text: str, full text of a PMCID article
    :param gene_list: list, a list of gene names
    :return: 2 lists, processed_sent_list -> nested list, with each sentence tokenized to words
                      processed gene_list -> list, use underline '_' to replace space ' ' for each gene
    '''

    processed_gene_list = []
    for gene in gene_list:
        if ' ' in gene:
            gene_underline = gene.replace(' ', '_')
            processed_gene_list.append(gene_underline)
            full_text = full_text.replace(gene, gene_underline)
        else:
            processed_gene_list.append(gene)

    sent_list = nltk.sent_tokenize(full_text)
    processed_sent_list = []
    for sent in sent_list:
        processed_sent_list.append(nltk.word_tokenize(sent))

    return processed_sent_list, processed_gene_list


if __name__ == "__main__":

    input_xml_file_from_pubtator_folder = './pmcid_pubtator_gene_annotation_retrieval/'
    output_txt_file_folder = './pmcid_full_text_txt_files/'
    csv_folder_path = './gene_co_occurrence/'

    for xml_file in glob.glob(input_xml_file_from_pubtator_folder + "*.xml"):
        if not os.path.isdir(output_txt_file_folder):
            os.mkdir(output_txt_file_folder)
        gene_annotation, full_text, pmcid = extract_gene_annotation_and_full_text(xml_file, output_txt_file_folder)
        if full_text == '':
            continue
        processed_sent_list, processed_gene_list = preprocess_sent_list_and_gene_list(full_text, gene_annotation)
        if not os.path.isdir(csv_folder_path):
            os.mkdir(csv_folder_path)
        gene_co_occurrence_in_sentence(processed_sent_list, processed_gene_list, csv_folder_path, pmcid)

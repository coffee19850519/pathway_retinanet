'''
This file is to utilize the abstracts/full_texts retrieved from Pubtator PMID/PMCID and extract "gene" relations
* step 1: extract both gene annotations and abstracts/full_ texts from Pubtator-retrieved XML file -> the process functions of PMID & PMCID XML file is different
* step 2: use OpenIE to extract general relations from abstracts/full_texts -> need to convert each abstract/full_text to txt file, for OpenIE
* step 3: use the gene annotations to filter the general relations with gene included --> get the final csv file result per abstract/full_text -- gene relations
'''

import pandas as pd
import numpy as np
import glob
import re
import os
import subprocess
from subprocess import *
import xml.etree.ElementTree as ET
import nltk

def extract_gene_annotation_and_abstract_text(input_file_path, output_file_folder, output_file_name):

    '''
    :param input_file_path: str, xml file retrived from Pubtator. PMID input file path
    :param output_file_folder: str, output file folder path
    :param output_file_name: str, output file name
    :return: none, save as a csv file with PMID, title, abstract, and gene annotation
    '''

    # parse input xml file
    tree = ET.parse(input_file_path)
    root = tree.getroot()

    pmid = []
    title = []
    abstract = []
    gene_annotation = []
    for document in root.findall('document'):
        # print(document.find('id').text)
        pmid.append(document.find('id').text)
        gene = set()
        for passage in document.iter('passage'):
            # print(passage.find('.//infon[@key="type"]').text)
            # print(passage.find('text').text)
            if passage.find('.//infon[@key="type"]').text == 'title':
                title.append(passage.find('text').text)
            if passage.find('.//infon[@key="type"]').text == 'abstract':
                if passage.find('text') != None:
                    abstract.append(passage.find('text').text)
                else:
                    abstract.append('')

            for annotation in passage.findall('annotation'):
                gene.add(annotation.find('text').text)

        gene_annotation.append(', '.join(gene))

    dict = {'PMID': pmid,
            'title': title,
            'abstract': abstract,
            'gene_annotation': gene_annotation}

    df = pd.DataFrame(dict)
    df.to_csv(output_file_folder + output_file_name, index=False)



def extract_gene_annotation_and_full_text(input_file_path, output_file_folder, output_file_name):

    '''
    :param input_file_path: str, xml file retrived from Pubtator. PMCID input file path
    :param output_file_folder: str, output file folder path
    :param output_file_name: str, output file name
    :return: none, save as a csv file with PMID, PMCID, title, full text, and gene annotation
    '''

    tree = ET.parse(input_file_path)
    root = tree.getroot()

    pmcid = []
    pmid = []
    title = []
    full_text = []
    gene_annotation = []
    for document in root.findall('document'):
        # print(document)
        # print(document.find('id').text)
        pmcid.append('PMC' + document.find('id').text)
        pmid.append(document.find('passage').find('.//infon[@key="article-id_pmid"]').text)
        full_text_string = ""
        gene = set()

        for passage in document.iter('passage'):
            # print(passage.find('.//infon[@key="section_type"]').text)
            # print(passage.find('text').text)
            if passage.find('.//infon[@key="section_type"]').text == 'TITLE':
                title.append(passage.find('text').text)

            if passage.find('text').text[-1] != '.':
                full_text_string = full_text_string + ' ' + passage.find('text').text + '.'
            else:
                full_text_string = full_text_string + ' ' + passage.find('text').text

            if passage.find('annotation') == None:
                continue

            for annotation in passage.findall('annotation'):
                gene.add(annotation.find('text').text)


        full_text.append(full_text_string)
        gene_annotation.append(', '.join(gene))

    print(len(full_text))

    dict = {'PMID': pmid,
            'PMCID': pmcid,
            'title': title,
            'full_text': full_text,
            'gene_annotation': gene_annotation}

    df = pd.DataFrame(dict)
    df.to_csv(output_file_folder + output_file_name, index=False)



def convert_csv_text_to_txt_multiple_files(input_file_path, output_file_folder):
    '''
    :param input_file_path:
    :param output_file_folder:
    :return:
    '''
    df = pd.read_csv(input_file_path)

    if 'abstract' in df.columns.values.tolist():
        for index, row in df.iterrows():
            with open(output_file_folder + str(row['PMID']) + ".txt", "w") as text_file:
                text_file.write(str(row['abstract']).strip())

    elif 'full_text' in df.columns.values.tolist():
        for index, row in df.iterrows():
            sent_text = nltk.sent_tokenize(str(row['full_text']))
            sent_chunks = [sent_text[x:x+10] for x in range(0, len(sent_text), 10)]
            directory = os.path.join(output_file_folder, str(row['PMCID']))
            if not os.path.exists(directory):
                os.mkdir(directory)
            for i in range(len(sent_chunks)):
                with open(directory + '/' + str(i) + ".txt", "w") as text_file:
                    text_file.write(' '.join(sent_chunks[i]))

def extract_gene_relations(input_txt_file_folder, input_csv_file_path, output_csv_file_folder):
    '''
    :param input_txt_file_folder:
    :param input_csv_file_path:
    :param output_csv_file_folder:
    :return:
    '''

    df = pd.read_csv(input_csv_file_path)
    df = df.set_index('PMID')
    df.fillna('', inplace=True)

    for file in glob.glob(input_txt_file_folder + "*.txt"):

        start = file.rfind('/')
        end = file.find('.txt')
        pmid = file[start+1:end]

        # pmid = re.search('/(.+?).txt', file).group(1)
        print(pmid)
        if df['gene_annotation'][int(pmid)] != '':
            confidence_score = []
            subject = []
            relation_entity = []
            object = []

            process = Popen(['java', '-mx1g', '-cp', '*', 'edu.stanford.nlp.naturalli.OpenIE', file], stdout=PIPE,
                            stderr=PIPE)
            stdout, stderr = process.communicate()
            for gene in list(df['gene_annotation'][int(pmid)].split(',')):
                for relation in list(stdout.decode("utf-8").split('\n')):
                    if gene in relation:
                        relation_list = list(relation.split('\t'))
                        confidence_score.append(relation_list[0])
                        subject.append(relation_list[1])
                        relation_entity.append(relation_list[2])
                        object.append(relation_list[3])

        if len(confidence_score) != 0:
            dict = {'confidence_score': confidence_score,
                    'subject': subject,
                    'relation': relation_entity,
                    'object': object}
            df_output = pd.DataFrame(dict)
            df_output.to_csv(output_csv_file_folder + pmid + '.csv', index=False)


def extract_gene_relations_pmc(input_txt_file_folder, input_csv_file_path, output_csv_file_folder):
    '''
    :param input_txt_file_folder:
    :param input_csv_file_path:
    :param output_csv_file_folder:
    :return:
    '''

    df = pd.read_csv(input_csv_file_path)
    # df = df.set_index('PMCID')
    df.fillna('', inplace=True)

    for directory in next(os.walk(input_txt_file_folder))[1]:
        print(directory)
        confidence_score = []
        subject = []
        relation_entity = []
        object = []

        for file in glob.glob(input_txt_file_folder + directory + "/" + "*.txt"):
            # start = file.rfind('/')
            # end = file.find('.txt')
            # pmcid = file[start+1:end]
            pmcid = directory
            # pmid = re.search('/(.+?).txt', file).group(1)
            # print(pmcid)
            # df['gene_annotation'][int(pmid)]
            if df.loc[df['PMCID'] == pmcid]['gene_annotation'].values[0] != '':

                process = Popen(['java', '-mx1g', '-cp', '*', 'edu.stanford.nlp.naturalli.OpenIE', file], stdout=PIPE,
                                stderr=PIPE)
                stdout, stderr = process.communicate()
                for gene in list(df.loc[df['PMCID'] == pmcid]['gene_annotation'].values[0].split(',')):
                    for relation in list(stdout.decode("utf-8").split('\n')):
                        if gene in relation:
                            relation_list = list(relation.split('\t'))
                            confidence_score.append(relation_list[0])
                            subject.append(relation_list[1])
                            relation_entity.append(relation_list[2])
                            object.append(relation_list[3])

        if len(confidence_score) != 0:
            dict = {'confidence_score': confidence_score,
                    'subject': subject,
                    'relation': relation_entity,
                    'object': object}
            df_output = pd.DataFrame(dict)
            df_output.to_csv(output_csv_file_folder + directory + '.csv', index=False)


if __name__ == "__main__":

    # input_pmid_w_o_pmcid_pubtator_retrieval_csv = '../../data/non_small_cell_lung_cancer/pmid_w_o_pmcid_pubtator_retrieval.xml'
    # input_pmcid_pubtator_retrieval_csv = '../../data/non_small_cell_lung_cancer/pmcid_pubtator_retrieval.xml'
    gene_text_csv_output_folder = '../../data/non_small_cell_lung_cancer/output_csv_files_gene_text/'

    gene_text_csv_output_pmid_w_o_pmcid_file_name = 'pmid_w_o_pmcid_gene_abstract_273_total.csv'
    # gene_text_csv_output_pmcid_file_name = 'pmcid_gene_full_text_117_total.csv'

    # txt_file_folder_pmcid = '../../data/non_small_cell_lung_cancer/txt_files_for_openie/pmcid/'
    txt_file_folder_pmid_w_o_pmicd = '../../data/non_small_cell_lung_cancer/txt_files_for_openie/pmid_w_o_pmcid/'
    # output_relation_csv_file_folder_pmid_all = '../../data/non_small_cell_lung_cancer/output_relations_csv_files/pmid_all/'
    output_relation_csv_file_folder_pmid_w_o_pmcid = '../../data/non_small_cell_lung_cancer/output_relations_csv_files/pmid_w_o_pmcid/'
    # output_relation_csv_file_folder_pmcid = '../../data/non_small_cell_lung_cancer/output_relations_csv_files/pmcid/'

    # extract_gene_annotation_and_abstract_text(input_pmid_w_o_pmcid_pubtator_retrieval_csv, gene_text_csv_output_folder, gene_text_csv_output_pmid_w_o_pmcid_file_name)
    # extract_gene_annotation_and_full_text(input_pmcid_pubtator_retrieval_csv, gene_text_csv_output_folder, gene_text_csv_output_pmcid_file_name)
    # convert_csv_text_to_txt_multiple_files(gene_text_csv_output_folder+gene_text_csv_output_pmid_w_o_pmcid_file_name, txt_file_folder_pmid_w_o_pmicd)
    # convert_csv_text_to_txt_multiple_files(gene_text_csv_output_folder+gene_text_csv_output_pmcid_file_name, txt_file_folder_pmcid)
    extract_gene_relations(txt_file_folder_pmid_w_o_pmicd, gene_text_csv_output_folder+gene_text_csv_output_pmid_w_o_pmcid_file_name, output_relation_csv_file_folder_pmid_w_o_pmcid)
    # extract_gene_relations_pmc(txt_file_folder_pmcid, gene_text_csv_output_folder+gene_text_csv_output_pmcid_file_name, output_relation_csv_file_folder_pmcid)

'''
This file is to convert pdf file to txt file, then format the converted txt file to BioC xml file. BioC xml file is
one of the formats which can be submitted to Pubtator tool. Other possible formats can be Json or Pubtator format.

This file can process a bunch of pdf files at the same time, just make sure all the pdf files are under the same folder.
'''

import textract
import glob
import os
import sys
import xml.etree.ElementTree as ET
import re
import shutil


def convert_pdf_to_text_file(source, destination_folder):
    '''
    :param source: str, source pdf file path
    :param destination_folder: str, txt file destination folder path
    :return: none, write txt file to destination_folder directly
    '''

    paperInText = textract.process(source).decode()
    pdfName = source[:-4]
    filename = destination_folder + os.path.basename(pdfName) + '.txt'
    with open(filename, 'w') as f:
        f.write(paperInText)


def convert_txt_to_BioC_xml_file(source, destination_folder):
    '''
    :param source: str, source txt file path
    :param destination_folder: str, BioC xml file destination folder path
    :return: none, write BioC xml file to destination_folder directly
    '''
    with open(source, encoding='utf8') as lines_file:
        lines = lines_file.read()

    sents = lines.split('\n\n')
    sep_sents = []
    for string in sents:
        string = string.replace('\n', ' ')
        sep_sents.append(re.sub('\W+',' ', string))

    grouped_sents = []
    sep_sents_len = len(sep_sents)
    
    # you can make n larger if there is no response from Pubtator
    n = 10
    for i in range(n):
        start = int(i * sep_sents_len / n)
        end = int((i + 1) * sep_sents_len / n)
        grouped_sents.append('. '.join(sep_sents[start:end]))

    offset_count = 0
    txtName = source[:-4]
    xml_folder_name = os.path.basename(txtName)

    shutil.rmtree(destination_folder + xml_folder_name, ignore_errors=True)
    if not os.path.isdir(destination_folder + xml_folder_name):
        os.mkdir(destination_folder + xml_folder_name)

    xml_file_counter = 0
    for sent in grouped_sents:
        xml_file_counter += 1
        collection = ET.Element('collection')
        document = ET.SubElement(collection, 'document')
        passage = ET.SubElement(document, 'passage')
        offset = ET.SubElement(passage, 'offset')
        offset.text = str(offset_count)
        text = ET.SubElement(passage, 'text')
        text.text = sent

        tree = ET.ElementTree(collection)

        filename = destination_folder + xml_folder_name + '/' + str(xml_file_counter) + '.xml'
        tree.write(filename, encoding="UTF-8")

if __name__ == "__main__":
    source_pdf_folder = './pdf/'
    txt_destination_folder = './txt/'
    BioC_xml_destination_folder = './xml/'

    # convert pdf file to txt file
    for pdf_file in glob.glob(source_pdf_folder + "*.pdf"):
        if not os.path.isdir(txt_destination_folder):
            os.mkdir(txt_destination_folder)
        convert_pdf_to_text_file(pdf_file, txt_destination_folder)

    # convert txt file to BioC xml file
    for txt_file in glob.glob(txt_destination_folder + "*.txt"):
        if not os.path.isdir(BioC_xml_destination_folder):
            os.mkdir(BioC_xml_destination_folder)
        convert_txt_to_BioC_xml_file(txt_file, BioC_xml_destination_folder)



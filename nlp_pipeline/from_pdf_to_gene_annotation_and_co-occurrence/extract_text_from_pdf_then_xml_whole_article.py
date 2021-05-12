'''
This file is to convert pdf file (store all pdf files in ./pdf path) to txt file, then format the converted txt
file to BioC xml file. BioC xml file is one of the formats which can be submitted to Pubtator tool. Other possible
formats can be Json or Pubtator format.
This file can process a bunch of pdf files at the same time.
'''

import textract
import glob
import os
import sys
import xml.etree.ElementTree as ET
import re


def convert_pdf_as_text_file(source, destination_folder):
    '''
    :param source: str, source pdf file path
    :param destination_folder: str, txt file destination folder path
    :return: none, write txt file to destination_folder directly
    '''
    print(source)
    paperInText = textract.process(source, encoding='ascii', errors='ignore').decode()
    pdfName = source[:-4]
    filename = destination_folder + os.path.basename(pdfName) + '.txt'
    with open(filename, 'w') as f:
        f.write(paperInText)


def convert_txt_as_BioC_xml_file(source, destination_folder):
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
        sep_sents.append(re.sub('\W+', ' ', string))


    collection = ET.Element('collection')
    document = ET.SubElement(collection, 'document')

    offset_count = 0
    for sent in sep_sents:
        passage = ET.SubElement(document, 'passage')
        offset = ET.SubElement(passage, 'offset')
        offset.text = str(offset_count)
        # offset_count += 1
        text = ET.SubElement(passage, 'text')
        text.text = sent

    tree = ET.ElementTree(collection)
    txtName = source[:-4]
    filename = destination_folder + os.path.basename(txtName) + '.xml'
    tree.write(filename, encoding="UTF-8")

if __name__ == "__main__":
    source_pdf_folder = './pdf/'
    txt_destination_folder = './txt/'
    BioC_xml_destination_folder = './xml/'

    # convert pdf file to txt file
    for pdf_file in glob.glob(source_pdf_folder + "*.pdf"):
        if not os.path.isdir(txt_destination_folder):
            os.mkdir(txt_destination_folder)
        convert_pdf_as_text_file(pdf_file, txt_destination_folder)

    # convert txt file to BioC xml file
    for txt_file in glob.glob(txt_destination_folder + "*.txt"):
        if not os.path.isdir(BioC_xml_destination_folder):
            os.mkdir(BioC_xml_destination_folder)
        convert_txt_as_BioC_xml_file(txt_file, BioC_xml_destination_folder)

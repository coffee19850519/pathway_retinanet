import os, shutil
from pathlib import Path
from pathway_identifier import image_identifier
from get_text_and_figures_new import extract_information
import cfg
from pipeline_hugo import run_model
import argparse
import os
import warnings

warnings.filterwarnings('ignore')


def get_images(pdf_path, img_path, identifier_model, img_identifier_output_path):
    '''
    Extract the required pictures from PDF
    :param pdf_path: Pdf path
    :param img_path: Extracted image path
    :param identifier_model: Determine whether it is a model of the pathway picture
    :param img_identifier_output_path: Path where the pathway image is stored
    :return: Folder where pathway pictures are stored
    '''
    pdf_file_path = Path(pdf_path)
    for pdf_file in pdf_file_path.glob("*.pdf"):
        pdf_name = os.path.split(pdf_file)[1].split('.')[0]
        print('pdf_name:', pdf_name)
        extract_information(pdf_file)
        img_path_2 = os.path.join(img_path, pdf_name)
        if os.path.isdir(img_path_2):
            image_identifier(img_path_2, None, identifier_model, img_identifier_output_path)
            print('Have done pathway image recognition')


def get_gene_relation(img_identifier_output_path):
    '''
    Get the gene and relationship in the pathway graph
    :param img_identifier_output_path: Path of the pathway graph
    :return: Get gene and relationship results
    '''
    for dir in os.listdir(img_identifier_output_path):
        if any((file.endswith('.jpg')) for file in os.listdir(os.path.join(img_identifier_output_path, dir, 'img'))):
            parser = argparse.ArgumentParser()
            args = parser.parse_args()
            args.dataset = os.path.join(img_identifier_output_path, dir)
            run_model(cfg, None, **vars(args))
        else:
            shutil.rmtree(os.path.join(img_identifier_output_path, dir))
            if os.path.exists(os.path.join(img_identifier_output_path, dir)):
                os.removedirs(os.path.join(img_identifier_output_path, dir))


if __name__ == '__main__':
    pdf_path = 'paper'
    img_path = 'extract_img'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    identifier_model = 'model/train3/csv_retinanet_4.pt'
    img_identifier_output_path = 'result'
    if not os.path.exists(img_identifier_output_path):
        os.makedirs(img_identifier_output_path)
    get_images(pdf_path, img_path, identifier_model, img_identifier_output_path)
    get_gene_relation(img_identifier_output_path)

import os
import shutil, json
from tools.label_file import LabelFile


def look_for_same_name_files(source_folder, look_for_folder, save_folder):
    for source_file in os.listdir(source_folder):
        target_file = os.path.join(look_for_folder, source_file)
        if os.path.exists(target_file):
            shutil.copyfile(target_file, os.path.join(save_folder, source_file))

def combine_relation_and_gene_annotations(relation_json, gene_json, save_file_name):
    gene_annotations = LabelFile(gene_json)
    relation_annotations = LabelFile(relation_json)

    gene_id_and_names = gene_annotations.get_all_gene_id_and_names()
    relation_annotations = relation_annotations.get_all_relations()
    formatted_relations = []
    for relation in relation_annotations:
        _, relation_type, invovling_genes = relation.split(':')
        startor_id, receptor_id = invovling_genes.split('|')
        startor = gene_id_and_names[startor_id]
        receptor = gene_id_and_names[receptor_id]
        formatted_relations.append((startor, relation_type, receptor))
    with open(save_file_name, 'w') as save_fp:
        json.dump(formatted_relations, save_fp)
    del gene_annotations, relation_annotations, gene_id_and_names, formatted_relations





if __name__ == "__main__":
    # source_folder = r'/home/fei/Desktop/chunhui/jsons_with_relation_only/'
    # look_for_image_folder = r'/home/fei/Desktop/image/'
    # look_for_json_folder = r'/home/fei/Desktop/0130/'
    # save_image_folder = r'/home/fei/Desktop/chunhui/orginal_images/'
    # save_json_folder = r'/home/fei/Desktop/chunhui/jsons_with_complete_annotations/'

    #look_for_same_name_files(source_folder, look_for_image_folder, save_image_folder)
    #look_for_same_name_files(source_folder, look_for_json_folder, save_json_folder)

    completed_relation_json_folder = r'/home/fei/Desktop/chunhui/jsons_with_complete_annotations/'
    partial_relation_json_foler = r'/home/fei/Desktop/chunhui/jsons_with_relation_only/'
    save_folder = r'/home/fei/Desktop/chunhui/formatted_annotations/'
    for json_file in os.listdir(partial_relation_json_foler):
        if 'cin_' not in json_file:
            continue
        #find the file with same name in completed_relation_json_folder
        relation_json = os.path.join(partial_relation_json_foler, json_file)
        gene_name_json = os.path.join(completed_relation_json_folder, json_file)
        save_json = os.path.join(save_folder, json_file)
        combine_relation_and_gene_annotations(relation_json, gene_name_json, save_json)

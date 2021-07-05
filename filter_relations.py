import csv
import pandas as pd
import os
import copy
import json
import numpy as np

df = pd.read_csv("checked_relation.csv")

directory = "filtered_val_images/img_jaccard/relation_subimage"
accuracies = []
recalls = []
precisions = []

gene_df = pd.read_csv("new_gene.csv")
for index, row in gene_df.iterrows():
    gene_df.loc[index,['annotated_gene_name']] = row['annotated_gene_name'].upper().replace(" ","").replace("+","").replace("-","").replace("(","").replace(")","").replace(".","").replace(":","")

count = 0
filtered_df = None
for filename in os.listdir(directory):

    print(filename)

    # get gene dictionary
    with open("all_gene_names.json") as gene_name_list_fp:
        gene_name_list = json.load(gene_name_list_fp)
    gene_name_list = [x.upper() for x in gene_name_list]

    # get annotated relationships
    # filter for string length and replace some special characters
    current_im_relations_annotation = df[(df['fig_name'] == filename)]
    # current_im_relations_annotation = current_im_relations_annotation.loc[current_im_relations_annotation['startor'].str.len() < 7]
    # current_im_relations_annotation = current_im_relations_annotation.loc[current_im_relations_annotation['receptor'].str.len() < 7]

    current_im_relations_annotation['startor_id'] = None
    current_im_relations_annotation['receptor_id'] = None
    for index, row in current_im_relations_annotation.iterrows():

        temp_startor = row['startor'].upper().replace(" ","").replace("+","").replace("-","").replace("(","").replace(")","").replace(".","").replace(":","")
        temp_receptor = row['receptor'].upper().replace(" ","").replace("+","").replace("-","").replace("(","").replace(")","").replace(".","").replace(":","")

        current_im_relations_annotation.loc[index,['startor']] = row['startor'].upper().replace(" ","").replace("+","").replace("-","").replace("(","").replace(")","").replace(".","").replace(":","")
        current_im_relations_annotation.loc[index,['receptor']] = row['receptor'].upper().replace(" ","").replace("+","").replace("-","").replace("(","").replace(")","").replace(".","").replace(":","")

        current_im_relations_annotation.loc[index,['startor_id']] = gene_df[gene_df['annotated_gene_name'] == temp_startor]['gene_id'].values[0]
        current_im_relations_annotation.loc[index,['receptor_id']] = gene_df[gene_df['annotated_gene_name'] == temp_receptor]['gene_id'].values[0]


    filtered_df = pd.concat([filtered_df,current_im_relations_annotation])

    

filtered_df.to_csv('filter_special_char_relations.csv',index=False)
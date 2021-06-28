import csv
import pandas as pd
import os
import copy
import json
import numpy as np

df = pd.read_csv("checked_relation.csv")

directory = "filtered_val_images/img_both/relation_subimage"
accuracies = []
recalls = []
precisions = []

count = 0
for filename in os.listdir(directory):

    # get gene dictionary
    with open("all_gene_names.json") as gene_name_list_fp:
        gene_name_list = json.load(gene_name_list_fp)
    gene_name_list = [x.upper() for x in gene_name_list]

    print(filename)

    # get annotated relationships
    current_im_relations_annotation = df[(df['fig_name'] == filename)]
    current_im_relations_annotation = current_im_relations_annotation.loc[current_im_relations_annotation['startor'].str.len() < 7]
    current_im_relations_annotation = current_im_relations_annotation.loc[current_im_relations_annotation['receptor'].str.len() < 7]
    anno_startors = current_im_relations_annotation['startor'].str.upper().replace(" ","")
    anno_receptors = current_im_relations_annotation['receptor'].str.upper().replace(" ","")
    anno_relations = list(zip(anno_startors, anno_receptors))

    # filter for only relationships with genes
    relations_to_remove = []
    for count,relationship in enumerate(anno_relations):

        startor, receptor = relationship
        if startor in gene_name_list and receptor in gene_name_list:
            continue
        else:
            relations_to_remove.append(count)
    for index in sorted(relations_to_remove, reverse=True):
        del anno_relations[index]

    print("gt")
    print(anno_relations)

    # no only gene relationships, then skip
    if len(anno_relations) == 0:
        continue


    # get predicted relationships
    image_name, ext = os.path.splitext(os.path.basename(filename))
    tmp_df = pd.read_json("filtered_val_images" + "/img_both/" + image_name + "_relation.json").T
    tmp_df = tmp_df.loc[tmp_df['startor'].str.len() < 7]
    tmp_df = tmp_df.loc[tmp_df['receptor'].str.len() < 7]
    pred_startors = tmp_df['startor'].str.upper().replace(" ","")
    pred_receptors = tmp_df['receptor'].str.upper().replace(" ","")
    pred_relations = list(zip(pred_startors, pred_receptors))

    # filter for only relationships with genes
    relations_to_remove = []
    for count,relationship in enumerate(pred_relations):

        startor, receptor = relationship
        if startor in gene_name_list and receptor in gene_name_list:
            continue
        else:
            relations_to_remove.append(count)
    for index in sorted(relations_to_remove, reverse=True):
        del pred_relations[index]

    print("predictions")
    print(pred_relations)




    conf_mat = np.zeros((2,2))
    tmp_relations = copy.deepcopy(anno_relations)
    for current_pred_relation in pred_relations:
        
        if current_pred_relation in tmp_relations:
            # tp
            conf_mat[0][0] += 1

            # remove detected element from pool
            index_to_remove = tmp_relations.index(current_pred_relation)
            del tmp_relations[index_to_remove]

        else:
            # fp
            conf_mat[0][1] += 1

    tmp_relations = copy.deepcopy(pred_relations)
    for anno_relation in anno_relations:

        if anno_relation not in tmp_relations:
            # fn
            conf_mat[1][0] += 1

        else:
            # remove detected element from pool
            index_to_remove = tmp_relations.index(anno_relation)
            del tmp_relations[index_to_remove]

    print(conf_mat)

    # could be lower by images with nothing
    tmp_acc = conf_mat[0][0] / np.sum(np.sum(conf_mat,axis=1))
    tmp_precision = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
    tmp_recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])

    accuracies.append(tmp_acc)
    precisions.append(tmp_precision)
    recalls.append(tmp_recall)

    print(conf_mat)
    print(tmp_acc)
    print(tmp_precision)
    print(tmp_recall)

    # if count == 5:
    #     break
    # count += 1

    # break

accuracies = np.array(accuracies)
accuracies = np.nan_to_num(accuracies)
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
median_acc = np.median(accuracies)

precisions = np.array(precisions)
precisions = np.nan_to_num(precisions)
mean_precisions = np.mean(precisions)
std_precisions = np.std(precisions)
median_precisions = np.median(precisions)

recalls = np.array(recalls)
recalls = np.nan_to_num(recalls)
mean_recalls = np.mean(recalls)
std_recalls = np.std(recalls)
median_recalls = np.median(recalls)

print("acc")
print(mean_acc)
print(std_acc)
print(median_acc)

print("precision")
print(mean_precisions)
print(std_precisions)
print(median_precisions)

print("recall")
print(mean_recalls)
print(std_recalls)
print(median_recalls)
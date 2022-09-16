import os

# element_list = ['activate_body', 'inhibit_body','activate_head', 'inhibit_head','gene']
element_list = ['activate', 'gene','inhibit']
relation_list = ['activate_relation', 'inhibit_relation']

element_threshold = 0.6
relation_threshold = 0.6
rotated_relation_threshold=0.8
sub_image_folder_for_ocr = r'/home/fei/Desktop/vis_results_old/temp_sub_images/'

#interface configuration
element_model=r'output/element_model.pth'
element_config_file=r'./Base-RetinaNet.yaml'

relation_model=r'./output/body_model.pth'
relation_config_file=r'./Base-RetinaNet-relation.yaml'

rotated_relation_model=r'rotated_relation/relation_model.pth'
rotated_relation_config_file=r'./Base-RelationRetinaNet.yaml'

OCR_OFFSET = 5
cover_ratio = 0.3


# OCR configurations
test_home_folder = r'/home/fei/Desktop/weiwei/data/use_case/test/'  # home folder
image_folder = os.path.join(test_home_folder, "images")
ground_truth_folder = image_folder
predict_folder = os.path.join(test_home_folder, "predict")
failed_folder = os.path.join(test_home_folder, "failed")
previous_dictionary_path = ''  # none if not needed

log_file = os.path.join(predict_folder, "log.txt")
# dictionary_path = os.path.join(predict_folder, "gene_dictionary.xlsx")
# dictionary_path = r"./all_gene_names.json"
dictionary_path = r"./exHUGO_latest.json"
word_file = os.path.join(predict_folder, "word_cloud.txt")  # word cloud
all_results_file = os.path.join(predict_folder, "all_results.txt")

OCR_hist_step_size = 15  # should perfectly divide 255
OCR_hist_num_steps = 3  # num of steps to check
OCR_hist_num_sub_steps = 3  # should perfectly divide step size
#sub_step = step_size / num_sub_steps

candidate_threshold = 20  # do not show corrected_results if fuzz_ratio < candidate_threshold
threshold = 80  # do not proceed to next range unless best_fuzz_ratio > threshold
early_stop_threshold = 10  # for patience

patience_2 = 10  # stop if x consecutive bests >= threshold
patience = 3  # stop if x bests >= early_stop_threshold

vertical_ratio_thresh = 1.5  # rotate 90c and 90cc if height / width >= vertical_ratio_thresh
detection_IoU_thresholds = [.1, .25, .5, .75]  #  threshold for evaluation

padding = 50  # for deskew
OCR_SCALE = 5  # for resizing image




# end of file




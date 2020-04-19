import os




element_threshold = 0.8
relation_threshold = 0.8
rotated_relation_threshold=0.8
sub_image_folder_for_ocr = r'/home/fei/Desktop/vis_results_old/temp_sub_images/'

# train_task_id = '3T' + str(choice([1024, 1280, 1536, 1792, 2048]))
max_train_img_size = 736
train_task_id = '3T' +str(max_train_img_size)
DEBUG = True

# train_task_id = '3T1500'
initial_epoch = 0
epoch_num = 200
lr = 1e-4
decay = 5e-5

# clip_value = 0.5  # default 0.5, 0 means no clip

patience = 40
load_weights = True
lambda_inside_score_loss = 3.0
lambda_cls_loss = 3.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0


total_img = 2516
validation_split_ratio = 0.25

# max_train_img_size = int(train_task_id[-4:])
max_predict_img_size = max_train_img_size  # int(train_task_id[-4:])  # 2400

assert max_train_img_size in [512, 736, 1024, 1280, 1536, 1792, 2048], \
    'max_train_img_size must in [1024, 1280, 1536, 1792, 2048]'
if max_train_img_size == 512:
    batch_size = 10
elif max_train_img_size == 736:
    batch_size = 8
elif max_train_img_size == 1024:
    batch_size = 5
elif max_train_img_size == 1280:
    batch_size = 4
elif max_train_img_size == 1536:
    batch_size = 3
elif max_train_img_size == 1792:
    batch_size = 2
else:
    batch_size = 1

steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

train_data_dir = r'C:\Users\LSC-110\Desktop\pubmed'
origin_image_dir_name = r'Images'
origin_txt_dir_name = r'labels'
train_image_dir_name = r'images_%s' % train_task_id
train_label_dir_name = r'labels_%s' % train_task_id
show_gt_image_dir_name = r'show_gt_images_%s' % train_task_id
show_act_image_dir_name = r'show_act_images_%s' % train_task_id
val_fname = 'val_%s.txt' % train_task_id
train_fname = 'train_%s.txt' % train_task_id

# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.8
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 0, -1)

# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]

locked_layers = True

if not os.path.exists('saved_models'):
    os.mkdir('saved_models')

saved_model_weights_file_path = r'saved_models\model_weights.h5'

text_pixel_threshold = 0.8  # text foreground & background score
action_pixel_threshold = 0.8  # relation foreground & background score
text_side_vertex_pixel_threshold = 0.85
action_side_vertex_pixel_threshold = 0.85
text_trunc_threshold = 0.2
nock_trunc_threshold = 0.05
arrow_trunc_threshold = 0.9

predict_cut_text_line = False
predict_write2txt = True
crop_width = 736
crop_height = 736

# OCR configurations
test_home_folder = r'C:\Users\hefe\Desktop\use case'  # home folder
image_folder = os.path.join(test_home_folder, "images")
ground_truth_folder = image_folder
predict_folder = os.path.join(test_home_folder, "predict")
failed_folder = os.path.join(test_home_folder, "failed")
previous_dictionary_path = ''  # none if not needed

log_file = os.path.join(predict_folder, "log.txt")
# dictionary_path = os.path.join(predict_folder, "gene_dictionary.xlsx")
dictionary_path = r"./all_gene_names.json"
word_file = os.path.join(predict_folder, "word_cloud.txt")  # word cloud
all_results_file = os.path.join(predict_folder, "all_results.txt")

OCR_hist_step_size = 15  # should perfectly divide 255
OCR_hist_num_steps = 3  # num of steps to check
OCR_hist_num_sub_steps = 3  # should perfectly divide step size
#sub_step = step_size / num_sub_steps

candidate_threshold = 20  # do not show corrected_results if fuzz_ratio < candidate_threshold
threshold = 70  # do not proceed to next range unless best_fuzz_ratio > threshold
early_stop_threshold = 100  # for patience

patience_2 = 10  # stop if x consecutive bests >= threshold
patience = 3  # stop if x bests >= early_stop_threshold

vertical_ratio_thresh = 1.5  # rotate 90c and 90cc if height / width >= vertical_ratio_thresh
detection_IoU_thresholds = [.1, .25, .5, .75]  #  threshold for evaluation

padding = 50  # for deskew
OCR_SCALE = 5  # for resizing image
OCR_OFFSET = 0

# relationship configuration
relationship_model = r'saved_models\bottleneck_fc_model11.h5'
sub_img_width_for_relation_predict = 196
sub_img_height_for_relation_predict = 140
# relationship_folder = os.path.join(test_home_folder, "relationship")
# testing_data_folder = os.path.join(relationship_folder, "test")
#not_classified_folder = os.path.join(test_home_folder, "not_classified")

#different threshold configuration
threshold_start_point = 0.6
threshold_end_point = 0.99
threshold_step = 0.1
# end of file

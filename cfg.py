import os




element_threshold = 0.8
relation_threshold = 0.8
sub_image_folder_for_ocr = r'/home/fei/Desktop/vis_results_old/temp_sub_images/'
log_file = os.path.join(sub_image_folder_for_ocr, "log.txt")


# OCR configurations
#test_home_folder = r'C:\Users\hefe\Desktop\use case'  # home folder
#image_folder = os.path.join(test_home_folder, "images")
#ground_truth_folder = image_folder


#predict_folder = os.path.join(sub_image_folder_for_ocr, "predict")
failed_folder = os.path.join(sub_image_folder_for_ocr, "failed")
#previous_dictionary_path = ''  # none if not needed


#dictionary_path = os.path.join(predict_folder, "gene_dictionary.xlsx")
#word_file = os.path.join(predict_folder, "word_cloud.txt")  # word cloud
#all_results_file = os.path.join(predict_folder, "all_results.txt")

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

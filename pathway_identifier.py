import shutil
import pandas as pd
import torch
import numpy as np
import os
import cv2
import argparse
# from pathway_identifier.retinanet.dataloader import Resizer
from retinanet.dataloader import Resizer


def image_identifier(image_path, csv_path, model_path, output_path):
    model = torch.load(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.training = False
    model.eval()

    if csv_path != None:
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        file = pd.read_csv(csv_path, header=None)
        img_name_list = list(file.iloc[:, 0])
        img_label_list = list(file.iloc[:, 5])
        fn_list = []
        fp_list = []

        for img_name in os.listdir(image_path):
            if img_name.endswith('jpg'):
                gt_label = img_label_list[
                    img_name_list.index(os.path.join('D:\PyCharmProject\RetinaNet', image_path, img_name))]
                # print(gt_label)
                image = cv2.imread(os.path.join(image_path, img_name))
                sample = {'img': image, 'annot': np.array([[0., 0., 1., 1., 0.]])}
                resize_img = Resizer()
                out = resize_img(sample)
                input_img = out['img'].numpy()
                input_img = np.expand_dims(input_img, 0)
                input_img = np.transpose(input_img, (0, 3, 1, 2))

                with torch.no_grad():
                    image = torch.from_numpy(input_img)
                    if torch.cuda.is_available():
                        image = image.cuda()

                    scores, classification = model(image.cuda().float())
                    if gt_label == 'pathway':
                        if gt_label == classification[0]:
                            tp += 1
                        else:
                            fn += 1
                            fn_list.append(img_name)
                    elif gt_label == 'none':
                        if gt_label == classification[0]:
                            tn += 1
                        else:
                            fp += 1
                            fp_list.append(img_name)
        print('tp fp tn fn', tp, fp, tn, fn)  # 真正例（正->正）， 假正例（反->正）， 真反例（反->反）， 假反例（正->反）
        print('fn_list:', fn_list)
        print('fp_list:', fp_list)
        print('model: ', model_path, 'acc:', (tp + tn) / (tp + fp + tn + fn), 'precision:', tp / (tp + fp), 'recall:',
              tp / (tp + fn))
        print('Sn:', tp / (tp + fn), 'Sp:', tn / (tn + fp),
              'MCC:', ((tp * tn) - (fp * fn)) / (np.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))))
    else:
        # path = model_path.split('/')[-1].split('\\')[0]
        import platform
        if platform.system() == 'Windows':
            path = image_path.split('\\')[-1]
        else:
            path = image_path.split('/')[-1]
        pathway_path = output_path + '/' + path + '/img'
        none_path = 'none_pathway/' + path
        if not os.path.exists(pathway_path):
            os.makedirs(pathway_path)
        # print('pathwat_img:', pathway_path)
        if not os.path.exists(none_path):
            os.makedirs(none_path)
        # print(pathway_path, none_path)
        for img_name in os.listdir(image_path):
            if img_name.endswith('jpg'):
                image = cv2.imread(os.path.join(image_path, img_name))
                sample = {'img': image, 'annot': np.array([[0., 0., 1., 1., 0.]])}
                resize_img = Resizer()
                out = resize_img(sample)
                input_img = out['img'].numpy()
                input_img = np.expand_dims(input_img, 0)
                input_img = np.transpose(input_img, (0, 3, 1, 2))

                with torch.no_grad():
                    image = torch.from_numpy(input_img)
                    if torch.cuda.is_available():
                        image = image.cuda()

                    scores, classification = model(image.cuda().float())
                    # print(img_name, ' result:', scores[0][0], classification[0])
                    if classification[0] == 'pathway' and scores[0][0] >= 0.85:
                        shutil.copy(os.path.join(image_path, img_name), pathway_path)
                    else:
                        shutil.copy(os.path.join(image_path, img_name), none_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    # parser.add_argument('--image_dir', help='Path to directory containing images', default='test_data/')
    # parser.add_argument('--image_dir', help='Path to directory containing images', default='pathway_img/')
    # parser.add_argument('--image_dir', help='Path to directory containing images', default='data_prepare\\test_1')
    # parser.add_argument('--image_dir', help='Path to directory containing images', default='data_prepare\\new_test_data')
    # parser.add_argument('--image_dir', help='Path to directory containing images', default='data_prepare\\pathways')
    parser.add_argument('--image_dir', help='Path to directory containing images', default='D:/PyCharmProject/Full_pipeline/img')
    # parser.add_argument('--image_dir', help='Path to directory containing images', default=r'D:/PyCharmProject/mmocr1/My_code/validation images')

    parser.add_argument('--csv_path', help='Path to directory containing images', default=None)

    parser.add_argument('--model_path', help='Path to model', default='model/model_final.pt')

    parser.add_argument('--output_path', help='Path to output', default='result')

    parser = parser.parse_args()
    # model_list = ['model/train1', 'model/train2', 'model/train3', 'model/train4', 'model/train5']
    # for model_path in model_list:
    #     for m in os.listdir(model_path):
    #         print(os.path.join(model_path, m))
    #         parser.model_path = os.path.join(model_path, m)
    #         # parser.csv_path = 'data_prepare\\new_test.csv'
    #         # parser.csv_path = 'data_prepare\\pathway_test.csv'
    #         image_identifier(parser.image_dir, parser.csv_path, parser.model_path, parser.output_path)
    # parser.csv_path = 'data_prepare\\test1.csv'
    # parser.csv_path = 'data_prepare\\new_test.csv'
    image_identifier(parser.image_dir, parser.csv_path, parser.model_path, parser.output_path)
'''
model/train1\csv_retinanet_1.pt
class:  {'pathway': 0, 'none': 1}
tp fp tn fn 1 27 68 0
fn_list: []
fp_list: ['10.1016@j.taap.2019.114617_page8_86.jpg', 'bejjani2018_page2_17.jpg', 'bejjani2018_page4_30.jpg', 'canonica2019_page1_36.jpg', 'canonica2019_page3_91.jpg', 'canonica2019_page5_136.jpg', 'canonica2019_page6_149.jpg', 'hunt2019_page1_20.jpg', 'lethaus2020_page1_3.jpg', 'patel2019_page1_32.jpg', 'patel2019_page6_107.jpg', 'patel2019_page7_122.jpg', 'patel2019_page7_125.jpg', 'patel2019_page8_132.jpg', 'patel2019_page8_135.jpg', 'patel2019_page9_145.jpg', 'patel2019_page9_148.jpg', 'sessa2021_page1_101.jpg', 'sessa2021_page1_102.jpg', 'sessa2021_page7_145.jpg', 'shi2019_page0_95.jpg', 'shi2019_page1_140.jpg', 'yang2019_page2_21.jpg', 'yang2019_page3_25.jpg', 'yang2019_page3_26.jpg', 'yang2019_page6_39.jpg', 'yang2019_page7_45.jpg']
model:  model/train1\csv_retinanet_1.pt acc: 0.71875 precision: 0.03571428571428571 recall: 1.0
model/train2\csv_retinanet_2.pt
class:  {'pathway': 0, 'none': 1}
tp fp tn fn 1 27 68 0
fn_list: []
fp_list: ['10.1016@j.taap.2019.114617_page8_86.jpg', 'bejjani2018_page4_30.jpg', 'canonica2019_page1_36.jpg', 'canonica2019_page3_91.jpg', 'canonica2019_page5_136.jpg', 'canonica2019_page6_149.jpg', 'graham2018_page2_68.jpg', 'hunt2019_page1_20.jpg', 'lethaus2020_page1_3.jpg', 'lethaus2020_page2_7.jpg', 'oup-accepted-manuscript-2019_page2_78.jpg', 'patel2019_page1_32.jpg', 'patel2019_page2_41.jpg', 'patel2019_page6_107.jpg', 'patel2019_page7_122.jpg', 'patel2019_page7_125.jpg', 'patel2019_page8_132.jpg', 'patel2019_page8_135.jpg', 'patel2019_page9_148.jpg', 'sessa2021_page7_145.jpg', 'shi2019_page0_95.jpg', 'shi2019_page1_140.jpg', 'yang2019_page2_21.jpg', 'yang2019_page3_25.jpg', 'yang2019_page3_26.jpg', 'yang2019_page6_39.jpg', 'yang2019_page7_45.jpg']
model:  model/train2\csv_retinanet_2.pt acc: 0.71875 precision: 0.03571428571428571 recall: 1.0
model/train3\csv_retinanet_4.pt
class:  {'pathway': 0, 'none': 1}
tp fp tn fn 1 29 66 0
fn_list: []
fp_list: ['10.1016@j.taap.2019.114617_page8_86.jpg', 'bejjani2018_page2_17.jpg', 'bejjani2018_page4_30.jpg', 'bejjani2018_page5_35.jpg', 'canonica2019_page1_36.jpg', 'canonica2019_page3_91.jpg', 'canonica2019_page5_136.jpg', 'canonica2019_page6_149.jpg', 'hunt2019_page1_20.jpg', 'lethaus2020_page1_3.jpg', 'lethaus2020_page2_7.jpg', 'oup-accepted-manuscript-2019_page15_172.jpg', 'oup-accepted-manuscript-2019_page2_78.jpg', 'patel2019_page1_32.jpg', 'patel2019_page6_107.jpg', 'patel2019_page7_122.jpg', 'patel2019_page7_125.jpg', 'patel2019_page8_132.jpg', 'patel2019_page8_135.jpg', 'patel2019_page9_145.jpg', 'patel2019_page9_148.jpg', 'sessa2021_page7_145.jpg', 'shi2019_page1_140.jpg', 'shi2019_page5_216.jpg', 'yang2019_page2_21.jpg', 'yang2019_page3_25.jpg', 'yang2019_page3_26.jpg', 'yang2019_page6_39.jpg', 'yang2019_page7_45.jpg']
model:  model/train3\csv_retinanet_4.pt acc: 0.6979166666666666 precision: 0.03333333333333333 recall: 1.0
model/train4\csv_retinanet_1.pt
class:  {'pathway': 0, 'none': 1}
tp fp tn fn 1 34 61 0
fn_list: []
fp_list: ['balzarro2017_page1_4.jpg', 'bejjani2018_page2_17.jpg', 'bejjani2018_page4_30.jpg', 'bejjani2018_page5_35.jpg', 'canonica2019_page1_36.jpg', 'canonica2019_page3_91.jpg', 'canonica2019_page5_136.jpg', 'canonica2019_page6_149.jpg', 'graham2018_page2_68.jpg', 'hunt2019_page1_20.jpg', 'lethaus2020_page1_3.jpg', 'lethaus2020_page2_7.jpg', 'oup-accepted-manuscript-2019_page15_172.jpg', 'oup-accepted-manuscript-2019_page2_78.jpg', 'patel2019_page1_32.jpg', 'patel2019_page2_41.jpg', 'patel2019_page2_44.jpg', 'patel2019_page3_60.jpg', 'patel2019_page6_107.jpg', 'patel2019_page7_122.jpg', 'patel2019_page7_125.jpg', 'patel2019_page8_132.jpg', 'patel2019_page8_135.jpg', 'patel2019_page9_145.jpg', 'patel2019_page9_148.jpg', 'sessa2021_page1_101.jpg', 'sessa2021_page7_145.jpg', 'shi2019_page0_95.jpg', 'shi2019_page1_140.jpg', 'yang2019_page2_21.jpg', 'yang2019_page3_25.jpg', 'yang2019_page3_26.jpg', 'yang2019_page6_39.jpg', 'yang2019_page7_45.jpg']
model:  model/train4\csv_retinanet_1.pt acc: 0.6458333333333334 precision: 0.02857142857142857 recall: 1.0
model/train5\csv_retinanet_0.pt
class:  {'pathway': 0, 'none': 1}
tp fp tn fn 1 30 65 0
fn_list: []
fp_list: ['10.1016@j.cbpc.2019.04.016_page0_754.jpg', 'bejjani2018_page1_5.jpg', 'bejjani2018_page2_17.jpg', 'bejjani2018_page4_30.jpg', 'canonica2019_page1_36.jpg', 'canonica2019_page3_91.jpg', 'canonica2019_page5_136.jpg', 'canonica2019_page6_149.jpg', 'graham2018_page2_68.jpg', 'lethaus2020_page1_3.jpg', 'lethaus2020_page2_7.jpg', 'oup-accepted-manuscript-2019_page15_172.jpg', 'oup-accepted-manuscript-2019_page2_78.jpg', 'patel2019_page1_32.jpg', 'patel2019_page2_41.jpg', 'patel2019_page3_60.jpg', 'patel2019_page7_125.jpg', 'patel2019_page8_132.jpg', 'patel2019_page8_135.jpg', 'patel2019_page9_145.jpg', 'patel2019_page9_148.jpg', 'sessa2021_page1_101.jpg', 'sessa2021_page7_145.jpg', 'shi2019_page0_95.jpg', 'shi2019_page1_140.jpg', 'yang2019_page2_21.jpg', 'yang2019_page3_25.jpg', 'yang2019_page3_26.jpg', 'yang2019_page6_39.jpg', 'yang2019_page7_45.jpg']
model:  model/train5\csv_retinanet_0.pt acc: 0.6875 precision: 0.03225806451612903 recall: 1.0

Process finished with exit code 0

'''

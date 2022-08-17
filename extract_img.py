import os
import difflib
import json
import shutil
from glob import glob

import fitz

from fitz import fitz
from pathlib import Path
from coverage.annotate import os
from multiprocessing import Process

from extract_new import OutlineException

pdf_path = 'fetched_pdfs'
img1_path = 'img1'
img2_path = 'img2'

def image_profile_2(page,img_count):
    image_caption = []
    for i in range(img_count):
        image_caption.append("-")
    pagetext = page.get_text("blocks")  # blocks
    for j in range(img_count):
        for page_info_index in range(len(pagetext)):
            i = str(pagetext[page_info_index]).find('<image:')
            if i != -1 and page_info_index < len(pagetext)-1:
                image_caption[j] = (pagetext[page_info_index + 1][4])
            # elif i != -1 and page_info_index == len(pagetext)-1:
            #     image_caption.append('-')
    return image_caption

def extract_image(pdf_file,img_path):

    doc = fitz.open(pdf_file)
    pdf_name = str(pdf_file).split('\\')[1].split('.pdf')[0]
    save_path_root = Path(img_path)

    # extract image
    image_dicts = []
    dir_name = os.path.join(save_path_root, pdf_name)
    # if not os.path.isdir(dir_name):
    #     os.makedirs(dir_name)
    # save_path_root = dir_name  # 按文件名分出单个目录

    for current_page in range(len(doc)):  # every page
        count = 0
        image_caption = []
        for image in doc.get_page_images(current_page):  # every image for the page
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            save_path_root = dir_name  # 按文件名分出单个目录
            page = doc.load_page(current_page)
            xref = image[0]
            pix = fitz.Pixmap(doc, xref)
            save_path = os.path.join(save_path_root, "%s_page%s_%s.jpg" % (pdf_name, current_page, xref))
            if pix.n < 4:  # this is GRAY or RGB
                pix.pil_save(save_path)
            else:  # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.save(save_path)
                pix1 = None
            pix = None

            img_count = len(doc.get_page_images(current_page))  # image count of the page
            image_caption = image_profile_2(page,img_count) # extract the next paragraph as the image caption
            image_caption[count] = image_caption[count].replace('\n', ' ')
            dic = {
                'Figure': pdf_name + "_page" + str(current_page) + "_" + str(xref) + ".jpg",
                'Figure_title': image_caption[count]
            }
            image_dic = {
                'page':current_page,'count':count,
                'image_name':pdf_name + "_page" + str(current_page) + "_" + str(xref) + ".jpg",
                'image_title':image_caption[count]
            }
            image_dicts.append(image_dic)
            outline_json_file_2 = open(str(save_path_root) + "/%s_page%s_%s.json" % (pdf_name, current_page, xref)
                                     , 'w', encoding='utf-8')
            outline_content_2 = json.dumps(dic, indent=2, sort_keys=True, ensure_ascii=False)
            outline_json_file_2.write(outline_content_2)
            count += 1


if __name__ == '__main__':
    i = 0
    pdf_file_path = Path(pdf_path)
    for pdf_file in pdf_file_path.glob("*.pdf"):
        pdf_name = os.path.split(pdf_file)[1].split('.')[0]
        print("[" + str(i + 1) + "]"+pdf_name)
        i += 1
        doc = fitz.open(pdf_file)
        outline_list = doc.get_toc()  # 获取目录
        if outline_list != []:
            print("have outline")
            extract_image(pdf_file,img1_path)
        else:
            print("not have outline")
            extract_image(pdf_file, img2_path)

    # # src_dir = './img_outline/' # 目的路径记得加斜杠
    # src_dir = './img_no_outline/'  # 目的路径记得加斜杠
    # src_file_list = glob(src_dir + "*")  # glob获得路径下所有文件，可根据需要修改
    # for srcfile in src_file_list:
    #     # print(srcfile)
    #     img_list = glob(srcfile + "/*.json")
    #     for i in img_list:
    #         os.remove(i)
    #         print("删除成功！")

    # # copy the image to new package
    # src_dir = './img2/'
    # dst_dir = './img_no_outline/'  # 目的路径记得加斜杠
    # src_file_list = glob(src_dir + "*.pdf")  # glob获得路径下所有文件，可根据需要修改
    # for srcfile in src_file_list:
    #     mycopyfile(srcfile, dst_dir)  # 复制文件




import ast
import os
import difflib
import json
import shutil

import fitz

from fitz import fitz
from pathlib import Path
from coverage.annotate import os
from typing_extensions import final

image_dicts = []
pdf_path = 'fetched_pdfs'
img1_path = 'img_new'
# img2_path = 'img_2'
text_path = Path('test_fetched_pdfs_temp')

class OutlineException(Exception):
    '''自定义的异常类'''

    def __init__(self, outline_list):
        self.outline_list = outline_list

    def __str__(self):
        if self.outline_list != []:
            return "have outline!"
        else:
            return "not have outline, skip the file"

def image_profile_2(page, img_count):
    image_caption = []
    for i in range(img_count):
        image_caption.append("-")
    pagetext = page.get_text("blocks")  # blocks
    for j in range(img_count):
        for page_info_index in range(len(pagetext)):
            i = str(pagetext[page_info_index]).find('<image:')
            if i != -1 and page_info_index < len(pagetext) - 1:
                image_caption[j] = (pagetext[page_info_index + 1][4])
    return image_caption

def extract_image(pdf_file, img_path):
    doc = fitz.open(pdf_file)
    pdf_name = os.path.split(pdf_file)[1].split('.pdf')[0]
    save_path_root = Path(img_path)

    # extract image
    global image_dicts
    dir_name = os.path.join(save_path_root, pdf_name)

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
            image_caption = image_profile_2(page, img_count)  # extract the next paragraph as the image caption
            image_caption[count] = image_caption[count].replace('\n', ' ')
            dic = {
                'Figure': pdf_name + "_page" + str(current_page) + "_" + str(xref) + ".jpg",
                'Figure_title': image_caption[count]
            }
            image_dic = {
                'page': current_page, 'count': count,
                'image_name': pdf_name + "_page" + str(current_page) + "_" + str(xref) + ".jpg",
                'image_title': image_caption[count]
            }
            image_dicts.append(image_dic)
            outline_json_file_2 = open(str(save_path_root) + "/%s_page%s_%s.json" % (pdf_name, current_page, xref)
                                       , 'w', encoding='utf-8')
            outline_content_2 = json.dumps(dic, indent=2, sort_keys=True, ensure_ascii=False)
            outline_json_file_2.write(outline_content_2)
            count += 1

def get_front_text(doc):
    page_texts_front = []
    if doc.page_count >= 1:
        page0 = doc.load_page(0)
        pagetext0 = page0.get_text("blocks")  # [bbox,text,paragraph,text(0),image(1)]
        page_texts_front.append(pagetext0)
    if doc.page_count >= 2:
        page1 = doc.load_page(1)
        pagetext1 = page1.get_text("blocks")  # [bbox,text,paragraph,text(0),image(1)]
        page_texts_front.append(pagetext1)
    if doc.page_count >= 3:
        page2 = doc.load_page(2)
        pagetext2 = page1.get_text("blocks")  # [bbox,text,paragraph,text(0),image(1)]
        page_texts_front.append(pagetext2)
    return page_texts_front

def get_metadata(doc):
    # print(doc.metadata)
    metadata = []
    paper_title = ''
    author = ''
    abstract = ''
    keywords = ''

    # extract paper_title
    if doc.metadata.get('title', "") != "":
        paper_title = doc.metadata.get('title')
        # print(metadata)
    else:  ### extract paper title through the biggest font size
        if doc.page_count >= 1:
            page0 = doc.load_page(0)
            pagetext0_size = page0.get_text("xml")  # xml: font size
            page0_size = str(pagetext0_size).split('\n')  # split the text in every line
            paper_titles = []  # font标签下的文字列表
            size = 0
            size_rank_list = []
            paper_title_temp = ''
            for i in range(len(page0_size)):
                # print(page0_size[i])
                if page0_size[i].find('" size="') != -1:  # 当前行是font起始标签行，有size
                    position = page0_size[i].find('" size="')
                    size = float(page0_size[i][position + 8:len(page0_size[i]) - 2])
                    size_rank_list.append(size)
                    # print("size :",size)
                elif page0_size[i].find(' c="') != -1:
                    position = page0_size[i].find(' c="')
                    paper_title_temp += page0_size[i][position + 4:position + 5]
                elif page0_size[i].find('</font>') != -1:
                    dict = {'size': size, 'title_temp': paper_title_temp}
                    paper_titles.append(dict)
                    size = 0
                    paper_title_temp = ''
                    # print(dict)
            size_rank_list = list(set(size_rank_list))
            size_rank_list.sort(reverse=True)
            titles = []
            for i in range(len(size_rank_list)):
                paper_title_temp = ''
                tag = 1  # 间断控制
                # print("size:",size_rank_list[i])
                for j in range(len(paper_titles)):
                    # print(paper_titles[j].get('size'))
                    if paper_titles[j].get('size') == size_rank_list[i]:
                        tag = 0
                        # print(size_rank_list[i],paper_titles[j])
                        paper_title_temp += paper_titles[j].get('title_temp')
                    elif paper_titles[j].get('size') != size_rank_list[i] and tag == 0:
                        titles.append(paper_title_temp)
                        break
            for i in range(len(titles)):
                paper_title = titles[i]
                # print("#####",paper_title)
                if len(titles[i]) < 20:
                    continue
                elif len(titles[i]) >= 20:
                    break
            # print(size_rank_list)
    print("paper_title:  ", paper_title)

    # extract author
    if doc.page_count >= 1:
        page0 = doc.load_page(0)
        pagetext0 = page0.get_text("blocks")
        for i in range(len(pagetext0)):
            # print(pagetext0[i])
            # print(pagetext0[i][4].replace('\n', ' '))
            if pagetext0[i][4].replace('\n', ' ') == paper_title \
                    or (difflib.SequenceMatcher(None, pagetext0[i][4].replace('\n', ' '),
                                                paper_title).quick_ratio() >= 0.75):
                try:
                    author = pagetext0[i+1][4].replace('\n', ' ')
                except:
                    return
                break
    print("author: ", author)

    # extract keywords
    page_texts_front = get_front_text(doc)
    k1 = 0
    k2 = 0
    keywords_tag = ['Keywords', 'Key words', 'KEYWORDS', 'Keyword', 'Summary', 'Keywords:']
    for pagetext in page_texts_front:
        for i in range(len(pagetext)):
            # print("####  ", pagetext[i][4].replace('\n', ' '))
            this_line = pagetext[i][4].replace('\n', ' ')
            if len(this_line) < 10:
                tag = 0
                for j in keywords_tag:
                    if this_line == j or (difflib.SequenceMatcher(None, this_line, j).quick_ratio() >= 0.8):
                        tag = 1
                        break
                if tag == 1:
                    keywords = pagetext[i + 1][4].replace('\n', ' ')
                    k1 = i
                    k2 = i + 1
            elif len(this_line) >= 10:
                for j in keywords_tag:
                    if this_line[:len(j)] == j or (
                            difflib.SequenceMatcher(None, this_line[:len(j)], j).quick_ratio() >= 0.8):
                        keywords = this_line.replace(j, '')
                        k1 = i
                        k2 = i
    print("keywords: ", keywords)

    # extract abstract
    page_texts_front = get_front_text(doc)
    abstract_tag = ['ABSTRACT', 'Abstract', 'abstract', 'Summary', 'a b s t r a c t']
    for pagetext in page_texts_front:
        for i in range(len(pagetext)):
            # print("####  ",pagetext[i][4].replace('\n', ' '))
            this_line = pagetext[i][4].replace('\n', ' ')
            if len(this_line) < 50:
                tag = 0
                for j in abstract_tag:
                    if this_line == j or (difflib.SequenceMatcher(None, this_line, j).quick_ratio() >= 0.8):
                        tag = 1
                if tag == 1:
                    abstract = pagetext[i + 1][4].replace('\n', ' ')
            elif len(this_line) >= 50:
                for j in abstract_tag:
                    if this_line[:len(j)] == j or (
                            difflib.SequenceMatcher(None, this_line[:len(j)], j).quick_ratio() >= 0.8):
                        abstract = this_line.replace(j, '')

    ## make the next paragraph after author to be abstract
    if (abstract == "" or len(abstract) < 40) and author != '':
        for pagetext in page_texts_front:
            for i in range(len(pagetext)):
                # print("####  ", pagetext[i][4].replace('\n', ' '))
                this_line = pagetext[i][4].replace('\n', ' ')
                try:
                    if difflib.SequenceMatcher(None, this_line, author).quick_ratio() >= 0.8:
                        if len(pagetext[i + 1][4].replace('\n', ' ')) >= 500:
                            abstract = pagetext[i + 1][4].replace('\n', ' ')
                            # print("abstract_author",abstract)
                        elif len(pagetext[i + 2][4].replace('\n', ' ')) >= 500:
                            abstract = pagetext[i + 2][4].replace('\n', ' ')
                except:
                    return

    ## make the paragraph after or before  to be abstract
    if (abstract == '' or len(abstract) <= 40) and keywords != '':
        abstract_1 = ''
        abstract_2 = ''
        tag = 0
        for pagetext in page_texts_front:
            # print("len: ",len(pagetext))
            for i in range(len(pagetext)):
                # print("####  ", pagetext[i][4].replace('\n', ' '))
                this_line = pagetext[i][4].replace('\n', ' ')
                if i == k1:
                    abstract_1 = pagetext[i - 1][4].replace('\n', ' ')
                if i == k2:
                    if i < len(pagetext) - 1:
                        abstract_2 = pagetext[i + 1][4].replace('\n', ' ')
                        tag = 1
                        break
            if tag == 1:
                break
        if len(abstract_1) >= len(abstract_2):
            abstract = abstract_1
        else:
            abstract = abstract_2
        # print("abstract_1",abstract_1)
        # print("abstract_2",abstract_2)
    print("abstract: ", abstract)

    metadata.append({'Paper_title': paper_title})
    metadata.append({'Author': author})
    metadata.append({'Abstract': abstract})
    metadata.append({'Keywords': keywords})
    # print(metadata)

    return metadata

def deal_with_outline(doc, metadata):
    dics = []

    # extract original outline
    outline_list = doc.get_toc()  # 检索文档中存在的文本大纲 , 返回列表 [layer,title,page]
    # pdf_name = os.path.split(pdf_file)[1].split('.pdf')[0]
    # print("目录: ")
    for i in outline_list:
        dic = {
            "section": i[0], "title": str(i[1]), "page": i[2]
        }
        if i[0] == 0:
            dic['section'] = i[0] + 1
        dics.append(dic)

    # remove the first section when the title in section
    try:
        paper_title = metadata[0].get("Paper_title")
    except:
        return
    print(dics[0]["title"])
    if difflib.SequenceMatcher(None, dics[0]["title"], paper_title).quick_ratio() >= 0.8\
            or difflib.SequenceMatcher(None, dics[0]["title"][:len(paper_title)+2], paper_title).quick_ratio() >= 0.8:
        dics.pop(0)
        for dic in dics:
            if dic["section"] == 1:
                continue
            else:
                dic["section"] = dic.get("section") - 1

    # remove the first section when the abstract in section
    abstract_tag = ['ABSTRACT', 'Abstract', 'abstract', 'Summary', 'a b s t r a c t', 'Main']
    for i in abstract_tag:
        if difflib.SequenceMatcher(None, dics[0]["title"], i).quick_ratio() >= 0.8:
            dics.pop(0)
            break

    # print("目录")
    # for i in dics:
    #     print(i)

    return dics

def make_level(text_group):
    #  思路：找到最大深度，从最大深度向前收缩，收缩要有顺序：从最后的位置向前收缩
    deepest = 0
    for i in range(len(text_group)):
        level = int(text_group[i].split("{'section")[1][:1])
        if level > deepest:
            deepest = level
    print("deepest",deepest)

    for i in range(len(text_group)):
        section_level = text_group[i][9:10]
        if int(section_level) != 1:
            text_group[i] = text_group[i][:9] + str(int(section_level)-1) + text_group[i][10:]
        else:
            break

    if deepest > 1:  # deepest is 1 ,then no deal
        for i in range(deepest,1,-1):
            for j in range(len(text_group)-1, -1, -1):
                first_section_level = int(text_group[j][9])
                if first_section_level == i:
                    brefore_section_level = int(text_group[j-1][9])
                    # print("first_section_level",first_section_level,"brefore_section_level",brefore_section_level)
                    if first_section_level == brefore_section_level:  # 向前添加
                        # print(j-1,text_group[j-1])
                        # print(j,text_group[j])
                        text_group[j-1] = text_group[j-1][:len(text_group[j-1])] + ',' + text_group[j]
                        text_group.pop(j)
                    elif first_section_level > brefore_section_level:  # 向前缩
                        text_group[j-1] = text_group[j-1][:len(text_group[j-1])-2] + ',' + text_group[j] + "]}"
                        text_group.pop(j)
                        continue

    # for i in text_group:
    #     print("After   ", i)

    return text_group

def get_pdf_text(pdf_file):
    global text_path,image_dicts
    # paper: every line text list , dics: title list

    if not os.path.isdir(text_path):
        os.makedirs(text_path)

    doc = fitz.open(pdf_file)
    pdf_name = os.path.split(pdf_file)[1].split('.pdf')[0]

    # get medadata
    metadata = get_metadata(doc)
    # get bodytext
    body_text = []

    # deal with the outline
    dics = deal_with_outline(doc, metadata)

    # extract body text(start with section1)
    try:
        first_section = dics[0]['title']
    except:
        return
    first_section_page = int(dics[0]['page'])
    # print(first_section_page, first_section)

    # 问题出在，找不到第一个标题的位置
    if first_section_page == -1:
        print("the outline has error!")
        return
    original_texts = []
    tag = 0
    if len(dics) > 1:
        for current_page in range(first_section_page-1,len(doc)):
            page = doc.load_page(current_page)
            pagetext = page.get_text("blocks")
            for page_info in pagetext:  # every paragraph in the page
                text_line = page_info[4].replace('\n', '')
                # print("###",text_line)
                if tag == 0 and (text_line[:len(first_section)+8].find(first_section) != -1
                                 or difflib.SequenceMatcher(None, first_section, text_line[:len(first_section)+4]).quick_ratio() >= 0.8
                                 or text_line[:len(first_section)+8].find(first_section.upper()) != -1
                                 or text_line[:len(first_section)+8].find(first_section.upper()) != -1
                                 or difflib.SequenceMatcher(None, first_section.upper(), text_line[:len(first_section)+4]).quick_ratio() >= 0.8) :
                    tag = 1
                    # print("1111111111")
                    if len(text_line)-len(first_section) > 8:
                        original_texts.append(text_line[:len(first_section)])
                        original_texts.append(text_line[len(first_section):])
                    else:
                        original_texts.append(text_line)
                elif tag == 1:
                    # print("2222222222")
                    original_texts.append(text_line)

    if tag == 0 and original_texts == []:  # 定位不到标题，并且标题可能是有意义的（无意义也这么做吧）
        for i in range(len(dics)):
            current_section = dics[i]['title']
            # print(current_section)
            for current_page in range(first_section_page - 1, len(doc)):
                page = doc.load_page(current_page)
                pagetext = page.get_text("blocks")
                for page_info in pagetext:  # every paragraph in the page
                    text_line = page_info[4].replace('\n', '')
                    # print("###",text_line)
                    if tag == 0 and (text_line[:len(current_section) + 8].find(current_section) != -1
                                     or difflib.SequenceMatcher(None, current_section, text_line[:len(
                                current_section) + 4]).quick_ratio() >= 0.8
                                     or text_line[:len(current_section) + 8].find(current_section.upper()) != -1
                                     or text_line[:len(current_section) + 8].find(current_section.upper()) != -1
                                     or difflib.SequenceMatcher(None, current_section.upper(), text_line[:len(
                                current_section) + 4]).quick_ratio() >= 0.8):
                        tag = 1
                        # print("1111111111")
                        if len(text_line) - len(current_section) > 8:
                            original_texts.append(text_line[:len(current_section)])
                            original_texts.append(text_line[len(current_section):])
                        else:
                            original_texts.append(text_line)
                    elif tag == 1:
                        # print("2222222222")
                        original_texts.append(text_line)
            if original_texts != []:
                break
        #
        # if original_texts == []:  # 一个标题都找不到，这咋办呢，用第一个标题替代吧。
        #
    ## when the outline is confused, it has no sense ,then make the last metadata to be body_text
    if len(dics) == 1 or original_texts == []:
        start_text = ''
        for i in metadata:
            for key in i.keys():
                if i[key] != '':
                    start_text = i[key]
        print(start_text)

        for current_page in range(len(doc)):
            page = doc.load_page(current_page)
            pagetext = page.get_text("blocks")
            for page_info in pagetext:  # every paragraph in the page
                text_line = page_info[4].replace('\n', '')
                # print("###",text_line)
                if (text_line[:len(start_text) + 8].find(start_text) != -1
                                 or difflib.SequenceMatcher(None, start_text,
                                                            text_line[:len(start_text) + 4]).quick_ratio() >= 0.8
                                 or text_line[:len(start_text) + 8].find(start_text.upper()) != -1
                                 or text_line[:len(start_text) + 8].find(start_text.upper()) != -1
                                 or difflib.SequenceMatcher(None, start_text.upper(),
                                                            text_line[:len(start_text) + 4]).quick_ratio() >= 0.8):
                    tag = 1
                    if len(text_line) - len(start_text) > 8:
                        original_texts.append(text_line[:len(start_text)])
                        original_texts.append(text_line[len(start_text):])
                    else:
                        original_texts.append(text_line)
                elif tag == 1:
                    original_texts.append(text_line)
        original_texts.pop(0)

    # print("original_texts", original_texts)

    # seperate the section and text when the section and text in one line
    new_text1 = original_texts
    k = 0
    for i in range(len(dics)):
        section_name = dics[i].get("title")
        section_level = dics[i].get("section")
        # print(section_name)
        for j in range(k,len(original_texts)):
            text_line = original_texts[j]
            if (text_line[:len(section_name)+8].find(section_name) != -1
                or difflib.SequenceMatcher(None, section_name, text_line[:len(section_name)+4]).quick_ratio() >= 0.8)\
                    or text_line[:len(section_name)+8].find(section_name.upper()) != -1\
                    or difflib.SequenceMatcher(None, section_name.upper(), text_line[:len(section_name)+4]).quick_ratio() >= 0.8:  # 找到标题位置
                k = j
                # print(text_line)
                new_text1[j] = str("<section")+str(section_level)+" " + new_text1[j]
                if len(text_line)-len(section_name) > 8:
                    new_text1[j] = str("<section")+str(section_level)+" " +text_line[len(section_name):]
                    new_text1.insert(j,text_line[:len(section_name)])
                    # print("new_text1", new_text1[j])
                    # print("new_text2", new_text1[j+1])
                break

    # change the image and figure
    img_count = 0
    img_ed_count = len(image_dicts)
    # if image_dicts != []:
    #     print(image_dicts)
    print("count_extract: ",img_ed_count)

    for i in range(len(new_text1)):
        if new_text1[i][:10].find("<image:") != -1:
            # print(new_text1[i])
            img_count += 1
    print("count: ",img_count)
    # print("new text one", new_text1)

    ###########  问题出现在new text one 和 new text two 上

    new_text2 = []
    if img_ed_count == 0 and img_count == 0:  # 压根就没图片
        new_text2 = new_text1
    elif img_ed_count == 0 and img_count != 0:  # 提取出的图片个数为0，但有image标签，则删除标签。
        for j in range(img_count):
            for i in range(len(new_text1)):
                if new_text1[i].find("<image:") != -1:
                    new_text1.pop(i)
                    break
        new_text2 = new_text1
    elif img_ed_count == img_count: # 提取出的和image标签个数相等。直接按顺序替换
        j = 0
        tag = 0
        for i in range(len(new_text1)):
            if new_text1[i].find("<image:") != -1:
                image_name = image_dicts[j].get('image_name')
                image_title = image_dicts[j].get('image_title')
                dict_temp_1 = {"Figure": image_name, "Figure_title": image_title}
                new_text2.append(str(dict_temp_1))
                tag = 1
                j += 1
            elif tag == 1:
                tag = 0
                continue
            else:
                new_text2.append(new_text1[i])
    elif img_ed_count < img_count or img_ed_count > img_count:  # 提取出的少于image标签的，倒数配对
        j = img_ed_count-1
        for i in range(len(new_text1)-1, -1, -1):
            if new_text1[i].find("<image:") != -1 and j >= 0:
                # print(new_text1[i])
                # print(image_dicts[j])
                image_name = image_dicts[j].get('image_name')
                image_title = image_dicts[j].get('image_title')
                dict_temp_1 = {"Figure": image_name, "Figure_title": image_title}
                new_text2.append(str(dict_temp_1))
                j -= 1
            else:
                new_text2.append(new_text1[i])
        new_text2.reverse()

        for i in range(len(new_text2)-1,-1,-1):
            # print(new_text2[i])
            if new_text2[i].find("{'Figure':") != -1 and i < len(new_text2)-1:
                new_text2.pop(i+1)

    # print("new text two", new_text2)

    # make paragraph, make section group, make layer
    # make paragraph
    paragraph = 0
    for i in range(len(new_text2)):
        if new_text2[i].find("<section") != -1: # 当前为标题行
            paragraph = 1
            continue
        elif new_text2[i].find("{'Figure': ") != -1: # 当前为图片行
            continue
        else:
            temp = {"paragraph"+str(paragraph):new_text2[i]}
            new_text2[i] = str(temp)
            paragraph += 1

    # for i in new_text2:
    #     print(i)

    ## make section group
    text_group = []
    section_tag_count = 0
    for i in range(len(new_text2)):
        if new_text2[i].find("<section") != -1:  # 当前为标题行
            section_tag_count += 1
    for i in range(section_tag_count):
        text_group.append("")
    print("section_tag_count", section_tag_count, "text_group_count", len(text_group))

    if section_tag_count == len(text_group) and section_tag_count != 0:
        j = 0
        section_layer = ""
        section_title = ""
        temp = ""
        for i in range(len(new_text2)):
            if new_text2[i].find("<section") != -1:  # 当前为标题行
                section_layer = new_text2[i].split("<section")[1][:1]
                section_title = new_text2[i].split("<section")[1][2:]
                # print(section_layer,section_title)
                section_temp_dict = {"section_title": section_title}
                temp = str({"section"+str(section_layer): [section_temp_dict]})
                # print(temp)
                text_group[j] = temp
                j += 1
            else:
                # print("标题", text_group[j-1])
                text_group[j-1] = text_group[j-1][:len(text_group[j-1])-2] + ',' + new_text2[i] + ']}'
                # print(text_group[j-1])
        # text_group = make_level(text_group)

    elif section_tag_count == len(text_group) and section_tag_count == 0:
        section_temp_dict = {"section_title": dics[0]['title']}
        temp = str({"section1": [section_temp_dict]})
        for i in range(len(new_text2)):
            if i == 0:
                text_group.append(temp)
            else:
                text_group[0] = text_group[0][:len(text_group[0])-2] + ',' + new_text2[i] + ']}'

    ##  make level
    # for i in text_group:
    #     print(i, "text_group   ")
    text_group = make_level(text_group)
    for i in range(len(text_group)):
        dict = ast.literal_eval(text_group[i])
        body_text.append(dict)

    text_json = [{'metatada': metadata}, {'body_text': body_text}]

    json_file_name = str(text_path) + '/' + pdf_name + '_text.json'
    file = open(json_file_name, 'w', encoding='utf-8')  # text to json file
    file_content = json.dumps(text_json, indent=2, ensure_ascii=False)
    file.write(file_content)

    image_dicts = []

def extract_information(pdf_file):
    global image_dicts
    image_dicts = []

    try:
        doc = fitz.open(pdf_file)
    except:
        print("pdf can't open, maybe damaged")
        return

    outline_list = doc.get_toc()  # 获取目录
    # print(pdf_file)
    pdf_name = os.path.split(pdf_file)[1].split('.pdf')[0]
    print('\n\n', pdf_name, "  page:", doc.page_count)

    try:
        if outline_list != []:
            exc = OutlineException(outline_list)
            print(exc)
            extract_image(pdf_file, img1_path)
            get_pdf_text(pdf_file)  # extract all text,image in paper to json
        else:
            # extract_image(pdf_file, img2_path)
            extract_image(pdf_file, img1_path)
            raise OutlineException(outline_list)
    except OutlineException as e:
        print(e)

if __name__ == '__main__':

    pdf_file_path = Path(pdf_path)
    for pdf_file in pdf_file_path.glob("*.pdf"):
        pdf_name = os.path.split(pdf_file)[1].split('.')[0]
        extract_information(pdf_file)  # extract pdf text to json
    # for json_file in Path('test_fetched_pdfs').glob('*.json'):
    #     print(json_file, os.path.getsize(json_file))
    #     if os.path.getsize(json_file) < 2000:
    #         pdf_name = os.path.split(json_file)[1].split('.json')[0].split('_text')[0]
    #         print(pdf_name)
    #         shutil.copy(os.path.join('fetched_pdfs', pdf_name+'.pdf'),
    #                     os.path.join('body_text_problem_pdf', pdf_name+'.pdf'))

    # for json_file in text_path.glob("*.json"):
    #     json_name = os.path.split(json_file)[1].split('.')[0]
    #     print(json_name)
    #     with open(json_file,'r',encoding='UTF-8') as f:
    #         d = json.load(f)
    #     data = d[1]['body_text']
    #     final = []
    #     for i in data:
    #         section_data = i['section1']
    #         for j in section_data:
    #             key = list(j.keys())[0]
    #             if key == "section_title":
    #                 if j[key] == "References" or j[key] == "Acknowledgments" or j[key] == 'Acknowledgements':
    #                     break
    #             if "para" in key:
    #                 final.append(j[key])
    #     print(final)

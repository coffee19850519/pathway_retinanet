import ast
import os
import difflib
import json
import fitz

from PyPDF2 import PdfFileReader as pdf_read
from fitz import fitz
from pathlib import Path
from coverage.annotate import os

dics = []
paper = []

def function3(list, layer):
    global dics
    n = 0
    list_back = []
    for i in range(len(list)):
        # print(dics[i])
        if int(dics[i]['section']) == layer:
            n += 1
            list_back.append([])
    n = 0

    for i in range(len(list)):
        if int(dics[i]['section']) == layer:
            temp = [list[i]]
            # print("enenen",temp)
            for j in range(i+1, len(list)):
                if int(dics[j]['section']) > int(dics[i]['section']):
                    temp.append(list[j])
                    # print(list[j])
                elif int(dics[j]['section']) == int(dics[i]['section']):
                    break
            if len(temp) ==1:
                list_back[n] = temp[0]
            else:
                list_back[n] = temp
            n += 1
    return list_back

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

def get_pdf_image(pdf_file):  # extract image and caption,put the information in json

    doc = fitz.open(pdf_file)
    # print("paper:",str(pdf_file),",number of pages: %i" % doc.page_count)

    pdf_name = str(pdf_file).split('\\')[1]
    pdf_name = pdf_name.split('.pdf')[0]

def get_file_name(file_path,endswitch):  # get the certain kind of files ,then return the file name in list(not include suffix)
    file_list = []
    for name in os.listdir(file_path):
        if name.endswith(endswitch):
            file_list.append(name)
    return file_list

def filter(string):
    x = ''
    for i in string:
        if i.isdigit() or i == '.':
            x += i
    return float(x)

def func(L): # return the list deepest deep
    if type(L) is not list:
        return 0
    k = 1
    while any([type(i) is list for i in L]):
        k += 1
        L = [i for i in L if type(i) is list]
        L = [j for i in L for j in i]
    return k

def make_layer(a,deep,dics):
    # print(a)
    b = []
    b_number = 0
    layer = 1
    if deep == 1:
        return a     # all the section are section1,the no deal

    for i in dics:
        if int(i['section']) == 1:
            b_number += 1

    for i in range(b_number):
        b.append([])  # construct the structure
    # print(b_number)
    b_number = 0

    b = function3(a, layer) # make section1 struture

    for i in range(len(b)):
        if isinstance(b[i], list):
            # print("ercichuli",b[i])
            temp_array = [b[i][0].get('section' + str(layer))]
            for j in range(len(b[i])):
                # print(b[i][j])
                if int(str(b[i][j])[9]) > layer:
                    temp_array.append(b[i][j])
            if temp_array != []:
                dic = {'section' + str(layer): temp_array}
                b[i] = dic

    position = 0
    count = 0
    sec2_number = 0
    n = 0
    copy = []
    if deep == 3:
        for i in range(len(b)):
            if isinstance(b[i],dict):
                # print("yici",b[i])  # 一级标题
                if isinstance(b[i].get('section' + str(1)),list):
                    temp = b[i].get('section' + str(1))
                    copy = []
                    for j in range(len(temp)):
                        if temp[j].get('section' + str(2)) != None:
                            sec2_number +=1
                    for c in range(sec2_number):
                        copy.append({})
                    sec2_number = 0
                    n = 0
                    temp_array = []
                    for j in range(len(temp)):
                        if count != 0:
                            count -= 1
                            continue
                        if isinstance(temp[j],dict):
                            # print(j,temp[j])
                            if temp[j].get('section' + str(2)) != None:
                                position = j
                                temp_array = [temp[j].get('section' + str(2))]
                                copy[n] = temp[j]
                                n += 1
                            elif temp[j].get('section' + str(3)) != None:
                                count = 0
                                for k in range(j,len(temp)):
                                    if temp[k].get('section' + str(3)) != None:
                                        count += 1
                                        temp_array.append(temp[k])
                                    else:
                                        break
                                dic = {"section"+str(2):temp_array}
                                # print("count",count,j)
                                count -= 1
                                copy[n-1] = dic
                                # print(dic)

                    for f in copy:
                        # print(f)
                        temp_array.append(f)
                    b[i] = {'section'+str(1):temp_array}
                    # print("temp_array",temp_array)
    # print(b)
    return b

class OutlineException(Exception):
    '''自定义的异常类'''

    def __init__(self, outline_list):
        self.outline_list = outline_list

    def __str__(self):
        if self.outline_list != []:
            return "have outline!"
        else:
            return "not have outline, skip the file"

def get_pdf_text(pdf_file):
    global dics, paper  # paper: every line text list , dics: title list

    pdf_name = str(pdf_file).split('\\')[1].split('.pdf')[0]

    outline_json_file = open('structure/' + pdf_name + '_outline.json', 'w', encoding='utf-8')  # title json file
    file = open('structure/' + pdf_name + '_text.json', 'w', encoding='utf-8')  # text to json file

    # extract original outline
    doc = fitz.open(pdf_file)
    outline_list = doc.get_toc()  # 检索文档中存在的文本大纲 , 返回列表 [layer,title,page]
    pdf_name = str(pdf_file).split('\\')[1].split('.pdf')[0]
    # print("目录: ")
    for i in outline_list:
        dic = {
            "section": str(i[0]), "title": str(i[1]), "page": str(i[2])
        }
        dics.append(dic)
        # print(dic)

    # extract metadata (from the front 3 pages)
    page0 = doc.load_page(0)
    page1 = doc.load_page(1)
    page2 = doc.load_page(2)

    ## extract paper title through the bigest font size
    pagetext0_size = page0.get_text("xml")  # xml: font size
    # print(pagetext0)
    page0_size = str(pagetext0_size).split('\n')
    paper_titles = []  # font标签下的文字列表
    paper_title = ""
    sizes = []  # 字体大小列表
    for i in range(len(page0_size)):
        # print(page0_size[i])
        if page0_size[i].find('" size="') != -1:
            if paper_title != "":
                paper_titles.append(paper_title)
            paper_title = ""
            position = page0_size[i].find('" size="')
            size = filter(page0_size[i][position + 8:])
            sizes.append(size)
            # print(size)
        elif page0_size[i].find(' c="') != -1:
            position = page0_size[i].find(' c="')
            paper_title += page0_size[i][position + 4:position + 5]

    size_sort = list(set(sizes))
    size_sort.sort(reverse=True)  # 字体大小，从大到小的列表
    # print(size_sort)
    n = 0
    title_temp_part = ""  # 第一行标题

    for i in range(len(paper_titles)):
        if sizes[i] == size_sort[n] and sizes[i + 1] != size_sort[n] and len(paper_titles[i]) <= 25:
            n += 1
            # print(paper_titles[i],"太小了,查找下一个长度")
            i = 0
        if sizes[i] == size_sort[n] and sizes[i + 1] == size_sort[n] and len(paper_titles[i]) > 25:
            title_temp_part = paper_titles[i]
            # print(paper_titles[i],"选作一部分的题目")
            break

    ## extract paper_title,author completely
    paper_title = ""
    pagetext0 = page0.get_text("blocks")  # [bbox,text,paragraph,text(0),image(1)]
    pagetext1 = page1.get_text("blocks")
    pagetext2 = page2.get_text("blocks")
    page_front = [pagetext0,pagetext1,pagetext2]
    page_texts_front = []
    for i in page_front:
        for page_info in i:
            if page_info[4].find('Keywords') == -1:
                page_texts_front.append(page_info[4].replace('\n', ' ')) # put the same paragraph in one line
            elif -1 < page_info[4].find('Keywords') <= 5:
                page_texts_front.append(page_info[4].replace('\n', ';'))
    for i in range(len(page_texts_front)):
        if page_texts_front[i].find(title_temp_part) != -1:
            paper_title = page_texts_front[i]
            break

    ## extract other metadata completely
    author = ""
    abstract = ""
    keywords = ""
    # organization = ""  不好处理，先不要了
    abstract_tag = ['ABSTRACT','Abstract','abstract','Summary','a b s t r a c t']
    for i in range(len(page_texts_front)):
        # print(page_texts_front[i])
        if page_texts_front[i].find(title_temp_part) != -1:
            author = page_texts_front[i+1]
            break
    for i in range(len(page_texts_front)):
        # print(page_texts_front[i])
        for j in abstract_tag:
            if difflib.SequenceMatcher(None, j,page_texts_front[i][:15]).quick_ratio() >= 0.9\
                    or difflib.SequenceMatcher(None, j,page_texts_front[i][:8]).quick_ratio() >= 0.9:
                if len(page_texts_front[i]) <= 20:
                    abstract = page_texts_front[i + 1]
                if len(page_texts_front[i]) >= 20:
                    abstract = page_texts_front[i]
                    # print("找到了摘要",abstract)
                continue
        if difflib.SequenceMatcher(None, 'Keywords',page_texts_front[i][:8]).quick_ratio() >= 0.9:
            if len(page_texts_front[i]) <= 20:
                keywords = page_texts_front[i+1]
            if len(page_texts_front[i]) >= 20:
                keywords = page_texts_front[i].replace('Keywords:','')
            # print("找到了keywords",keywords)
            continue
    ### put metadata in dict
    metadata = []
    body_text = []
    metadata.append({'Paper_title':paper_title})
    metadata.append({'Author': author})
    metadata.append({'Abstract': abstract})
    metadata.append({'Keywords': keywords})

    # extract image
    image_dicts = []
    save_path_root = Path('pdf_parser_results')
    dir_name = os.path.join(save_path_root, pdf_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    save_path_root = dir_name  # 按文件名分出单个目录

    for current_page in range(len(doc)):  # every page
        count = 0
        image_caption = []
        for image in doc.get_page_images(current_page):  # every image for the page
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
    count = 0

    ### deal with the outline again (when the paper title is the section 1, make the section2 to 1,3 to 2)
    if difflib.SequenceMatcher(None, dics[0]["title"], paper_title).quick_ratio() >= 0.9:
        dics.pop(0)
        for i in dics:
            i["section"] = str(int(i["section"]) - 1)
    for j in abstract_tag:
        if difflib.SequenceMatcher(None, j, dics[0]['title']).quick_ratio() >= 0.9:
            dics.pop(0)
            break
    outline_content = json.dumps(dics, indent=2, ensure_ascii=False)
    outline_json_file.write(outline_content)


    # extract body text(start with section1)
    first_section = dics[0]['title']
    first_section_page = int(dics[0]['page'])

    ## find the start line,and body text original
    texts_temp = []  # 从section[0] 开始的所有text
    original_texts = []
    for current_page in range(len(doc)):
        page = doc.load_page(current_page)
        pagetext = page.get_text("blocks")
        for page_info in pagetext:  # every paragraph in the page
            i = str(page_info).find('<image:')
            j = str(page_info).find('\'TABLE')
            if i != -1:
                original_texts.append("<image:"+image_dicts[count]['image_name'])
                count += 1
            elif j != -1:
                original_texts.append("<TABLE:")
            else:
                original_texts.append(page_info[4].replace('\n',''))

    tag = 0
    for i in original_texts:
        # print(i)
        if i[:len(first_section)+3].find(first_section) != -1\
                or difflib.SequenceMatcher(None,first_section,i[:len(first_section)+3]).quick_ratio() >= 0.9:
            # print("找到了位置", first_section, i)
            tag = 1
        if tag == 1:
            texts_temp.append(i)

    for i in texts_temp:
        body_text.append(i)
    ## deal with figure_title, section, paragraph
    ### deal with the figure_title, figure to dictionary
    for i in range(len(texts_temp)):
        if texts_temp[i][:10].find('<image') != -1 and i != len(texts_temp)-1:
            # print(texts_temp[i][7:])
            for j in image_dicts:
                if j['image_name'] == texts_temp[i][7:]:
                    texts_temp[i] = str({'Figure':texts_temp[i][7:]})
                    body_text[i] = {'Figure':texts_temp[i][7:]}
                    texts_temp[i+1] = str({'Figure_title':j['image_title']})
                    body_text[i+1] = {'Figure_title': j['image_title']}
                    # print(texts_temp[i],texts_temp[i+1])


    ### deal with the section position and dictionary
    position = -1
    for i in dics:
        for j in range(len(texts_temp)):
            if texts_temp[j][:len(i['title'])+5].find(i['title']) != -1:
                texts_temp[j] = str({'section' + str(i['section']): {'section_title':i['title']}})
                body_text[j] = {'section' + str(i['section']): {'section_title': i['title']}}
                position = j
                break
            elif difflib.SequenceMatcher(None,i['title'],texts_temp[j][:len(i['title'])+5]).quick_ratio() >= 0.8 and position < j:
                texts_temp[j] = str({'section' + str(i['section']): {'section_title':i['title']}})
                body_text[j] = {'section' + str(i['section']): {'section_title': i['title']}}
                position = j
                break

    ### deal with the paragraph dictionary
    paragraph = 1
    tag = ["{'section","{'Figure':","{'Figure_title':"]
    for i in range(len(texts_temp)):
        # print(texts_temp[i])
        if texts_temp[i].find("{'section") != -1:
            paragraph = 1
        elif texts_temp[i].find(tag[1]) != -1 or texts_temp[i].find(tag[2]) != -1:
            paragraph = paragraph
        else:
            body_text[i] = {'para' + str(paragraph): texts_temp[i]}
            texts_temp[i] = str({'para'+str(paragraph):texts_temp[i]})
            paragraph += 1

    body_text_result = []
    count = 0
    tag = 0

    # make section layer
    ## put the section content in the section list
    for i in range(len(texts_temp)):
        temp = []
        x = 0
        if texts_temp[i].find("{'section") != -1:
            body_text_result.append(body_text[i])
            tag = 1
            for k in range(3):
                if body_text[i].get('section'+str(k+1)) != None:
                    x = k+1
            for j in range(i+1,len(texts_temp)):
                if texts_temp[j].find("{'section") == -1:
                    temp.append(body_text[j])
                    # print(body_text_result[count]['section'+str(x)])
                else:
                    break
        if temp != []:
            dic_array = [body_text[i].get('section'+str(x))]
            for i in temp:
                dic_array.append(i)
            # print(dic_array)
            body_text_result[count] = {'section'+str(x):dic_array}
            count += 1
            tag = 0
        elif tag ==1:
            body_text_result[count] = body_text[i]
            count += 1

    ### make layer
    max = 0
    for i in dics:
        if max < int(i["section"]):
            max = int(i["section"])
    print("max deep: ",max)

    body_text = make_layer(body_text_result,max,dics)

    text_json = [{'medatada': metadata}, {'body_text': body_text}]
    file_content = json.dumps(text_json, indent=2, ensure_ascii=False)
    file.write(file_content)

    dics = []
    paper = []

if __name__ == '__main__':
    pdf_file_path = Path('paper')
    for pdf_file in pdf_file_path.glob("*.pdf"):
        doc = fitz.open(pdf_file)
        outline_list = doc.get_toc()  # 获取目录
        pdf_name = str(pdf_file).split('\\')[1].split('.pdf')[0]
        print(pdf_name)

        try:
            if outline_list != []:
                exc = OutlineException(outline_list)
                print(exc)
                # get_pdf_image(pdf_file)  # extract image and caption
                get_pdf_text(pdf_file)  # extract all text in paper to json
            else:
                raise OutlineException(outline_list)
        except OutlineException as e:
            print(e)

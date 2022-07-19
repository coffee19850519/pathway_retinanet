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

def image_profile_2(page):  # return the image caption of a page
    image_caption = []
    pagetext = page.get_text("blocks")  # blocks
    for page_info_index in range(len(pagetext)):
        i = str(pagetext[page_info_index]).find('<image:')  # find the image tag position
        if i != -1 and page_info_index < len(pagetext)-1:
            image_caption.append(pagetext[page_info_index + 1][4])
        elif i != -1 and page_info_index == len(pagetext)-1:
            image_caption.append('-')
    return image_caption

def get_pdf_image(pdf_file): # extract image and caption,put the information in json

    doc = fitz.open(pdf_file)
    print("paper:",str(pdf_file),",number of pages: %i" % doc.page_count)

    pdf_name = str(pdf_file).split('\\')[1]
    pdf_name = pdf_name.split('.pdf')[0]
    save_path_root = Path('pdf_parser_results')
    dir_name = os.path.join(save_path_root,pdf_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    save_path_root = dir_name  # 按文件名分出单个目录

    for current_page in range(len(doc)):  # every page
        count = 0
        for image in doc.get_page_images(current_page): # every image for the page
            page = doc.load_page(current_page)

            xref = image[0]
            pix = fitz.Pixmap(doc, xref)
            save_path = os.path.join(save_path_root,"%s_page%s_%s.jpg" % (pdf_name,current_page, xref))
            if pix.n < 5:        # this is GRAY or RGB
                pix.pil_save(save_path)
            else:                # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.pil_save(save_path)
                pix1 = None
            pix = None

            img_count = len(doc.get_page_images(current_page)) # image count of the page
            # print("page:  ",current_page)
            image_caption = image_profile_2(page)  # extract the next paragraph as the image caption
            image_caption[count] = image_caption[count].replace('\n',' ')
            dic = {
                'Figure':pdf_name+"_page"+str(current_page)+"_"+str(xref)+".jpg" ,
                'Figure_title':image_caption[count]
            }
            outline_json_file = open(str(save_path_root)+"/%s_page%s_%s.json" % (pdf_name,current_page, xref)
                , 'w', encoding='utf-8')
            outline_content = json.dumps(dic, indent=2, sort_keys=True, ensure_ascii=False)
            outline_json_file.write(outline_content)
            count = + 1

def func(L): # return the list deepest deep
    if type(L) is not list:
        return 0
    k = 1
    while any([type(i) is list for i in L]):
        k += 1
        L = [i for i in L if type(i) is list]
        L = [j for i in L for j in i]
    return k

def get_file_name(file_path, endswitch): # get the certain kind of files ,then return the file name in list(not include suffix)
    file_list = []
    for name in os.listdir(file_path):
        if name.endswith(endswitch):
            file_list.append(name)
    return file_list

def check_deep(i,text_outline_list):  # 查找嵌套列表各个元素的所在深度（标题级别）
    global dics
    for message in text_outline_list:
        if isinstance(message,dict):
            dic = {
                "section":"section:"+str(i),"title":message["/Title"]
            }
            dics.append(dic)
        else:
            check_deep(i+1,message)

def filter(string):
    x = ''
    for i in string:
        if i.isdigit():
            x += i
    return int(x)

def get_pdf_strcuture(pdf_file): # extract all the text in paper to json

    global dics,paper

    pdf = pdf_read(pdf_file, 'rb')
    pdf_name = str(pdf_file).split('\\')[1]
    pdf_name = pdf_name.split('.pdf')[0]
    print(pdf_name)

    outline_json_file = open('structure/' + pdf_name + '_outline.json', 'w', encoding='utf-8')
    file = open('structure/' + pdf_name + '_text.json', 'w', encoding='utf-8')

    # 检索文档中存在的文本大纲 , 返回的对象是一个嵌套的列表
    text_outline_list = pdf.getOutlines() # 测试过无大纲pdf,返回的是空list
    check_deep(1,text_outline_list)  # 获取标题和对应级别的字典,适用于任何层数的目录结构（第一个参数后期可修改为0）
    deep = func(text_outline_list) # 获取嵌套list的最大深度

    outline_content = json.dumps(dics,indent=2, sort_keys=True, ensure_ascii=False)
    outline_json_file.write(outline_content)

    # 获取图片的名字
    img_names = get_file_name('pdf_parser_results/'+pdf_name, 'jpg')
    # print(len(img_names))
    img_n = 0  # image gross
    table_n = 0  # table gross

    # 开始处理text
    pdf_document = fitz.open(pdf_file)
    for current_page in range(len(pdf_document)):
        page = pdf_document.load_page(current_page)
        pagetext = page.get_text("blocks")
        for page_info in pagetext:  # every paragraph in the page
            i = str(page_info).find('<image:')
            j = str(page_info).find('\'TABLE')
            if i != -1:
                new = "<image :"+img_names[img_n]
                paper.append(new)
                img_n += 1
            elif j != -1:
                new = "<table :" + str(table_n+1)
                paper.append(new)
                table_n += 1
            else:
                new = str(page_info[4]).replace('\n', ' ')
                paper.append(new)

    # 处理标题
    temp = []
    for dic in dics:
        for i in range(len(paper)):
            # print(dic['title'])
            if str(paper[i]).find(dic['title']) == 0 \
                    or (difflib.SequenceMatcher(None, str(paper[i])[0:len(dic['title'])],dic['title']).quick_ratio() >= 0.95)\
                    or (difflib.SequenceMatcher(None, str(paper[i])[4:len(dic['title'])+5],dic['title']).quick_ratio() >= 0.95):
                temp.append(i)
                break

    # print(len(dics))
    for i in range(len(temp)-1 ,0, -1):  # 分离标题
        # print('1',paper[temp[i]],i)
        # print('2', len(dics[i]))
        if len(paper[temp[i]])-len(dics[i]['title']) >= 2:
            paper[temp[i]] = paper[temp[i]][len(dics[i]['title'])+1:]
            paper.insert(temp[i],dics[i]['title'])

    for dic in dics: # 用dict换原来的标题
        for i in range(len(paper)):
            if str(paper[i]).find(dic['title']) == 0:
                dict = {
                    dic["section"]:dic["title"]
                }
                paper[i] = str(dict)

    paper_copy = paper
    for i in range(len(paper)):
        if len(paper[i]) >= 5:
            paper_copy.append(paper[i])
    paper = paper_copy
    paper = paper[0:int(len(paper) / 2)]

    # 处理其他问题：都换成dict
    paragraph = 1
    for i in range(len(paper)):
        # 如果是图片，换成image,image_name
        if str(paper[i]).find("<image :") != -1:
            dict = {"Figure": paper[i][8:]}
            paper[i] = str(dict)
        # elif str(paper[i]).find("<table: ") == 0:
        #     dict = {"table": paper[i][8:]}
        #     paper[i] = str(dict)
        elif str(paper[i]).find("{'section") == 0:
            paragraph = 1
        else:
            a = str(paper[i])[0:6].find('Fig.')
            b = str(paper[i])[0:8].find('Figure')
            c = str(paper[i])[0:8].find('FIGURE')
            if str(paper[i-1]).find("Figure") != -1 and str(paper[i-1]).find("Figure_title") == -1 and (a != -1 or b != -1 or c != -1):
                dict = {"Figure_title": str(paper[i])}
                paper[i] = str(dict)
                print(paper[i])
            else:
                dict = {"paragraph"+str(paragraph):paper[i]}
                paper[i] = str(dict)
                paragraph += 1

    # str --> dict
    for i in range(len(paper)):
        # print(paper[i])
        paper[i] = ast.literal_eval(paper[i])  # 转化为dict

    inden = 5
    file.write('[')
    for i in range(len(paper)):
        if str(paper[i].keys()).find("section:") != -1:
            inden = filter(str(paper[i].keys()))*5+5
        paper_content = json.dumps(paper[i],indent=inden, sort_keys=True, ensure_ascii=False)
        file.write(paper_content)
        if i != len(paper)-1:
            file.write(',\n')
    file.write(']')

    dics = []
    paper = []

if __name__ == '__main__':
    pdf_file_path = Path('paper')
    for pdf_file in pdf_file_path.glob("*.pdf"):
        pdf = pdf_read(pdf_file, 'rb')
        text_outline_list = pdf.getOutlines()
        if text_outline_list != []:
            get_pdf_image(pdf_file)  # extract image and caption
            get_pdf_strcuture(pdf_file)  # extract all text in paper to json

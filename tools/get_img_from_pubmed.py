import urllib
import io
from bs4 import BeautifulSoup
#from urllib import request
from urllib.parse import unquote




# html_doc = "https://www.ncbi.nlm.nih.gov/pmc/?term=Non-small+lung+cancer+pathway+proliferation&report=imagesdocsum"
#
# req = urllib.request.Request(html_doc)
# webpage = urllib.request.urlopen(req)
# html = webpage.read()
#soup = BeautifulSoup(html, 'html.parser')
path = '/home/fei/Desktop/weiwei/pathway_style_classification/3.htm'
#htmlfile = open(path, 'r', encoding='utf-8')
htmlfile = io.open(path, 'r', encoding='utf-8')
html = htmlfile.read()

soup = BeautifulSoup(html, 'lxml')

# pathhead='https://www.ncbi.nlm.nih.gov'
pathhead='https://proteinlounge.com/images/pathways/'
pathlist=[]
articlelist=[]
img=[]
img_name=[]
article_src=[]

img_srcs=soup.findAll("img")
article_srcs=soup.findAll("a",{'class':'rprt_img figpopup imagepopup'})


for img in img_srcs:

    image=img.get('src')
    try:
        if image:
            image_name=image.split('/')[2].replace(' ','%20')
            # image_name_encode = unquote(image_name, 'utf-8')
            path = pathhead + image_name
            # path=pathhead+image
            pathlist.append(path)
            img_name.append(image)
        else:

            continue
        print(path)
    except:
        continue

# for article_src in article_srcs:
#     article_src = article_src.get('image-link')
#     articlelist.append(article_src)

idx = 0
for imgPath in pathlist:


#    a='/Users/weiwei/Documents/PGen Workflow/img_download/'+ articlelist[idx].replace('/','_')+".jpg"
    a='/home/fei/Desktop/weiwei/pathway_style_classification/protein_lounge_image/' + img_name[idx].replace('/', '_')
    try:
        #f = open('/Users/weiwei/Documents/PGen Workflow/img_download/'+ articlelist[idx].replace('/','_')+".jpg", 'wb')
        # f = open('/Users/weiwei/Documents/Pathway/img_download/Cancer signaling pathway/' + img_name[idx].replace('/', '_'),
        #          'wb')
        #f.write((urllib.urlopen(imgPath)).read())
        #f.write((urllib.request.urlopen(imgPath)).read())
        # f=open(a,'wb')
        # f.write((urllib.urlopen(imgPath)).read())
        # print(imgPath)
        # f.close()
        request = urllib.request.Request(imgPath)
        response = urllib.request.urlopen(request)
        get_img = response.read()
        with open(a, 'wb') as fp:
            fp.write(get_img)

    except Exception as e:
        print(imgPath+" error")
    idx += 1

del img_srcs,image



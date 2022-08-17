from bs4 import BeautifulSoup
import os
import requests
from lxml import etree
import pandas as pd

term = 'non-small cell lung cancer'  # search content

class NCBISpider:
    def __init__(self):
        self.headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"}
        self.start_url = "https://pubmed.ncbi.nlm.nih.gov/?term="+term+"&page=1"

    def url_lists(self, total_num):
        url_lists = []
        for i in range(total_num):
            url = "https://pubmed.ncbi.nlm.nih.gov/?term="+term+"&page={}".format(i)  # 需要判断i是否需要字符串还是数字
            print(url)
            url_lists.append(url)
        return url_lists

    def parase_url(self, url):  # 爬取内容
        print(url)
        response = requests.get(url, headers=self.headers, timeout=8)
        return response.content.decode()

    def save_csv_title(self):  # 先保存headers，也是就title
        columns = ["title","PMID", "paper_citation","DOI", "author", "Abstract", "paper_url"]
        title_csv = pd.DataFrame(columns=columns)
        title_csv.to_csv('paper_information.csv', mode="a", index=False, header=1, encoding="utf-8")

    def get_content(self, html):  # 获取相关内容

        nodes = etree.HTML(html)
        articel = nodes.xpath('//div[@class="search-results-chunk results-chunk"]/article')
        # print(articel)
        ret = []
        for art in articel:
            # pass
            item = {}
            # 实现标题的去换行、空字符和连接
            item["title"] = art.xpath(
                './div[@class="docsum-wrap"]/div[@class="docsum-content"]/a[@class="docsum-title"]//text()')
            item["title"] = [i.replace("\n", "").strip() for i in item["title"]]
            item["title"] = [''.join(item["title"])]

            # PMID
            item["PMID"] = art.xpath('./div[@class="docsum-wrap"]//span[@class="citation-part"]/span/text()')

            # 期刊相关信息
            item["paper_citation"] = art.xpath(
                './div[@class="docsum-wrap"]//span[@class="docsum-journal-citation full-journal-citation"]/text()')

            # DOI
            if str(item["paper_citation"]).find("doi: ") != -1:
                item["DOI"] = str(item["paper_citation"]).split("doi: ")[1].replace("\'","").replace("]","").replace(" ","")
                if str(item["DOI"]).find("Epub") != -1:
                    item["DOI"] = str(item["DOI"]).split("Epub")[0]
                item["DOI"] = str(item["DOI"])[:len(item["DOI"]) - 1]
            else:
                item["DOI"] = "-"

            # 作者
            item["author"] = art.xpath('./div[@class="docsum-wrap"]//span[@class="docsum-authors full-authors"]/text()')
            # 摘要
            item["Abstract"] = art.xpath('./div[@class="docsum-wrap"]//div[@class="full-view-snippet"]//text()')
            item["Abstract"] = [i.replace("\n", "").strip() for i in item["Abstract"]]
            item["Abstract"] = [''.join(item["Abstract"])]

            # 文章地址
            item["url"] = art.xpath('./div[@class="docsum-wrap"]//div[@class="share"]/button/@data-permalink-url')
            ret.append(item)


        self.save_content(ret)
        print("保存好了！！！")

    def save_content(self, ret):  # 保存到指定内容
        pf = pd.DataFrame(ret)
        pf.to_csv('paper_information.csv', mode="a", index=False, header=0, encoding="utf-8")

    def run(self,total_num):  # 实现主要逻辑
        self.save_csv_title()
        start_html = self.parase_url(self.start_url)
        # total_num = re.findall('totalResults: parseInt\("(.*?)", 10\)', start_html, re.S)[0]
        # total_num = int(total_num)
        print(type(total_num))

        # 1、构造url列表
        url_lists = self.url_lists(total_num)
        for url in url_lists:
            # 2、requests爬虫
            htmls = self.parase_url(url)
            self.get_content(htmls)

def download_pdf(path):
    if os.path.exists(path) == False:
        os.mkdir(path)  # 20210607更新，创建保存下载文章的文件夹
    head = { \
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36' \
        }  # 20210607更新，防止HTTP403错误
    doi_f = pd.read_csv("paper_information.csv")  # 存放DOI码的文件中，每行存放一个文献的DOI码，完毕须换行（最后一个也须换行！）
    for i in range(doi_f.shape[0]):
        doi = doi_f.loc[i][3]
        print("[" + str(i + 1) + "]")
        if doi == "-":
            print("no doi,skip")
            continue
        else:
            url = "https://www.sci-hub.ren/" + doi + "#"  # 20210515更新：现在换成这个sci hub检索地址
            try:
                download_url = ""  # 20211111更新
                r = requests.get(url, headers=head)
                r.raise_for_status()
                r.encoding = r.apparent_encoding
                soup = BeautifulSoup(r.text, "html.parser")
                # download_url = "https:" + str(soup.div.ul.find(href="#")).split("href='")[-1].split(".pdf")[0] + ".pdf" #寻找出存放该文献的pdf资源地址（该检索已失效）
                if soup.iframe == None:  # 20211111更新
                    download_url = "https:" + soup.embed.attrs["src"]  # 20211111更新
                else:
                    download_url = soup.iframe.attrs["src"]  # 20210515更新
                print(doi + " is downloading...\n  --The download url is: " + download_url)
                download_r = requests.get(download_url, headers=head)
                download_r.raise_for_status()
                path_new = os.path.join(path, doi.replace("/", "_") + ".pdf")
                with open(path_new, "wb+") as temp:
                    temp.write(download_r.content)
            except:
                with open("error.txt", "a+") as error:
                    error.write(doi + " occurs error!\n")
                    if "https://" in download_url:
                        error.write(" --The download url is: " + download_url + "\n\n")
            else:
                download_url = ""  # 20210801更新
                print(doi + " download successfully.\n")

def download(path,total_num):

    # get DOI(article information) from pubmed
    ncbi_spider = NCBISpider()
    total_num = int(total_num/10)+1
    ncbi_spider.run(total_num)

    # download pdf
    download_pdf(path)

# if __name__ == "__main__":
#
#     # get DOI(article information) from pubmed
#     ncbi_spider = NCBISpider()
#     ncbi_spider.run()
#
#     # download pdf
#     download_pdf("paper")




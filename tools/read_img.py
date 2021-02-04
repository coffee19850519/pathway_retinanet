import base64
import os


def encode_base64(file):
    with open(file, 'rb') as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data)
        print(type(base64_data))


        base64_str = str(base64_data, 'utf-8')
        print(base64_str)
        return base64_data


def decode_base64(base64_data):
    with open('./images/base64.jpg', 'wb') as file:
        img = base64.b64decode(base64_data)
        file.write(img)


if __name__ == '__main__':
    # img_path = '/home/fei/Desktop/weiwei/pathway_style_classification/protein_lounge_image/images_pathways_4-1BB Pathway.jpg'
    # base64_data = encode_base64(img_path)
    # decode_base64(base64_data)

    file_path='/home/fei/Desktop/weiwei/pathway_web/SkyEye/users/upload-files/2020-04-21T22:35:29.355Z/'
    with open(file_path+'input.txt','r') as file:
        with open(file_path+'input.jpg','wb') as f:
            img_data = file.read().split(',',1)[1]
            img=base64.b64decode(img_data)
            f.write(img)
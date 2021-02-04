import base64
import json
import os.path
import numpy as np
#from shape_tool import is_smallpolygon_covered_by_largeone

class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = '.json'

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.text_num = 0
        self.arrow_num = 0
        self.inhibit_num = 0
        self.gene_num = 0

        if filename is not None:
            self.load(filename)
        self.filename = filename

    def load(self, filename):
        keys = [
            'imageData',
            'imagePath',
            'lineColor',
            'fillColor',
            'shapes',  # polygonal annotations
            'flags',  # image level flags
            'imageHeight',
            'imageWidth',
        ]
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            if data['imageData'] is not None:
                image_data = base64.b64decode(data['imageData'])
            else:
                # relative path from label file to relative path from cwd
                # imagePath = os.path.join(cfg.data_dir, cfg.origin_image_dir_name,
                #                         data['imagePath'])
                # parent_path = os.path.dirname(filename)
                # imagePath = os.path.join(parent_path, data['imagePath'])
                # with open(imagePath, 'r') as f:
                #    image_data = f.read()
                image_data = None

            flags = data.get('flags')
            imagePath = data['imagePath']
            lineColor = data['lineColor']
            fillColor = data['fillColor']
            imageHeight = data['imageHeight']
            imageWidth = data['imageWidth']

            shapes = []
            for s in data['shapes']:
                shapes.append(s)
                category = s['label'].split(':')[0]
                if category == 'inhibit' and s['label'].find('*\t*\t*\t*\t*') == -1:
                    self.inhibit_num = self.inhibit_num + 1
                elif category == 'activate' and s['label'].find('*\t*\t*\t*\t*') == -1:
                    self.arrow_num = self.arrow_num + 1
                else:
                    if s['label'].find('*\t*\t*\t*\t*') == -1:
                        self.text_num = self.text_num + 1

        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = image_data
        self.lineColor = lineColor
        self.fillColor = fillColor
        self.filename = filename
        self.otherData = otherData
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth

    def save(
            self,
            filename,
            shapes,
            imagePath,
            imageHeight,
            imageWidth,
            lineColor=None,
            fillColor=None,
            otherData=None,
            flags=None,
    ):
        if otherData is None:
            otherData = {}

        if flags is None:
            flags = {}

        data = dict(
            version=None,
            flags=flags,
            shapes=shapes,
            lineColor=[0, 255, 0, 128],
            fillColor=[255, 0, 0, 128],
            imagePath=imagePath,
            imageData=None,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            data[key] = value
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    def get_all_genes(self):
        gene_names = []
        for shape in self.shapes:
            if shape['label'].find('gene:') == 0:
                _, _, gene_name = str(shape['label']).partition('gene:')
                gene_names.append(gene_name)
        return gene_names

    def get_all_relations(self):
        relations = []
        for shape in self.shapes:
            if shape['label'].find('|') >= 0 and shape['label'].find('*\t*\t*\t*\t*') == -1:
                relation = str(shape['label'])
                relations.append(relation)
        return relations

    def get_all_gene_id_and_names(self):
        gene_id_and_names = {}
        for shape in self.shapes:
            if shape['label'].find('gene:') >= 0:
                try:
                    id, _, gene_name = str(shape['label']).split(':')
                    gene_id_and_names.update({id : gene_name})
                except:
                    pass
        return gene_id_and_names

    def get_all_text(self):
        text_list = []
        for shape in self.shapes:
            if shape['label'].find('activate:') == -1 and \
                    shape['label'].find('inhibit:') == -1:
                try:
                    _, text = str(shape['label']).split(':', 1)
                except:
                    pass
                if text is not None:
                    text_list.append(text.upper())
        return text_list

    # def generate_category(self, shape):
    #     category = ''
    #     # if shape['label'].find('arrow') != -1 or shape['label'].find(
    #     #     '<activate>') != -1 or shape['label'] == 'action:':
    #     if shape['label'].split(':')[0] == 'activate':
    #         # arrow
    #         category = 'arrow'
    #     elif shape['label'].split(':')[0] == 'inhibit':
    #         category = 'nock'
    #     else:
    #         category = 'text'
    #     return category

    def get_all_boxes_for_category(self, category_name):
        text_boxes = []
        for shape in self.shapes:
            if LabelFile.get_shape_category(shape) == category_name:
                text_boxes.append(shape['points'])
        return text_boxes
        # 2019/3/6 WWW get_all_boxes_for_category

    def get_all_shapes_for_category(self, category_name):
        targets = []
        for shape in self.shapes:
            if LabelFile.get_shape_category(shape) == category_name:
                targets.append(shape)
        return targets

    def check_relationship_box(self, covered_shapes):
        only_one_relation_symbo = False
        for shape in covered_shapes:
            if LabelFile.get_shape_category(shape) != 'gene' and not only_one_relation_symbo:
                only_one_relation_symbo = True
        return False


    # def get_sub_box_in_relationship(self, current_shape):
    #     #get its involving entity names
    #     #content = shape['label'].split(':',1)[1]
    #     covered_shapes = []
    #     #check all sub objects it covers
    #     for shape in self.shapes:
    #         #if shape is covered by current_shape
    #         if is_smallpolygon_covered_by_largeone(current_shape, shape):
    #             covered_shapes.append(shape)
    #     if len(covered_shapes) > 2 and self.check_relationship_box(covered_shapes):
    #         return covered_shapes
    #     else:
    #         return None


    def get_duplicated_genes(self):
        duplicated_genes = {}
        gene_names = self.get_all_genes()
        gene_set = set(gene_names)
        for gene in gene_set:
           gene_count = gene_names.count(gene)
           if gene_count > 1:
                duplicated_genes.update({gene : self.get_duplicated_gene_IDs(gene)})
        return duplicated_genes

    def get_duplicated_gene_IDs(self, query_gene_name):
        gene_shapes = self.get_all_shapes_for_category('gene')
        IDs = []
        for gene_shape in gene_shapes:
            try:
                if query_gene_name == gene_shape['label'].split(':', 1)[1]:
                    IDs.append(gene_shape['ID'])
            except:
                continue
        del gene_shapes
        return IDs

    def get_gene_ID_by_name(self, query_gene_name):
        gene_shapes = self.get_all_shapes_for_category('gene')
        for gene_shape in gene_shapes:
            if gene_shape['label'].split(':', 1)[1] == query_gene_name:
                del gene_shapes
                return gene_shape['ID']
        del gene_shapes
        return None

    def export_predict_correct_txt(self):
        predict_results = []
        correct_results = []
        for shape in self.shapes:
            category, content = shape['label'].split(':', 1)
            coords = np.array (shape['points'], np.int)
            if category == 'gene':
                predict_temp = 'text\t'
                correct_temp = '0@' + content + '@'
                if len(coords) == 2:
                    # convert it into 8-dimension
                    predict_temp += str(coords[0][0]) + ',' + str(coords[0][1]) + ','
                    predict_temp += str(coords[1][0]) + ',' + str(coords[0][1]) + ','
                    predict_temp += str(coords[1][0]) + ',' + str(coords[1][1]) + ','
                    predict_temp += str(coords[0][0]) + ',' + str(coords[1][1]) + '\n'
                    correct_temp += str([[coords[0][0], coords[0][1]], [coords[1][0], coords[0][1]],[coords[1][0], coords[1][1]], [coords[0][0], coords[1][1]]]) + '\n'
                else:
                    predict_temp += str(coords[0][0]) + ',' + str(coords[0][1]) + ','
                    predict_temp += str(coords[1][0]) + ',' + str(coords[1][1]) + ','
                    predict_temp += str(coords[2][0]) + ',' + str(coords[2][1]) + ','
                    predict_temp += str(coords[3][0]) + ',' + str(coords[3][1]) + '\n'
                    correct_temp += str(shape['points']) + '\n'
                predict_results.append(predict_temp)
                correct_results.append(correct_temp)
            elif category == 'inhibit':
                predict_temp = 'nock\t'
                if len(coords) == 2:
                    # convert it into 8-dimension
                    predict_temp += str(coords[0][0]) + ',' + str(coords[0][1]) + ','
                    predict_temp += str(coords[1][0]) + ',' + str(coords[0][1]) + ','
                    predict_temp += str(coords[1][0]) + ',' + str(coords[1][1]) + ','
                    predict_temp += str(coords[0][0]) + ',' + str(coords[1][1]) + '\n'
                    #correct_temp += str([[coords[0][0], coords[0][1]], [coords[1][0], coords[0][1]],[coords[1][0], coords[1][1]], [coords[0][0], coords[1][1]]]) + '\n'
                else:
                    predict_temp += str(coords[0][0]) + ',' + str(coords[0][1]) + ','
                    predict_temp += str(coords[1][0]) + ',' + str(coords[1][1]) + ','
                    predict_temp += str(coords[2][0]) + ',' + str(coords[2][1]) + ','
                    predict_temp += str(coords[3][0]) + ',' + str(coords[3][1]) + '\n'
                    #correct_temp += str(shape['points']) + '\n'
                predict_results.append(predict_temp)
            elif category == 'activate':
                predict_temp = 'arrow\t'
                if len(coords) == 2:
                    # convert it into 8-dimension
                    predict_temp += str(coords[0][0]) + ',' + str(coords[0][1]) + ','
                    predict_temp += str(coords[1][0]) + ',' + str(coords[0][1]) + ','
                    predict_temp += str(coords[1][0]) + ',' + str(coords[1][1]) + ','
                    predict_temp += str(coords[0][0]) + ',' + str(coords[1][1]) + '\n'
                else:
                    predict_temp += str(coords[0][0]) + ',' + str(coords[0][1]) + ','
                    predict_temp += str(coords[1][0]) + ',' + str(coords[1][1]) + ','
                    predict_temp += str(coords[2][0]) + ',' + str(coords[2][1]) + ','
                    predict_temp += str(coords[3][0]) + ',' + str(coords[3][1]) + '\n'
                predict_results.append(predict_temp)
            else:
                continue

            image_name, image_ext = os.path.splitext(self.imagePath)
        with open(os.path.join(r'C:\Users\coffe\Desktop\test\export', image_name + '_0.6_predict.txt'),'w') as predict_fp:
            predict_fp.writelines(predict_results)
        with open(os.path.join(r'C:\Users\coffe\Desktop\test\export', image_name + '_0.6_correct.txt'),'w') as correct_fp:
            correct_fp.writelines(correct_results)

    def get_max_shape_id(self):
        max_id = 0
        for shape in self.shapes:
            try:
                current_id = int(LabelFile.get_shape_index(shape))
            except:
                #if cannot get numeric id, maybe lack of saving index to the 'label' part
                # then try next shapr
                continue
            if  current_id > max_id:
                max_id = current_id
        return max_id

    def get_index_with_id(self, id):
        for index, shape in enumerate(self.shapes):
            if  LabelFile.get_shape_index(shape) == int(id):
                return index
        raise Exception('cannot find id:' +  str(id) + ' shape')

    def reset(self):
        new_shapes = []
        for i, shape in enumerate(self.shapes):
            if not str(str(shape['label'])).split(':',1)[0].isdigit():
                shape['label'] = str(i) + ':' + str(shape['label'])

            shape['line_color'] = None
            shape['fill_color'] = None
            #only keep element shapes
            if LabelFile.get_shape_category(shape) == 'gene' or \
               LabelFile.get_shape_category(shape) == 'activate' or \
               LabelFile.get_shape_category(shape) == 'inhibit':
                new_shapes.append(shape)
        self.shapes = new_shapes

    # def generate_category_id(self, shape, category_list):
    #     try:
    #         return list(category_list).index(LabelFile.get_shape_category(shape))
    #     except:
    #         raise Exception('the shape '+ shape['label'] + ' in file ' + self.filename + ' has invalid category')

    def generate_category_id(self, shape, category_list):
        try:
            if LabelFile.get_shape_category(shape)== 'gene':
                return 1
            elif LabelFile.get_shape_category(shape)== 'activate' :
                return 0
            elif LabelFile.get_shape_category(shape)== 'inhibit':
                return 2

            else:
                raise Exception('the shape ' + shape['label'] + ' in file ' + self.filename + ' has invalid category')
        except:
            raise Exception('the shape '+ shape['label'] + ' in file ' + self.filename + ' has invalid category')


    @staticmethod
    def isLabelFile(filename):
        return os.path.splitext(filename)[1].lower() == LabelFile.suffix

    @staticmethod
    def get_shape_index(shape):
        return int(shape['label'].split(':', 2)[0])

    @staticmethod
    def get_shape_category(shape):
        return shape['label'].split(':', 2)[1]

    @staticmethod
    def get_shape_annotation(shape):
        return shape['label'].split(':', 2)[2]


if __name__ == '__main__':
    # ground_truth_folder = r'C:\Users\LSC-110\Desktop\ground_truth'
    # all_genes_in_gt = []
    # for json_file in os.listdir(ground_truth_folder):
    #     if os.path.splitext(json_file)[-1] == '.json':
    #         json_data = LabelFile(os.path.join(ground_truth_folder, json_file))
    #         all_genes_in_gt.extend(json_data.get_all_text())
    #         del json_data
    # with open(os.path.join(ground_truth_folder, 'text_list.txt'), 'w') as gene_fp:
    #     gene_fp.write('\n'.join(all_genes_in_gt))
    ground_truth_folder = r'C:\Users\coffe\Desktop\test\images'
    for json_file in os.listdir(ground_truth_folder):
        if os.path.splitext(json_file)[-1] != '.json':
            continue
        else:
            json_data = LabelFile(os.path.join(ground_truth_folder, json_file))
            json_data.export_predict_correct_txt()


    # with open(os.path.join(ground_truth_folder, 'gene_list.txt'),'w') as
    # gene_fp:

# end of file

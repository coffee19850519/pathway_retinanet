import os

import cv2
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
from tools.label_file import LabelFile

ia.seed(1017)


def apply_transform_on_an_image(image, bbox=None):


    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        # iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        # iaa.Sometimes(0.5,
        #               iaa.GaussianBlur(sigma=iap.Uniform(0.0, 1.0))
        #               ),

        iaa.ContrastNormalization(
            iap.Choice(
                [1.0, 1.5, 3.0],
                p=[0.5, 0.3, 0.2]
            )
        ),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply(
            iap.Positive(iap.Normal(0.0, 0.1)) + 1.0, per_channel=0.2),
        # Apply affine transformations to each image.
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        # iaa.Affine(
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #     # translate_percent=0.1,
        #     rotate=iap.Normal(-45, 45),
        #     shear=(-8, 8),
        #     order=[0, 1],
        #     cval=(0, 255),
        #     # translate_px=iap.RandomSign(iap.Poisson(3)),
        #     # translate_px=3,
        #     # mode=["constant", "edge"]
        #     mode=ia.ALL
        # ),

        # iaa.AddElementwise(
        #     iap.Discretize(
        #         (iap.Beta(0.5, 0.5) * 2 - 1.0) * 64
        #     )
        # ),
        #
        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((0, 5),
                   [
                       # Convert some images into their superpixel representation,
                       # sample between 20 and 200 superpixels per image, but do
                       # not replace all superpixels with their average, only
                       # some of them (p_replace).
                       # sometimes(
                       #     iaa.Superpixels(
                       #         p_replace=(0, 0.5),
                       #         n_segments=(1, 4)
                       #     )
                       # ),

                       # Blur each image with varying strength using
                       # gaussian blur (sigma between 0 and 3.0),
                       # average/uniform blur (kernel size between 2x2 and 7x7)
                       # median blur (kernel size between 3x3 and 11x11).
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 0.3)),
                           iaa.AverageBlur(k=(1, 2)),
                           iaa.MedianBlur(k=(1, 3)),
                       ]),

                       # iaa.OneOf([
                       #     iaa.GaussianBlur((0, 3.0)),
                       #     iaa.AverageBlur(k=(2, 7)),
                       #     iaa.MedianBlur(k=(3, 11)),
                       # ]),

                       # Sharpen each image, overlay the result with the original
                       # image using an alpha between 0 (no sharpening) and 1
                       # (full sharpening effect).
                       iaa.Sharpen(alpha=(0, 0.01), lightness=(0, 0.01)),

                       # Same as sharpen, but for an embossing effect.
                       iaa.Emboss(alpha=(0, 0.01), strength=(0, 0.01)),

                       # Search in some images either for all edges or for
                       # directed edges. These edges are then marked in a black
                       # and white image and overlayed with the original image
                       # using an alpha of 0 to 0.7.
                       sometimes(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0, 0.005)),
                           iaa.DirectedEdgeDetect(
                               alpha=(0, 0.001), direction=(0.0, 0.001)
                           ),
                       ])),

                       # Add gaussian noise to some images.
                       # In 50% of these cases, the noise is randomly sampled per
                       # channel and pixel.
                       # In the other 50% of all cases it is sampled once per
                       # pixel (i.e. brightness change).
                       # iaa.AdditiveGaussianNoise(
                       #     loc=0, scale=(0.0, 0.001 * 255), per_channel=0.5
                       # ),

                       # Either drop randomly 1 to 10% of all pixels (i.e. set
                       # them to black) or drop them on an image with 2-5% percent
                       # of the original size, leading to large dropped
                       # rectangles.
                       # iaa.OneOf([
                       #     iaa.Dropout((0, 0.05), per_channel=0.5),
                       #     iaa.CoarseDropout(
                       #         (0, 0.01), size_percent=(0.1, 0.2),
                       #         per_channel=0.2
                       #     ),
                       # ]),

                       # Invert each image's channel with 5% probability.
                       # This sets each pixel value v to 255-v.
                       iaa.Invert(0.1, per_channel=True),  # invert color channels

                       # Add a value of -10 to 10 to each pixel.
                       #iaa.Add((-40, 40), per_channel=0.5),

                       # Change brightness of images (50-150% of original value).
                       # iaa.Multiply((0.5, 1.5), per_channel=0.5),

                       # Improve or worsen the contrast of images.
                       # iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),

                       # Convert each image to grayscale and then overlay the
                       # result with the original with random alpha. I.e. remove
                       # colors with varying strengths.
                       # iaa.Grayscale(alpha=(0.0, 1.0)),

                       # In some images move pixels locally around (with random
                       # strengths).
                       # sometimes(
                       #     iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)
                       # ),

                       # In some images distort local areas with varying strength.
                       # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                   ],
                   # do all of the above augmentations in random order
                   random_order=True
                   )
    ], random_order=True)

    # Augment BBs and images.
    aug_image= seq(image=image)
    return aug_image
    # return aug_image, aug_bbox
    # return aug_image, aug_bbox.remove_out_of_image().clip_out_of_image()

if __name__ == "__main__":


    img_path = r'/home/fei/Desktop/train_data/0108/images/'
    json_path = r'/home/fei/Desktop/train_data/0108/jsons/'
    aug_img_path = r'/home/fei/Desktop/train_data/0108/aug_images/'
    aug_json_path = r'/home/fei/Desktop/train_data/0108/aug_jsons/'

    # for img in os.listdir(img_path):
    for json in os.listdir(json_path):
        if os.path.splitext(json)[1] != '.json':
            continue
        #img_name=os.path.splitext(json)[0]

        img_anns = LabelFile(os.path.join(json_path, json))
        if os.path.exists(os.path.join(img_path, img_anns.imagePath)):
            img=imageio.imread(os.path.join(img_path, img_anns.imagePath))

        # else:
        #     img = imageio.imread(os.path.join(img_path, img_name + '.png'))
        #img_anns = LabelFile(os.path.join(json_path+json))
        # bbox=[]
        # for anno in img_anns.shapes:
        #
        #     poly_points = np.array(anno['points'], np.float32).reshape((-1, 2))
        #     rect=cv2.boxPoints(cv2.minAreaRect(poly_points))
        #     rect=np.int32(rect)
        #     bbox.append(BoundingBox(x1=rect[0][0], x2=rect[2][0], y1=poly_points[0][1], y2=poly_points[2][1]))
        #     # two_points=[]
        #     # two_points.append(poly_points[0])
        #     # two_points.append(poly_points[2])
        #     # bbox.append(two_points)
        # bbs = BoundingBoxesOnImage(bbox, shape=img.shape)
            for i in range(10):
                aug_img = apply_transform_on_an_image(img)
                try:
                    imageio.imwrite(os.path.join(aug_img_path,'aug'+str(i)+img_anns.imagePath),aug_img)
                except Exception as e :
                    print(img_anns.imagePath+' '+str(i))
                    print(e)
                    continue
                img_anns.imagePath='aug'+str(i)+img_anns.imagePath

                # for i,anno in enumerate(img_anns.shapes):
                #     # try:
                #     # point_0_x = aug_bbox.bounding_boxes[i].x1.astype(np.int32)
                #     point_0_x=aug_bbox.bounding_boxes[i].x1
                #     point_0_y = aug_bbox.bounding_boxes[i].y1
                #     point_1_x = aug_bbox.bounding_boxes[i].x2
                #     point_1_y = aug_bbox.bounding_boxes[i].y1
                #     point_2_x = aug_bbox.bounding_boxes[i].x2
                #     point_2_y = aug_bbox.bounding_boxes[i].y2
                #     point_3_x = aug_bbox.bounding_boxes[i].x1
                #     point_3_y = aug_bbox.bounding_boxes[i].y2
                #     anno['points']=[]
                #     anno['points'].append([float(point_0_x),float(point_0_y)])
                #     anno['points'].append([float(point_1_x), float(point_1_y)])
                #     anno['points'].append([float(point_2_x), float(point_2_y)])
                #     anno['points'].append([float(point_3_x), float(point_3_y)])
                    # except Exception as e:
                    #     print(img_name+'  '+str(i))
                    #     print(str(e))

                img_anns.save(os.path.join(aug_json_path, 'aug'+str(i)+json),
                           img_anns.shapes,
                           img_anns.imagePath,
                           img_anns.imageHeight, img_anns.imageWidth, img_anns.lineColor, img_anns.fillColor, img_anns.otherData, img_anns.flags)


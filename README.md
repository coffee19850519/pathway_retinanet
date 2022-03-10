# pathway_retinanet
This repository contains the Pathway Team's code and files for extracting relationships from pathway diagrams.

# Dependencies
A list of python packages can be found in the requirements.txt

Our models are trained on gpu using cuda. Refer to the link below for installation guide <br>
https://docs.nvidia.com/cuda/index.html

Refer to the following link to get the correct version of torch installed for your cuda or cpu version <br>
https://pytorch.org/get-started/locally/

Installation steps for Detectron2 can be found on their github page<br>
https://github.com/facebookresearch/detectron2

# Training
To train the retinanet, you just need to specify the image directory (img_path), labels directory (json_path), and config file (args.config_file) locations in the plain_train_net.py file.

### File Structure

    .
    ├── dir_name                   # Location of training data
      ├── img                      # Folder containing training and validation images
        ├── *                      # image files (.jpg,.png,.etc)
      ├── json                     # Folder containing json labels
        ├── train_0                # Folder containing training labels
          ├── *                    # json label files (.jpg,.png,.etc)
        ├── val_0                  # Folder containing validation labels
          ├── *                    # json label files (.jpg,.png,.etc)
          
The file structure of the training data should be organized as above.


# Notes for extending provided code
To train for more classes, add to the category_list in plain_train_net.py and account for handling new labels in label_file.py 




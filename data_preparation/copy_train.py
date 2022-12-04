'''
    All images of raw dataset in 1 folder
    This script splits to 2 folders Dogs and Cats
'''
import argparse
import os

from Mexp.explorer import explorer as Mex



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Args for data preparation')
    parser.add_argument('--input_folder', type=str, default='./dogs-vs-cats/train',
                        help='Path to raw dataset')
    parser.add_argument('--output_folder', type=str, default='./data',
                        help='Path to save dataset')

    args = parser.parse_args()

    class_infos = {
        'cat': [],
        'dog': []
    }

    for filename in os.listdir(args.input_folder):
        for classname in class_infos.keys():
            if filename.startswith(classname):
                class_infos[classname].append(os.path.join(args.input_folder, filename))
    
    for classname in class_infos.keys():
        if not os.path.exists(os.path.join(args.output_folder, classname)):
            os.mkdir(os.path.join(args.output_folder, classname))
            print(f'Create "{classname}" folder')

        for img_link in class_infos[classname]:
            Mex.copy_one_file(img_link, os.path.join(args.output_folder, classname))
            filename = img_link.split('\\')[-1]
            print(f'Copy {filename}')



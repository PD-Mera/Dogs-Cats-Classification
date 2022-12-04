'''
    All images of raw dataset in 1 folder
    This script splits to 2 folders Dogs and Cats
'''
import argparse
import os

from Mexp.explorer import explorer as Mex



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Args for data preparation')
    parser.add_argument('--input_folder', type=str, default='/i/dogs-vs-cats/train',
                        help='Path to raw dataset')
    parser.add_argument('--output_folder', type=str, default='/d/AI/Selfcode/Dogs-Cats-Classification/data',
                        help='Path to save dataset')

    args = parser.parse_args()

    args.input_folder = R"I:\dogs-vs-cats\test1\test1"
    args.output_folder = R"D:\AI\Selfcode\Dogs-Cats-Classification\data\test"

    
    Mex.copy_all_file(args.input_folder, args.output_folder)



from Mdeeplearning.data import split

if __name__ == '__main__':
    split.split_train_valid_test(dataroot='/media/mera/Mera/AI/Selfcode/Dogs-Cats-Classification/data_raw',
                                 ratio = '8:2:0',
                                 combine= False)
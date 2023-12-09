import os
import random
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_subset(dataset_folder, save_dir, ratio, op):
    """
    Create subset of the original data
    :param dataset_folder: folder with data to take subset of
    :param save_dir: directory to move subset to
    :param ratio: ratio of files to move from dataset_folder to save_dir
    :param op: "copy" to copy the files or "move" to move the files
    :return: nothing
    """
    subfolders = os.listdir(dataset_folder)

    # Create class folder in save directory
    for folder in subfolders:
        os.makedirs(os.path.join(save_dir, folder))

    for folder in tqdm(subfolders):
        folder_path = os.path.join(dataset_folder, folder)

        # shuffle files so image selection is random
        files = os.listdir(folder_path)
        random.shuffle(files)

        # Calculate number of samples to move
        num = len(files)
        num = int(num * ratio)

        # Move samples
        for i in range(0, num):
            source = os.path.join(dataset_folder, folder, files[i])
            dest = os.path.join(save_dir, folder, files[i])

            if op == 'move':
                shutil.move(source, dest)
            elif op == 'copy':
                shutil.copy(source, dest)


def make_plot():
    y1 = [0.749, 0.650, 0.474, 0.299, 0.185]  # lr=0.01
    # y2 = [0.968, 0.984, 0.990, 0.992, 0.993]  # 0.001
    # y3 = [0.978, 0.989, 0.994, 0.995, 0.996]  # lr=0.0001

    x = [0, 1, 2, 3, 4]

    plt.plot(x, y1, label='lr=0.01')
    # plt.plot(x, y2, label='lr=0.001')
    # plt.plot(x, y3, label='lr=0.0001')

    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.show()


# create_subset(r"C:\Users\kaden\Main\CS678\final_project\sets\ASL",
#               r"C:\Users\kaden\Main\CS678\final_project\sets\0.1_test",
#               0.1, 'move')

make_plot()


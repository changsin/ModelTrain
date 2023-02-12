import argparse
import re
import os
import random
import csv

from openpyxl import load_workbook
from pathlib import Path

from utils import glob_folders, glob_files, glob_files_all


def get_labels_from_xls(filename):
    ID_FILE_NAME = 2
    ID_LABEL_TEXT = 3
    ID_SOUND_TEXT = 5

    script_texts = dict()

    workbook = load_workbook(filename)

    for ws in workbook.worksheets:
        ws = workbook[ws.title]

        id = 0
        for row in ws.rows:
            if id == 0:
                id += 1
                continue
            id += 1

            if row[ID_FILE_NAME].value and row[ID_LABEL_TEXT].value and row[ID_SOUND_TEXT].value:
                data_filename = Path(row[ID_FILE_NAME].value).stem
                label_text = row[ID_LABEL_TEXT].value.replace(",", "")
                sound_text = row[ID_SOUND_TEXT].value.replace(",", "")

                if script_texts.get(data_filename):
                    print("***ERROR: duplicate filename found {}".format(data_filename))
                else:
                    script_texts[data_filename] = (label_text, sound_text)

    return script_texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_in", action="store", dest="path_in", type=str)
    parser.add_argument("--path_data", action="store", dest="path_data", type=str)
    parser.add_argument("--path_out", action="store", dest="path_out", type=str)

    args = parser.parse_args()
    print(args)

    data_files = glob_files_all(args.path_data)
    print("Found {} data files".format(len(data_files)))

    data_files_dict = dict()
    for filepath in data_files:
        filename = Path(os.path.basename(filepath)).stem
        if data_files_dict.get(filename.lower()):
            print("***Dupe data file found {} {}".format(filename, filepath))
        else:
            data_files_dict[filename.lower()] = filepath

    missing_data_files = []

    count_labels = 0
    script_labels = []
    label_sub_folders = glob_folders(args.path_in)
    for label_sub_id, label_sub_folder in enumerate(label_sub_folders):
        label_files = glob_files(label_sub_folder)

        count_labels += len(label_files)

        for xls_filename in label_files:
            script_texts = get_labels_from_xls(xls_filename)
            for data_filename, texts in script_texts.items():
                matched = re.search('^[a-z0-9_]*-', data_filename, re.IGNORECASE).group(0)
                sub_folder = matched.removesuffix('-').removeprefix('script1_')
                sub_filename = os.path.join(sub_folder, data_filename)

                if not data_files_dict.get(data_filename.lower()):
                    print("***Image file Not found for {} in {}".format(data_filename, xls_filename))
                    missing_data_files.append(data_filename.lower())

                # if sub_folder in lite_folders:
                script_labels.append((sub_filename, texts[0], texts[1]))
    print("Found {} items".format(len(script_labels)))

    # for line in script_labels:
    #     filepath = line[0].lower()
    #     filename = Path(os.path.basename(filepath)).stem
    #     if not data_files_dict.get(filename):
    #         # print("***Image file Not found for {}".format(filename))
    #         missing_data_files.append(filename)
    print("Found {} missing data files".format(len(missing_data_files)))
    print(missing_data_files)

    random.shuffle(script_labels)

    with open(args.path_out, 'w', encoding="utf-8") as file_out:
        # file_out.write("[\"{}\", \"{}\"]\n".format("FileName", "Text"))
        file_out.write("{},{},{}\n".format("FileName", "LabelText", "SoundText"))
        for line in script_labels:
            filepath = line[0]
            filename = Path(os.path.basename(filepath)).stem
            if filename.lower() in missing_data_files or filename == "zzpw4041_1-209520-02-99-LGS-F-08-A":
                print("***Not adding missing data file {}".format(filename))
                continue

            # file_out.write("[\"{}.wav\", \"{}\"]\n".format(filepath, line[1]))
            file_out.write("{},{},{}\n".format(filepath, line[1], line[2]))
        # for filename, texts in script_texts.items():
        #     file_out.write("[\"{}.wav\", \"{}\"]\n".format(filename, texts[0]))
    file_out.close()
    print("Wrote to {}".format(args.path_out))

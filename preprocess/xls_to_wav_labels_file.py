import argparse
from openpyxl import load_workbook
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_in", action="store", dest="path_in", type=str)
    parser.add_argument("--path_out", action="store", dest="path_out", type=str)

    args = parser.parse_args()
    print(args)

    labels_xls = "../data/senior_voice_commands/AI-Robot_q_0047_ShinSoonYoung.xlsx"

    ID_FILE_NAME = 2
    ID_LABEL_TEXT = 3
    ID_SOUND_TEXT = 5

    script_texts = dict()

    workbook = load_workbook(args.path_in)

    for ws in workbook.worksheets:
        ws = workbook[ws.title]

        id = 0
        for row in ws.rows:
            if id == 0:
                id += 1
                continue
            id += 1

            filename = Path(row[ID_FILE_NAME].value).stem
            label_text = row[ID_LABEL_TEXT].value
            sound_text = row[ID_SOUND_TEXT].value

            if script_texts.get(filename):
                print("***ERROR: duplicate filename found {}".format(filename))
            else:
                script_texts[filename] = (label_text, sound_text)

    print("Found {} items".format(script_texts.items()))

    with open(args.path_out, 'w', encoding="utf-8") as file_out:
        for filename, texts in script_texts.items():
            file_out.write("[\"{}.wav\", \"{}\"]\n".format(filename, texts[0]))
    file_out.close()

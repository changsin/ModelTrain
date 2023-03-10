import argparse
from collections import namedtuple

from constants import Mode, LabelFormat
from readers import CoCoJsonReader, CVATXmlReader, PascalVOCReader
from utils import *
from writers import YoloV5Writer, EdgeImpulseWriter, CVATXmlWriter, CoCoWriter, PascalVOCWriter

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

"""
convert label files into different formats
"""


def convert_labels(path_in, path_out, from_format, to_format=LabelFormat.EDGE_IMPULSE):
    parser = None
    convertor = None

    if from_format == LabelFormat.CVAT_XML:
        parser = CVATXmlReader()
    elif from_format == LabelFormat.PASCAL_VOC:
        parser = PascalVOCReader()
    elif from_format == LabelFormat.COCO_JSON:
        parser = CoCoJsonReader()
    else:
        print('Unsupported input format {}'.format(from_format))

    if to_format == LabelFormat.EDGE_IMPULSE:
        convertor = EdgeImpulseWriter()
    elif to_format == LabelFormat.YOLOV5:
        convertor = YoloV5Writer()
    elif to_format == LabelFormat.CVAT_XML:
        convertor = CVATXmlWriter()
    elif to_format == LabelFormat.COCO_JSON:
        convertor = CoCoWriter()
    elif to_format == LabelFormat.PASCAL_VOC:
        convertor = PascalVOCWriter()
    else:
        print('Unsupported output format {}'.format(to_format))

    convertor.convert(parser, path_in, path_out)


def convert_xmls(path_in, path_out):
    parser = CVATXmlReader()

    parser.convert_xml(path_in, path_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", action="store", type=Mode.argparse, choices=list(Mode), dest="mode")
    parser.add_argument("--format_in", action="store", type=LabelFormat.argparse, choices=list(LabelFormat), dest="format_in")
    parser.add_argument("--format_out", action="store", type=LabelFormat.argparse, choices=list(LabelFormat), dest="format_out")
    parser.add_argument("--path_in", action="store", dest="path_in", type=str)
    parser.add_argument("--path_out", action="store", dest="path_out", type=str)

    parser.add_argument("--path_data", action="store", dest="path_data", type=str)
    parser.add_argument("--path_labels", action="store", dest="path_labels", type=str)

    args = parser.parse_args()
    print(args.mode)

    if args.mode == Mode.REMOVE_UNLABELED_FILES:
        parent_folder = args.path_in[:args.path_in[:-2].rfind('\\'):]
        args.path_out = os.path.join(parent_folder, "unlabeled")
        if not os.path.exists(args.path_out):
            os.mkdir(args.path_out)

        if os.path.isdir(args.path_in):
            # files = glob_files(args.path, file_type='*')
            files = glob_files(args.path_in, file_type='*.jpg')

            for file in files:
                txt_file = os.path.basename(file)[:-3] + 'txt'
                txt_file = os.path.join(os.path.dirname(file), txt_file)
                if not os.path.exists(txt_file):
                    print('does not have a label file:', txt_file)
                    dest = os.path.join(args.path_out, os.path.basename(file))
                    shutil.move(file, dest)

    elif args.mode == Mode.CONVERT_XML:

        if os.path.isdir(args.path_in):
            files = []
            folders = glob_folders(args.path_in, file_type='*')
            if folders:
                for folder in folders:
                    files.extend(glob_files(folder, file_type='*.xml'))
            # print(files)
            else:
                files = glob_files(args.path_in, file_type='*.xml')

            print(files)

            for file in files:
                convert_xmls(file, args.path_out)
        else:
            convert_xmls(args.path_in, args.path_out)

    elif args.mode == Mode.CONVERT:
        files_in = glob_files_all(args.path_in, file_type='*.xml')
        for file_in in files_in:
            convert_labels(file_in, args.path_out, args.format_in, args.format_out)

    elif args.mode == Mode.SPLIT:
        split_train_val_test_files(args.path_in, args.path_in, args.path_out, ratio=0.1)

    elif args.mode == Mode.COPY_LABEL_FILES:
        copy_label_files(args.path_data, args.path_labels)

    elif args.mode == Mode.FLAT_COPY:
        flat_copy(args.path_in, args.path_out)

    else:
        print("Please specify the mode")

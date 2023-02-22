import argparse
from collections import namedtuple

from constants import Mode, LabelFormat
from readers import CVATXmlReader
from utils import *

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

OVERLAP_PERCENT = [0.95, 0.90, 0.85, 0.80, 0.75]


def check_overlaps(path_in, from_format):
    parser = None

    if from_format == LabelFormat.CVAT_XML:
        parser = CVATXmlReader()
    else:
        print('Unsupported input format {}'.format(from_format))

    parsed = parser.parse(path_in)

    label_count = 0
    overlap_dict_total = dict()
    for threshold in OVERLAP_PERCENT:
        overlap_dict_total[threshold] = 0

    dupe_count = 0
    for image in parsed:
        labels = image[5]
        label_count += len(labels)

        overlap_dict_cur = dict()

        for id1 in range(len(labels)):
            for id2 in range(id1 + 1, len(labels)):
                tag1, xtl1, ytl1, xbr1, ybr1, _, _ = labels[id1]
                tag2, xtl2, ytl2, xbr2, ybr2, _, _ = labels[id2]
                rect1 = Rectangle(float(xtl1), float(ytl1), float(xbr1), float(ybr1))
                rect2 = Rectangle(float(xtl2), float(ytl2), float(xbr2), float(ybr2))
                overlapped_area, max_area = calculate_overlapped_area(rect1, rect2)
                #
                #         if intersect_area >= max_area*threshold:
                # #           if intersect_area >= min_area * threshold and intersect_area < min_area * 0.95:
                #             overlapped_area = intersect_area
                if overlapped_area > 0:
                    for threshold, count in overlap_dict_total.items():
                        if overlapped_area >= max_area * threshold:
                            overlap_dict_cur[threshold] = count + 1

        for threshold, count in overlap_dict_cur.items():
            if overlap_dict_total.get(threshold):
                overlap_dict_total += count

    if dupe_count > 0:
        print("Dupes {}: {} {}".format(path_in, dupe_count, overlap_dict_total))

    return overlap_dict, label_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", action="store", type=Mode.argparse, choices=list(Mode), dest="mode")
    parser.add_argument("--format_in", action="store", type=LabelFormat.argparse, choices=list(LabelFormat), dest="format_in")
    parser.add_argument("--path_in", action="store", dest="path_in", type=str)

    args = parser.parse_args()
    print(args.mode)

    if args.mode == Mode.CHECK_OVERLAPS:
        files_in = glob_files_all(args.path_in, file_type='*.xml')

        total_label_count = 0
        overlap_dict = dict()
        for file_in in files_in:
            tmp_dict, label_count = check_overlaps(file_in, args.format_in)

            for key, value in tmp_dict.items():
                if overlap_dict.get(key):
                    overlap_dict[key] += value
                else:
                    overlap_dict[key] = value

            total_label_count += label_count

        # Calculate the total counts
        for key, value in overlap_dict.items():
            print("{}: {} ({}%)".format(key, value, float(value)/float(total_label_count)*100))

    else:
        print("Please specify the mode")

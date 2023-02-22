# Scripts related to training ML Models

The project contains ML model training and related preprocessing scripts

## Setup
- Optional1: Setup a virtual env.
```commandline
python -m venv venv
```
- Optional2: Activate the virtual env.
```commandline
source venv/bin/activate
```

1. Install the dependencies from venv

```commandline
pip install -r requirements.txt
```

## Train


## Convert Labels


### 1. From CVAT xml to COCO JSON
예: [수도권 도로 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=61)

```commandline
python main_convert_labels.py --mode convert --format_in cvat_xml --format_out coco_json --path_in [folder containing cvat xmls] --path_out [output folder]
```


### 2. From CVAT XML to Pascal VOC
예: [어린이 구역 위험 영상](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=169)

```commandline
python main_convert_labels.py --mode convert --format_in cvat_xml --format_out pascal_voc --path_in [folder containing cvat xmls] --path_out [output folder]
```

## Check overlapping labels


```commandline
python main_convert_labels.py --mode check_overlaps --format_in cvat_xml --path_in D:\data\AI-Hub\ChildZoneCCTV\labels_cvat
```
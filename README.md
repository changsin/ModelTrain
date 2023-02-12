# Model Train

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
pip install -qr requirements.txt
```

## Train

### 1. voice-recognition
For 명령어

The command below will run 2 runs with 20 epochs each

Run 1: use_column 1 will use AI-Hub labels 
Run 2: use_column 2 will use Testworks labels 

The option --divide 40000 will use only the first 40,000 files & labels.
This is to cut down the time to train (taks a long time)

```commandline
python3 main.py --epochs 20 --divide 40000 --use_column 1 ; python3 main.py --epochs 20 --divide 40000 --use_column 2
```

## Convert Labels


## 1. From CVAT xml to COCO JSON
예: [수도권 도로 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=61)

```commandline
python main_convert_lables.py --mode convert --format_in cvat_xml --format_out coco_json --path_in [folder containing cvat xmls] --path_out [output folder]
```


## 2. From CVAT XML to Pascal VOC
예: [어린이 구역 위험 영상](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=169)

```commandline
python main_convert_lables.py --mode convert --format_in cvat_xml --format_out pascal_voc --path_in [folder containing cvat xmls] --path_out [output folder]
```

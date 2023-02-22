# Model Train

This project contains ML models of different kind

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

## Trained Models

The folder '[notebooks](models/notebooks) contains experiments and visualizations of the trained models.

### 1. voice-recognition
[명령어(노인남여)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=94)

**Model:** Resnet50

The command below will run 2 runs with 20 epochs each

Run 1: use_column 1 will use AI-Hub labels 
Run 2: use_column 2 will use Testworks labels 

The option --divide 40000 will use only the first 40,000 files & labels.
This is to cut down the time to train (takes a long time ~2 hours)

```commandline
python3 main.py --epochs 20 --divide 40000 --use_column 1 ; python3 main.py --epochs 20 --divide 40000 --use_column 2
```
### 2. Child Zone Dangerous Acts
[어린이 구역 위험 영상](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=169)

**Model:** YOLOv5

The same model was used twice.
1. Using AI-Hub data
2. Using TW re-labeled data

Using the default hyper-parameters, the model is trained with:
- epochs=100
- batch size = 32

The trained models are saved here:
- [AI-Hub trained model](models/yolov5/childzone/ai-hub)
- [TW trained model](models/yolov5/childzone/tw)

#### Data
Total image data: 17,280 (Train) + 2,160 (Valid) + 2,160 (Test) = 21,600

Train:Valid:Test = 8:1:1

The images are randomly shuffled.


#### Results
The model trained with TW re-labeled data performed 9% (81.7% -> 90.8%) better than the AI-Hub data trained model.

**AI-Hub**

                 Class     Images  Instances          P          R      mAP50   
                   all       2160      31233      0.913       0.69      0.817      0.578
                person       2160       7116      0.927      0.764      0.868      0.586
               vehicle       2160      14547       0.93      0.779      0.877      0.679
                 cycle       2160        715      0.968        0.8      0.896       0.72
                  kick       2160        521       0.91      0.311      0.618      0.377
                  face       2160        935      0.721      0.301      0.509      0.201
         license_plate       2160       2154      0.921      0.768      0.871      0.624
              umbrella       2160        476      0.966      0.836      0.913      0.608
         traffic_light       2160       4353      0.953      0.857      0.924      0.742
             motorbike       2160        416       0.92      0.798      0.881      0.666

**TW**

                 Class     Images  Instances          P          R      mAP50   
                   all       2160      31233      0.955      0.834      0.908      0.738
                person       2160       7116      0.939      0.852      0.918       0.72
               vehicle       2160      14547      0.948      0.869      0.929      0.825
                 cycle       2160        715      0.981       0.94      0.969      0.876
                  kick       2160        521      0.973      0.693      0.838       0.57
                  face       2160        935      0.847      0.561      0.726      0.396
         license_plate       2160       2154      0.968      0.882      0.938      0.782
              umbrella       2160        476      0.968      0.954      0.975      0.788
         traffic_light       2160       4353       0.99      0.909      0.954      0.869
             motorbike       2160        416      0.981      0.846      0.921      0.816
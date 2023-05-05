# HelpDefault
한양대학교 ERICA 공학대학 전자공학부 전공학회 DEFAULT

__2022 DEFAULT 메이커톤 1등 수상작 "도와줘요 디폴트!"__

23기 곽민창, 23기 위성규


# 도와줘요 디폴트!

<div align="center">
<img width="329" alt="image" src="https://github.com/kwarkmc/HelpDefault/blob/main/templates/defal.jpg?raw=true">

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkwarkmc%2FHelpDefault&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

# HelpDefault
> **2022 DEFAULT 메이커톤 1등 수상작** <br/> **2022. 07 무박 2일 개발완료**

## 세미나 멤버 소개

|      곽민창       |          위성규         |
|:---:|:---:|
   <img width="160px" src="https://avatars.githubusercontent.com/u/41298500?v=4" />|<img width="160px" src="https://avatars.githubusercontent.com/u/49241440?v=4" />|
|   [@kwarkmc](https://github.com/kwarkmc)   |    [@wiesunggue](https://github.com/wiesunggue)  |
| 한양대학교 ERICA 전자공학부      4학년 | 한양대학교 ERICA 전자공학부     4학년 |

## 2022 DEFAULT 메이커톤

**한양대학교 ERICA 전자공학부 전공학회 DEFAULT** 에서 22년도 7월 여름방학 중에 진행된 제 2회 메이커톤에서 1등을 수상했다. ML/DL 뿐만 아니라 HW/SW 등 모든 분야에 대한 작품들이 동시에 출품되었다.

## 프로젝트 소개

학회방에 배치되어 있는 각종 공구 및 소자를 신규 학회원들로 하여금 선배들의 도움 없이 쉽고 간편하게 사용할 수 있도록 사용을 돕고, 금방 더러워지는 학회방을 깨끗하게 유지하고 공구 및 소자를 분실하지 않도록 정리를 돕는 서비스를 개발했다.

직접 학회방에 있는 각종 공구 및 소자의 사진을 여러 각도 및 형태에 따라 다양하게 촬영하여 10개의 Label에 대해 Custom DataSet을 구성하여 학습에 사용하였다.

빠른 학습 및 높은 Accuracy를 위하여 ResNet의 Pre-Trained 데이터를 사용하여 전이학습 (Transfered - Learning) 을 활용하여 데이터를 학습하였다.

## 시작 가이드 🚩
### Requirements
For building and running the application you need:

- [Anaconda3](https://www.anaconda.com/download/)
- [Pytorch 2.0](https://pytorch.org/)
- [Jupyter Notebook](https://jupyter.org/)
- [Pycharm](https://www.jetbrains.com/ko-kr/pycharm/download/#section=windows)

### Installation
``` bash
$ git clone git@github.com:kwarkmc/HelpDefault.git
$ cd HelpDefault
```
#### 서버 실행
``` bash
$ python ./rest_test.py
```

#### 모델 학습
``` bash
$ python ./defaultdeep.py
```

---

## Stacks 📚

### Environment
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white)
   

### Framework
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

### Development
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)

### Communication
![Microsoft PowerPoint](https://img.shields.io/badge/Microsoft_PowerPoint-B7472A?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)          

---
## 학습 결과 📺
| Sequential Model 결과  |  ResNet Model 결과   |
| :-------------------------------------------: | :------------: |
|  <img width="329" src="https://github.com/kwarkmc/RestApp/blob/main/Documents/pic/RestApp%20result.png?raw=true"/> |  <img width="329" src="https://github.com/kwarkmc/RestApp/blob/main/Documents/pic/ResNet_example_result.png?raw=true"/>|  
| Layer에 따른 Accuracy의 변화   |  2개 - 3개 - 4개 - 5개   |  
| <img width="329" src="https://github.com/kwarkmc/RestApp/blob/main/Documents/pic/Result_model1.JPG?raw=true"/>   |  <img width="329" src="https://github.com/kwarkmc/RestApp/blob/main/Documents/pic/Result_model2.JPG?raw=true"/>     |

> Layer가 많아지면 Overfitting이 발생하여 오히려 Accuracy가 낮게 나올 수 있다는 것을 볼 수 있다.
---
## 주요 기능 📦

### ❗ Custom DataSet을 폴더에서 Open 하여 Data와 Label로 준비
- 파일의 이름에 Labeling이 되어있고, OS API를 Import 하여 20개의 라벨에 맞춰 병합
- 학습에 사용하기 위해 `LabelBinarizer()`를 이용하여 원핫 인코딩
- train / Test / Validation 데이터를 `train_test_split()` 을 이용하여 나눠서 구현
- 학습에 대한 검증은 Validation 데이터로, 실전 정확도는 Test 데이터를 사용하여 Overfitting을 방지할 수 있다.

### ❗ 데이터 증강 테크닉 사용
- `keras.preprocessing.image` 의 `ImageDataGenerator`를 사용하여 **Rotation / Zoom / Shift / Shear / Flip** 등의 CV적 Augmentation을 진행하여 학습에 사용하였다.

### ❗ 모델을 H5 파일로 저장하여 Weight만 예측에 사용
- 학습한 Weight들을 H5 파일로 저장하여 Python 파일 외부에서 입력 데이터에 대해 예측을 진행할 때 매번 학습을 하지 않고 모델을 다시 불러와 쉽게 사용할 수 있다.
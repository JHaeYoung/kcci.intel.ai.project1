# kcci.intel.ai.project1

# Project name

* AI로 만드는 인생네컷
* 402-pose-estimation 모델과 205-background-removal 모델을 적용해 웹캠으로 동작을 인식하고 각 동작의 배경을 바꾼 사진을 출력할 수 있다.

## Requirement

* (프로젝트를 실행시키기 위한 최소 requirement들에 대해 기술)

```
* 10th generation Intel® CoreTM processor onwards
* At least 32GB RAM
* Ubuntu 22.04
* Python 3.9
```

## Clone code

* (Code clone 방법에 대해서 기술)

```shell
git clone https://github.com/zzz/yyy/xxxx
```

## Prerequite

* (프로잭트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정방법에 대해 기술)

```shell
python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install wheel

python -m pip install openvino-dev

cd /path/to/repo/xxx/
python -m pip install -r requirements.txt
```

## Steps to build

* (프로젝트를 실행을 위해 빌드 절차 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

make
make install
```

## Steps to run

* (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

cd /path/to/repo/xxx/
python demo.py -i xxx -m yyy -d zzz
```

## Output

![./images/result.jpg](./images/result.jpg)

## Appendix

# ⚖️ TF-IDF & Transformer 기반 법원 판결 예측

미국 연방대법원 판례의 승소 당사자를 예측하는 머신러닝·딥러닝 프로젝트.
TF-IDF 기반 classical ML(LightGBM, SVM)과 트랜스포머 기반 언어모델(Legal-BERT, DeBERTa-v3, RoBERTa)을 비교 분석하여 법률 판결 예측 태스크에서의 효과를 평가한다.


## 📌 Overview

미국 연방대법원 판례의 사건 개요(facts)는 두 당사자(`first_party`, `second_party`) 간 분쟁을 요약한 짧은 자연어 텍스트다. `facts` 텍스트만으로 `first_party`의 승소 여부를 예측하는 것이 이 과제의 목표다.

법률 판결 예측은 까다로운 NLP 태스크다 — 법령, 판례, 법리적 추론 없이 오직 사건 개요만으로 결과를 추론해야 한다.

이 프로젝트는 TF-IDF + classical ML부터 트랜스포머 파인튜닝까지 복잡도가 점차 증가하는 6개 모델을 구축·비교하고, local 검증 점수가 리더보드로 이어지지 않은 원인을 규명하는 진단 과정까지 포함한다.

- 대회: [DACON](https://dacon.io/competitions/official/236112)
- 평가지표: **Accuracy**


## 🎯 Objectives

- 사건 개요(facts) 텍스트로 `first_party_winner`(0/1) 예측
- TF-IDF + classical ML vs. 트랜스포머 파인튜닝 성능 비교
- Local CV 성능이 리더보드로 이어지지 않은 원인 진단
- 소규모(2,478행) 데이터셋에서 법률 판결 예측에 가장 적합한 아키텍처 탐색


## 🧠 Key Features

- TF-IDF + classical ML baseline(LightGBM, SVM) 전 과정의 하이퍼파라미터/진단 로그 기록
- 도메인 특화 vs. 일반 도메인 트랜스포머 비교(Legal-BERT vs. DeBERTa-v3 vs. RoBERTa)
- 5-fold Stratified CV 기반 OOF(out-of-fold) 평가를 전 모델에 일관 적용
- **Local CV–LB 점수 괴리**의 원인을 train/test 클래스 분포 차이로 규명한 근본 원인 분석
- GPU 환경 트러블슈팅 기록(Colab/Kaggle, `transformers` v5 인자명 변경, fp16/bf16, Kaggle P100 비호환 이슈 등)


## 🗂️ Dataset

| 파일 | 설명 |
|---|---|
| `train.csv` | `ID`, `first_party`, `second_party`, `facts`, `first_party_winner`(0/1) — 2,478행 |
| `test.csv` | `ID`, `first_party`, `second_party`, `facts` — 1,240행 |
| `sample_submission.csv` | `ID`, `first_party_winner` — 제출 양식 |

- 언어: 영어 (미국 연방대법원 공개 판례)
- Train 클래스 비율: `1`(승소) : `0`(패소) ≈ 66.5% : 33.5%

> 대회 규정상 데이터 파일은 이 저장소에 포함하지 않았습니다. [대회 페이지](https://dacon.io/competitions/official/236112)에서 다운로드해 각 노트북이 참조하는 경로에 두고 실행하세요.


## ⚙️ Methodology

모델 비교를 반복하는 파이프라인으로 진행:

1. **데이터 전처리**
   - classical ML: facts/party명 각각 별도 TF-IDF 벡터화
   - 트랜스포머: 토크나이징(max length 384)
2. **모델 개발**
   - TF-IDF + Logistic Regression / LightGBM / SVM
   - Fine-tuning: Legal-BERT / DeBERTa-v3-base / RoBERTa-base
3. **학습**
   - 5-fold Stratified CV, 이진분류
   - 대회 지표와 일치시키기 위해 `metric_for_best_model="accuracy"` 사용
4. **평가**
   - Accuracy(공식 지표), Macro F1(진단용 보조 지표)
   - OOF(out-of-fold) 점수를 LB의 local 대리 지표로 사용


## 🧩 Models

### 1. TF-IDF + Logistic Regression (baseline)
- 대회 제공 baseline
- vectorizer 재사용 버그 발견(facts로만 fit한 vectorizer를 party명에도 그대로 transform)

### 2. TF-IDF + LightGBM
- Sparse TF-IDF 피처에 트리 기반 앙상블 적용
- 2~14 boosting round 만에 다수 클래스 예측으로 수렴(고차원·저샘플 과적합)

### 3. TF-IDF + SVM (LinearSVC)
- 고차원 sparse 피처엔 트리보다 유리한 선형모델
- `class_weight="balanced"` 효과를 무력화시키던 `CalibratedClassifierCV` 버그 진단 및 제거

### 4. Legal-BERT
- 도메인 특화 사전학습 모델(EU 법령·계약서 + 판례 중심)
- 미 연방대법원 서술체와 도메인이 어긋나며, 2,478행 소규모 데이터에서 과적합에 취약

### 5. DeBERTa-v3-base
- 일반 도메인, disentangled attention 구조
- 현재 하이퍼파라미터 기준 5-fold 전부 다수 클래스 예측으로 수렴

### 6. RoBERTa-base
- 일반 도메인, mixed precision과 호환성이 좋은 구조
- 지금까지 local 최고 성능이나, fold별 수렴 양상은 불안정


## 📊 Model Evaluation

| 모델 | Local OOF Accuracy | Local OOF Macro F1 | 상태 |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | - | - | 완료 |
| TF-IDF + LightGBM | 0.6655 | 0.3996 | 완료 — 다수 클래스 수렴 |
| TF-IDF + SVM (class_weight 없음, C=0.02) | 0.6667 | ~0.42 | 완료 — 다수 클래스와 유사 |
| Legal-BERT | 0.60–0.63 | 0.52–0.54 (참고치) | 완료 |
| DeBERTa-v3-base | 0.6655 | 0.3996 | 완료 — 전 fold 다수 클래스 수렴 |
| RoBERTa-base | **0.6711** | **0.4282** | 완료 — local 최고|


## 🛠️ Tech Stack

- Python, pandas, numpy
- scikit-learn, LightGBM
- PyTorch, Hugging Face `transformers`
- Google Colab / Kaggle Notebook (GPU: T4)


## 📁 Project Structure

```
.
├── DACON-base-LR.ipynb         # Baseline: TF-IDF + Logistic Regression
├── DACON-base-lightGBM.ipynb   # TF-IDF + LightGBM
├── DACON-base-SVM.ipynb        # TF-IDF + LinearSVC
├── DACON-legalBERT.ipynb       # Fine-tuning: Legal-BERT
├── DACON-DeBERTa.ipynb         # Fine-tuning: DeBERTa-v3-base
├── DACON-RoBERTa.ipynb         # Fine-tuning: RoBERTa-base
└── README.md
```


## 🚀 Results

- Classical ML → 트랜스포머 파인튜닝까지 6개 모델을 구축·비교
- RoBERTa-base가 local OOF 기준 최고 성능(Acc 0.6711, Macro F1 0.4282) 기록
- Local CV–LB 괴리의 원인을 파이프라인 결함이 아닌 train/test 클래스 분포 차이로 규명
- 여러 환경 이슈(transformers v5 인자명 변경, DeBERTa-v3 fp16 불안정, Kaggle P100 GPU 비호환, Colab 세션 끊김 대응)를 문서화하고 해결


## 📌 Limitations

- Train 데이터가 작고(2,478행) 불균형(66:34)한 반면, test는 50:50에 가까운 것으로 추정 — all-constant 제출로 아직 확정되지 않음
- 지금까지 어떤 모델도 unseen 데이터에서 majority baseline을 뚜렷하게 넘는 안정적인 예측 신호를 보여주지 못함
- 도메인 특화 사전학습(Legal-BERT)이 일반 도메인 모델보다 낫지 않았음 — EU 법률 코퍼스와 미 연방대법원 서술체 간 도메인 불일치로 추정
- 동일 모델·동일 하이퍼파라미터에서도 fold별 학습 안정성 편차가 큼(RoBERTa fold 1 vs. fold 2)


## 👤 Author

Park, Eunsoo


## ✔️ Summary

> TF-IDF 기반 classical ML과 트랜스포머 파인튜닝을 비교해 미국 연방대법원 판결을 예측하고, local–리더보드 점수 괴리를 끝까지 진단한 NLP 비교 실험 프로젝트.

# HEAT 중간발표 Q&A 통합본

출처 표시: **[PDF]** Q&A.pdf · **[Word]** QNA_intermediate.docx · **[추가]** 두 파일에 없어서 새로 넣은 질문
두 파일에 겹치는 질문은 하나로 합쳤고, 내용이 서로 다른 부분은 맨 아래 "확인 필요"에 모았다.

---

## A. 정답(GT)과 핫플 정의

### Q1. 왜 20~39세로 제한했나? [PDF]
*Why did you limit the target to ages 20–39?*

- 선행연구: 서울 상권 젠트리피케이션은 40세 이하 매출 비중, 40세 미만 유동인구 비율과 관련이 컸다(김경선·김동섭, 2019). 20·30대는 매출 규모보다 상권 성장에 더 큰 영향을 줬다(강현모·이상경, 2019).
- 우리 데이터(2025년, 부록 그래프 age_share_appendix): 방문객 중 20~39세 비중이 서교동 61%, 역삼1동 52%, 여의동 43%. 전 연령 방문량 상위 10% 동의 중앙값은 40%로, 서울 동 중앙값(28.5%)의 약 1.4배다.
- 상위 10%를 **전 연령** 방문량으로 뽑았기 때문에 "20~39세로 뽑아서 비중이 높은 것"이라는 순환 논리 공격을 피할 수 있다.
- 논문 확인(초록 기준): 김경선·김동섭(2019), 「서울시 상업젠트리피케이션 영향요인에 관한 연구」, 한국콘텐츠학회논문지 19(2). 강현모·이상경(2019), 「시계열 군집분석과 로지스틱 회귀분석을 이용한 골목상권 성장요인 연구」, 한국측량학회지 37(6).

> Prior research shows that gentrification in Seoul's commercial districts is strongly linked to the sales share of customers under 40 and the share of floating population under 40 (Kim & Kim, 2019). It also found that people in their 20s and 30s affect district growth more than total sales volume does (Kang & Lee, 2019). Our data shows the same pattern: in Seogyo-dong, 61% of visitors are aged 20–39, and in the busiest 10% of dongs the median is 40%, compared with 28.5% across Seoul. So we used visits by 20–39-year-olds as the target, because that is where the signal of a rising district shows up first and most clearly.

**Keyword:** prior research (under 40, growth not sales), Seogyo 61%, busiest 10% = 40% vs 28.5%

### Q2. 왜 상위 10%를 기준으로 했나? [PDF]
*Why did you use the top 10% as the threshold?*

- +20% 같은 고정 기준은 서울 전체가 오르내릴 때 핫플 개수가 같이 출렁인다. 상위 10%는 서울 전체와 함께 움직여서 시점끼리 비교할 수 있다.
- 10%는 관례적인 선택이라 5%·20% 결과도 보고할 예정이다.

> With a fixed cutoff like +20%, the number of hotspots would rise and fall whenever all of Seoul goes up or down. A top-10% threshold moves with the city as a whole, so we can compare results across time points. The 10% figure itself is a conventional choice, so we will also report results at 5% and 20%.

**Keyword:** fixed cutoff moves with Seoul, comparable across time, 5% and 20% planned

### Q3. 핫플 정의가 타당하다는 근거는? [PDF]
*What evidence shows that your definition is valid?*

- 현재 핫스팟으로 서교동, 역삼1동, 여의동, 종로, 한강로동, 명동처럼 누구나 아는 곳이 나온다.
- 다른 출처 자료와의 순위 상관(Spearman): 카드 매출 0.84, 음식점 수 0.83, 지하철 하차 0.73, 검색 비중 0.46.
- 핫스팟 43곳 중 72%는 독립 자료 2개 이상에서도 상위 10%다.

> The current hotspots are places everyone knows: Seogyo-dong, Yeoksam 1-dong, Yeoui-dong, Jongno, Hangangno-dong, and Myeong-dong. Their rankings also correlate strongly with independent sources (Spearman: card sales 0.84, restaurant count 0.83, subway alightings 0.73, search share 0.46). And 72% of the 43 hotspots are also in the top 10% in at least two independent sources.

**Keyword:** well-known places, rank correlation, 72% of 43 in two sources

---

## B. 데이터

### Q4. 2024년 8~10월을 데이터 오류로 본 근거는? [PDF + Word 통합]
*Why did you treat August–October 2024 as a data error?*

- 서울 전체 생활이동이 이 3개월에만 약 1.5배 뛰었다가 11월에 원래대로 돌아왔다.
- 같은 기간 지하철 하차, 생활인구, 카드 매출은 작년 수준 그대로였다(부록 그래프).
- 출근·귀가 목적 이동까지 같이 뛰었다. 핫플 방문이 실제로 늘었다면 출근이 늘 이유가 없다. [추가]
- 서울시 공지나 집계 방식 변경 이력은 찾지 못했다.
- 보정 후 작년 같은 주 모델 WAPE가 21.6% → 13.9%. 평가는 보정 전 원래 값으로 했다.

> Visits across all of Seoul jumped about 1.5 times during just these three months, then returned to normal in November. In the same period, subway ridership, living population and card sales stayed at last year's level (see Appendix). Even commute and return-home trips jumped, which a real rise in visits would not cause. We found no notice about this from the data provider, so we treated it as a data error. After the correction, the WAPE of the same-week-last-year model dropped from 21.6% to 13.9%. We still evaluated on the original, uncorrected values.

**Keyword:** only mobility jumped, subway / living pop / card flat, even commute jumped, no notice, evaluated on original

---

## C. 평가 방법

### Q5. 왜 MAPE 대신 WAPE를 썼나? [PDF]
*Why did you use WAPE instead of MAPE?*

- WAPE는 방문이 적은 주에서 오차율이 폭발하지 않고, "전체 방문량 대비 몇 % 틀렸나"로 바로 읽힌다.
- 단점: 큰 동의 오차가 크게 반영된다. 그래서 동별 WAPE 분포를 같이 보여주고, 모델 비교는 동별로 짝지은 Wilcoxon 검정으로 했다.

> With WAPE, the error rate doesn't blow up in weeks with few visits, and the result reads simply as "how many percent off we were relative to total visits." But overall WAPE sums the errors before dividing, so dongs with large populations count more. An error of 10,000 people in Seogyo-dong doesn't carry the same weight as an error of 100 people in a small dong. That's why we also show the WAPE distribution by dong, and we compared models with a paired Wilcoxon test by dong, which isn't affected by dong size.

**Keyword:** no blow-up at small values, big dongs weigh more, per-dong distribution, paired Wilcoxon

### Q6. 예측 시점 4개는 너무 적지 않나? 그 4개로 모델도 고르지 않았나? [PDF + 추가]
*Aren't four time points too few? Didn't you also choose the model on them?*

- 맞다. 시점 하나가 평균을 크게 바꿀 수 있다. 최종 발표 전에 2024년 7월~2025년 7월 매월 13개 시점으로 늘린다.
- [추가] 같은 4개 시점으로 모델을 고르고 성능도 보고했으므로, 지금 숫자는 검증(validation) 결과다. 결합은 1:1 고정이라 조정한 가중치는 없다. 최종 시험은 아직 안 쓴 2026년 구간으로 한 번만 한다.

> Yes. A single time point can shift the average a lot. Before the final presentation, we plan to expand to 13 time points, one per month from July 2024 to July 2025. Also, since we chose the model on these same dates, we treat the current numbers as validation results. The combination is a fixed 1:1 average, so no weight was tuned, and we will run one final test on 2026 data that we have not used.

**Keyword:** yes, expand to 13 monthly points, validation not test, fixed 1:1, final test on 2026

### Q7. 왜 13주 앞을 예측하나? [PDF]
*Why a 13-week forecast horizon?*

- 3개월은 상인과 행정이 준비할 수 있는 기간이고, 분기 단위 의사결정과 맞다.
- 월별 실험에서 기준선보다 나은 건 3개월까지였고, 6개월은 기준선과 비슷했다. 최종 전에 4·8·13·26주를 비교한다.

> Three months gives business owners and local government time to prepare, and it matches quarterly decision-making. In our monthly experiment, the model beat the baseline only up to three months; at six months it was about the same as the baseline. Before the final presentation, we will compare results at 4, 8, 13, and 26 weeks.

**Keyword:** time to prepare, quarterly decisions, better than baseline up to 3 months, 4/8/13/26 planned

### Q8. WAPE가 낮으면 떠오르는 핫플을 잘 찾는다는 뜻인가? [Word]
*Does a low WAPE mean you find emerging hot places well?*

- 지금 결과는 "이동량을 잘 예측한다"까지만 보여준다.
- 외부 근거 자료를 모으고 있고, 예측 증가율 상위 10%와 실제 상위 10%를 비교해 precision/recall을 추가할 예정이다.

> WAPE tells us how well we forecast the level of mobility, but it does not directly measure whether we identify the right emerging dongs. We are collecting external evidence for hot place predictions. We will compare the predicted top 10% by growth with the actual top 10%, and report precision and recall.

**Keyword:** not validated yet, collecting external evidence, compare prediction and actual

---

## D. 모델 한계와 검색량

### Q9. 급변을 못 잡는 모델로 떠오르는 핫플을 찾을 수 있나? [Word]
*Can a model that misses sudden changes find emerging hot places?*

- 지금 모델은 안정적이지만 급변에 느리게 반응하는 것이 한계가 맞다.
- 다음 단계는 오류 분석과 머신러닝·딥러닝 모델이다.

> Yes, that's the main limitation of our current model: it reacts slowly to sudden rises. So our next step focuses on error analysis and on machine learning and deep learning models.

**Keyword:** limitation of current model, next step ML/DL

### Q10. 검색량이 생활이동을 선행하는지 검정했나? (교차상관, 그랜저 인과) [Word]
*Did you test whether search leads mobility?*

- 아직 안 했다. 다음에 한다.

> Not yet. We will run cross-correlation and a Granger causality test to check this properly.

**Keyword:** not yet, will check next

### Q11. 검색으로 좋아진 139개 동은 우연 아닌가? [추가]
*Any extra input improves some dongs by chance. How do you know these 139 dongs improved because of search?*

- 가짜 입력 실험에서 다른 동의 검색량을 넣어도 4시점 평균 오차가 거의 같았다(자기 동 23.6% vs 가짜 23.7%).

> Search improved 139 dongs, and it hurt least where search volume is mid to high, so it is not useless. But we can't yet separate that from chance: in our placebo test, another dong's search gave almost the same average error. Our next step is to find which dongs benefit consistently across forecast dates, and use search only there.

**Keyword:** not useless, placebo almost same, find consistent dongs

### Q12. 52주 SARIMA가 가장 나쁜데 연간 계절성이 중요하다는 말과 모순 아닌가? [추가]
*52-week SARIMA is your worst model at 29.9%. Doesn't that contradict your claim that annual seasonality matters?*

> That result is on raw data, where last year's weeks contain the 2024 jump, so the seasonal model copies the anomaly. After cleaning, annual information — last year's value and Fourier terms — gave the best results.

**Keyword:** raw data, copies the anomaly, annual info best after cleaning

---

## E. 발표 구성

### Q13. 피드백 반영 슬라이드는 어디 있나? [Word]
*Where is the reflection on feedback?*

- 대부분은 핫플 정의 슬라이드(4번)에 있고, 나머지는 발표 전체 구조에 녹였다.
- 아직 반영하지 못한 것(지속성, 민감도, 외부 검증)은 최종 발표에서 한다.

> Most of the reflections are shown in slide 4, and we built the feedback into the whole presentation. We replaced the Hotness score with an observed value, changed to weekly data, and redefined "hot place". Persistence, sensitivity tests and external validation are planned for the final.

**Keyword:** mostly slide 4, replaced Hotness score, weekly data, redefined hot place, rest in final

---

## 확인 필요 (두 파일이 서로 다르거나 다른 자료와 안 맞는 부분)

- [ ] **Q1 원래 숫자 "핫플 50~60%":** 2025년 데이터로는 50~60%가 서교동·역삼1동뿐이고, 상위 10% 중앙값은 40%다. 위 답변은 새 숫자로 바꿨다.
- [ ] **4+5+7의 실제 구성:** 2025년 기준 기타 98.7%, 쇼핑 1.2%, 관광 0.2%. "사실상 기타 아니냐"는 질문에 대비해야 한다.
- [ ] **한강로동:** Q3에서는 현재 핫스팟인데, 2025년 20~39세 방문량으로는 93위다. 정의 방식 차이인지 재현에게 확인.

- [ ] **Q8 "아직 검증 안 됨" vs 재현 문서:** 재현 문서(10_실험_출처별_정리)에는 떠오르는 곳 상위 10% 적중률이 이미 있다(모델 58%, 지금 그대로 53%, 무작위 10%). "Not validated yet"보다 "first check: 58% vs 53%, small gap"이 정확하다. 팀이 이 숫자를 쓸지 정해야 한다.
- [ ] **Q4 생활인구를 "독립 자료"라고 부를지:** Word는 "two independent sources"라고 했는데, PDF는 생활인구가 생활이동과 "같은 통신 원자료"라고 했다. 같은 통신 자료가 맞다면 "independent"는 빼야 한다. 합친 답에서는 "independent"를 뺐다.
- [ ] **Q4 근거 자료 목록:** PDF는 카드 매출·생활인구, Word는 지하철·생활인구였다. 합친 답은 셋 다 넣었고, 부록 그래프에도 셋 다 있다.
- [ ] **Q6 13개 시점 계획:** 2024년 7~10월 시점은 학습 끝이나 예측 구간이 이상 구간과 겹친다. 이 시점들을 어떻게 다룰지 정해 두면 좋다.
- [ ] **Q9 다음 단계 모델 이름:** Word는 "ML/DL", 재현 문서는 "LightGBM + Ridge". 하나로 맞추면 좋다.

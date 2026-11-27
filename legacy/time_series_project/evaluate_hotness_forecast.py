"""Held-out 2025 Hotness Score ranking evaluation using 2024-only scales."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/semester_pilot'

actual=pd.read_csv(OUT/'named_place_hotness_score_quarterly.csv',dtype={'quarter':str})
bridge=pd.read_csv(OUT/'place_trade_bridge_provisional.csv',dtype={'trade_code':str})
search=pd.read_csv(OUT/'search_3month_backtest_predictions.csv')
search['quarter']=search.target_month.str[:4]+(((search.target_month.str[5:7].astype(int)-1)//3)+1).astype(str)
search=search.groupby(['search_area','quarter','model'],as_index=False).agg(pred_search=('forecast','mean'),months=('target_month','nunique'))
search=search[search.months==3]
foot=pd.read_csv(OUT/'trade_estimated_footfall_next_quarter_backtest_predictions.csv',dtype={'trade_code':str,'target_quarter':str})
foot=foot.merge(bridge[['search_area','trade_code']],on='trade_code',how='inner',validate='many_to_one')
foot=foot.groupby(['search_area','target_quarter','model'],as_index=False).agg(pred_foot=('forecast','sum'),pred_trade_count=('trade_code','nunique'))
foot=foot.rename(columns={'target_quarter':'quarter'})

reference=actual[(actual.quarter>='20241')&(actual.quarter<='20244')]
refs={col:np.sort(reference[col].dropna().to_numpy()) for col in
      ('search_interest_mean','search_interest_mean_yoy','footfall_per_trade','footfall_per_trade_yoy')}
prior=actual[['search_area','quarter','search_interest_mean','footfall_per_trade']].copy()
prior['quarter']=(prior.quarter.str[:4].astype(int)+1).astype(str)+prior.quarter.str[-1]
prior=prior.rename(columns={'search_interest_mean':'search_prior_year','footfall_per_trade':'footfall_prior_year'})
base=actual[(actual.quarter.str[:4]=='2025')&actual.score_valid].merge(prior,on=['search_area','quarter'],how='left',validate='one_to_one')

def percentile(col,values):
    ref=refs[col]
    return np.searchsorted(ref,values,side='right')/len(ref)*100

rows=[]
for lag,method in ((2,'last_available_quarter_score'),(4,'same_quarter_last_year_score')):
    prior_score=actual[['search_area','quarter','hotness_score']].copy()
    def advance(q):
        year=int(q[:4]);season=int(q[-1])+lag
        return str(year+(season-1)//4)+str((season-1)%4+1)
    prior_score['quarter']=prior_score.quarter.map(advance)
    prior_score=prior_score.rename(columns={'hotness_score':'pred_score'})
    naive=base.merge(prior_score,on=['search_area','quarter'],how='inner').dropna(subset=['pred_score']).copy()
    naive['method']=method
    rows.append(naive[['search_area','quarter','method','hotness_score','pred_score','trade_count']])
for sm in ('same_month_last_year','sarimax_seasonal_ar1'):
    for fm in ('last_available_quarter','ridge_lagged_stores'):
        x=base.merge(search[search.model==sm][['search_area','quarter','pred_search']],on=['search_area','quarter'],how='inner')
        x=x.merge(foot[foot.model==fm][['search_area','quarter','pred_foot','pred_trade_count']],on=['search_area','quarter'],how='inner')
        x=x[x.pred_trade_count==x.trade_count].copy()
        x['pred_footfall_per_trade']=x.pred_foot/x.trade_count
        x['pred_search_yoy']=(x.pred_search/x.search_prior_year-1).clip(-1,3)
        x['pred_foot_yoy']=(x.pred_footfall_per_trade/x.footfall_prior_year-1).clip(-1,3)
        x['pred_score']=(.35*percentile('search_interest_mean',x.pred_search)+
                         .15*percentile('search_interest_mean_yoy',x.pred_search_yoy)+
                         .35*percentile('footfall_per_trade',x.pred_footfall_per_trade)+
                         .15*percentile('footfall_per_trade_yoy',x.pred_foot_yoy))
        x['method']=sm+' + '+fm
        rows.append(x[['search_area','quarter','method','hotness_score','pred_score','trade_count']])
pred=pd.concat(rows,ignore_index=True)
pred.to_csv(OUT/'hotness_2025_backtest_predictions.csv',index=False,encoding='utf-8-sig')
result={}
for method,g in pred.groupby('method'):
    summaries=[]
    for q,z in g.groupby('quarter'):
        k=min(3,len(z))
        top_actual=set(z.nlargest(k,'hotness_score').search_area)
        top_pred=set(z.nlargest(k,'pred_score').search_area)
        summaries.append({'quarter':q,'n':len(z),'spearman':float(z.hotness_score.rank().corr(z.pred_score.rank())),
                          'precision_at_3':len(top_actual&top_pred)/k})
    result[method]={'observations':len(g),'places_per_quarter':[s['n'] for s in summaries],
                    'score_mae':round(float((g.hotness_score-g.pred_score).abs().mean()),3),
                    'mean_spearman':round(float(np.mean([s['spearman'] for s in summaries])),3),
                    'mean_precision_at_3':round(float(np.mean([s['precision_at_3'] for s in summaries])),3),
                    'quarters':summaries}
(OUT/'hotness_forecast_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))

"""Fixed-weight late fusion of independently forecast context streams.

Context is neighborhood/transit signal, not direct visits. We test 0% versus
30% context weight; weights are not fitted to the 2025 test set.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/semester_pilot'
SOURCE={'living':'living_weekend_day_night_ratio','mobility':'mobility_shop_tour_other_share',
        'subway':'subway_exits_daily'}
base=pd.read_csv(OUT/'named_place_hotness_score_quarterly.csv',dtype={'quarter':str})
base=base[base.score_valid].copy()
context=pd.read_csv(OUT/'named_place_context_monthly_2023_2025.csv',dtype={'quarter':str})
context=context[context.search_area.isin(base.search_area.unique())].copy()
q=context.groupby(['search_area','quarter'],as_index=False)[list(SOURCE.values())].mean()
q=q.sort_values(['search_area','quarter'])
for col in SOURCE.values():
    q[col+'_yoy']=(q[col]/q.groupby('search_area')[col].shift(4)-1).replace([np.inf,-np.inf],np.nan).clip(-1,3)
reference=q[(q.quarter>='20241')&(q.quarter<='20244')]
refs={col:np.sort(reference[col+'_yoy'].dropna().to_numpy()) for col in SOURCE.values()}
assert all(len(x)==40 for x in refs.values()),{k:len(v) for k,v in refs.items()}
def percentile(col,v):
    return np.searchsorted(refs[col],v,side='right')/len(refs[col])*100
for col in SOURCE.values():q[col+'_yoy_pct']=percentile(col,q[col+'_yoy'].to_numpy())
score=base.merge(q,on=['search_area','quarter'],how='inner',validate='one_to_one')
score=score[(score.quarter>='20241')&(score.quarter<='20254')].copy()
assert len(score)==80,len(score)
score['five_stream_score']=.7*score.hotness_score+.1*sum(score[col+'_yoy_pct'] for col in SOURCE.values())
score.to_csv(OUT/'five_stream_observed_score_2024_2025.csv',index=False,encoding='utf-8-sig')

base_pred=pd.read_csv(OUT/'hotness_2025_backtest_predictions.csv',dtype={'quarter':str})
base_pred=base_pred[base_pred.method=='same_month_last_year + last_available_quarter'][['search_area','quarter','pred_score']]
stream_pred=pd.read_csv(OUT/'context_stream_2025_backtest_predictions.csv')
stream_pred['quarter']=stream_pred.target_month.str[:4]+(((stream_pred.target_month.str[5:7].astype(int)-1)//3)+1).astype(str)
stream_pred=stream_pred.groupby(['search_area','quarter','stream','model'],as_index=False).agg(pred_level=('forecast','mean'),n_months=('target_month','nunique'))
stream_pred=stream_pred[stream_pred.n_months==3]
prior=q[['search_area','quarter']+list(SOURCE.values())].copy()
prior['quarter']=(prior.quarter.str[:4].astype(int)+1).astype(str)+prior.quarter.str[-1]
prior=prior.rename(columns={v:v+'_prior_year' for v in SOURCE.values()})
targets=score[score.quarter.str[:4]=='2025'].merge(base_pred,on=['search_area','quarter'],how='inner',validate='one_to_one')
targets=targets.merge(prior,on=['search_area','quarter'],how='left',validate='one_to_one')
rows=[]
for model in ('same_month_last_year','last_available_month','sarimax_seasonal_ar1'):
    z=targets.copy()
    for stream,col in SOURCE.items():
        p=stream_pred[(stream_pred.stream==stream)&(stream_pred.model==model)][['search_area','quarter','pred_level']]
        z=z.merge(p.rename(columns={'pred_level':col+'_pred'}),on=['search_area','quarter'],how='inner',validate='one_to_one')
        yoy=(z[col+'_pred']/z[col+'_prior_year']-1).clip(-1,3)
        z[col+'_pred_yoy_pct']=percentile(col,yoy.to_numpy())
    z['pred_five_stream_score']=.7*z.pred_score+.1*sum(z[col+'_pred_yoy_pct'] for col in SOURCE.values())
    z['context_forecast_model']=model
    rows.append(z[['search_area','quarter','context_forecast_model','hotness_score','five_stream_score','pred_score','pred_five_stream_score','sales_transactions','sales_reporting_trades','trade_count']])
pred=pd.concat(rows,ignore_index=True)
pred.to_csv(OUT/'five_stream_2025_late_fusion_predictions.csv',index=False,encoding='utf-8-sig')

def evaluate(g,target,forecast):
    quarter=[]
    for q,z in g.groupby('quarter'):
        k=min(3,len(z))
        top=set(z.nlargest(k,target).search_area)
        chosen=set(z.nlargest(k,forecast).search_area)
        quarter.append({'quarter':q,'n':len(z),'spearman':float(z[target].rank().corr(z[forecast].rank())),
                        'precision_at_3':len(top&chosen)/k})
    return {'n':len(g),'score_mae':round(float((g[target]-g[forecast]).abs().mean()),3),
            'mean_rank_spearman':round(float(np.mean([v['spearman'] for v in quarter])),3),
            'precision_at_3':round(float(np.mean([v['precision_at_3'] for v in quarter])),3),
            'quarters':quarter}

result={'two_stream_forecast':evaluate(targets,'hotness_score','pred_score'),
        'five_stream_forecast':{model:evaluate(g,'five_stream_score','pred_five_stream_score') for model,g in pred.groupby('context_forecast_model')},
        'five_stream_forecast_against_original_score':{model:evaluate(g,'hotness_score','pred_five_stream_score') for model,g in pred.groupby('context_forecast_model')}}
observed_sales=score[(score.quarter.str[:4]=='2025')&(score.sales_reporting_trades==score.trade_count)].copy()
observed_sales['sales_per_trade']=observed_sales.sales_transactions/observed_sales.trade_count
result['sales_external_validation']={}
for col in ('hotness_score','five_stream_score'):
    cs=[float(g[col].rank().corr(g.sales_per_trade.rank())) for _,g in observed_sales.groupby('quarter')]
    result['sales_external_validation'][col]={'n':len(observed_sales),'mean_spearman':round(float(np.mean(cs)),3),'by_quarter':[round(v,3) for v in cs]}
baseline_sales=targets[targets.sales_reporting_trades==targets.trade_count].copy()
baseline_sales['sales_per_trade']=baseline_sales.sales_transactions/baseline_sales.trade_count
base_corr=[float(z.pred_score.rank().corr(z.sales_per_trade.rank())) for _,z in baseline_sales.groupby('quarter')]
result['sales_external_validation']['pred_two_stream']={'n':len(baseline_sales),'mean_spearman':round(float(np.mean(base_corr)),3)}
for model,g in pred.groupby('context_forecast_model'):
    sales=g[g.sales_reporting_trades==g.trade_count].copy()
    sales['sales_per_trade']=sales.sales_transactions/sales.trade_count
    cs=[float(z.pred_five_stream_score.rank().corr(z.sales_per_trade.rank())) for _,z in sales.groupby('quarter')]
    result['sales_external_validation']['pred_five_'+model]={'n':len(sales),'mean_spearman':round(float(np.mean(cs)),3)}
(OUT/'five_stream_late_fusion_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))

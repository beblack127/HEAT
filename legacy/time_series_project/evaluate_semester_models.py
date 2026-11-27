"""Rolling forecasts: quarterly source availability lagged by two quarters."""
from __future__ import annotations
import json,sys,warnings
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent/'tmp/pythonlibs'))
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/semester_pilot'

def metric(df,actual='actual',forecast='forecast'):
    a=df[actual].to_numpy(float);f=df[forecast].to_numpy(float)
    return {'n':len(a),'mae':round(float(np.mean(np.abs(a-f))),3),
            'wape':round(float(np.sum(np.abs(a-f))/np.sum(np.abs(a))),4),
            'bias':round(float(np.mean(f-a)),3)}

def search_eval():
    raw=pd.read_csv(ROOT/'data/naver/areas_20261106T102026Z/monthly_search_interest_31areas.csv')
    raw=raw.sort_values(['search_area','month'])
    outputs=[]
    for area,g in raw.groupby('search_area'):
        y=g.search_interest.to_numpy(float)
        months=g.month.tolist()
        assert len(y)==36 and months[0]=='2023-01' and months[-1]=='2025-12'
        for origin in (24,27,30,33):
            hist=y[:origin]
            forecasts={'last_value':np.repeat(hist[-1],3),
                       'same_month_last_year':y[origin-12:origin-9]}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model=SARIMAX(hist,order=(1,0,0),seasonal_order=(0,1,0,12),
                                  trend='n',enforce_stationarity=False,enforce_invertibility=False)
                    forecasts['sarimax_seasonal_ar1']=np.maximum(0,model.fit(disp=False,maxiter=100).forecast(3))
            except Exception:
                pass
            for name,pred in forecasts.items():
                for h in range(3):
                    outputs.append({'search_area':area,'origin_month':months[origin-1],
                                    'target_month':months[origin+h],'horizon_months':h+1,
                                    'model':name,'actual':y[origin+h],'forecast':float(pred[h])})
    pred=pd.DataFrame(outputs)
    pred.to_csv(OUT/'search_3month_backtest_predictions.csv',index=False,encoding='utf-8-sig')
    return {name:metric(g) for name,g in pred.groupby('model')}

def trade_eval(target):
    df=pd.read_csv(OUT/'trade_food_quarterly_2023_2025.csv',dtype={'trade_code':str,'quarter':str})
    df=df.sort_values(['trade_code','quarter'])
    # Missing observations are unavailable, not zero. Use only complete series.
    valid=df.groupby('trade_code').filter(lambda x:len(x)==12 and x[target].notna().all())
    valid=valid.sort_values(['trade_code','quarter']).copy()
    group=valid.groupby('trade_code')
    for lag in (2,3,4):
        valid[f'log_target_lag{lag}']=np.log1p(group[target].shift(lag))
    for col in ('store_count','openings','closures'):
        valid[f'{col}_lag2']=group[col].shift(2)
    valid['quarter_index']=valid.quarter.str[-1].astype(int)
    features=['log_target_lag2','log_target_lag3','log_target_lag4','store_count_lag2','openings_lag2','closures_lag2','quarter_index']
    test=valid[valid.quarter.str[:4]=='2025'].copy()
    records=[]
    all_quarters=sorted(valid.quarter.unique())
    for target_q in ('20251','20252','20253','20254'):
        available_q=all_quarters[all_quarters.index(target_q)-2]
        train=valid[(valid.quarter<=available_q)&valid[features].notna().all(axis=1)]
        hold=test[(test.quarter==target_q)&test[features].notna().all(axis=1)]
        model=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
        model.fit(train[features],np.log1p(train[target]))
        pred=np.expm1(model.predict(hold[features])).clip(0)
        sales_only=['log_target_lag2','log_target_lag3','log_target_lag4','quarter_index']
        ablation=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
        ablation.fit(train[sales_only],np.log1p(train[target]))
        pred_without_stores=np.expm1(ablation.predict(hold[sales_only])).clip(0)
        for row,p,p0 in zip(hold.itertuples(),pred,pred_without_stores):
            records.extend([
                {'trade_code':row.trade_code,'target_quarter':target_q,'model':'ridge_lagged_stores','actual':getattr(row,target),'forecast':p},
                {'trade_code':row.trade_code,'target_quarter':target_q,'model':'ridge_target_lags_only','actual':getattr(row,target),'forecast':p0},
                {'trade_code':row.trade_code,'target_quarter':target_q,'model':'last_available_quarter','actual':getattr(row,target),'forecast':np.expm1(row.log_target_lag2)},
                {'trade_code':row.trade_code,'target_quarter':target_q,'model':'same_quarter_last_year','actual':getattr(row,target),'forecast':np.expm1(row.log_target_lag4)},
            ])
    result=pd.DataFrame(records)
    result.to_csv(OUT/f'trade_{target}_next_quarter_backtest_predictions.csv',index=False,encoding='utf-8-sig')
    return {name:metric(g) for name,g in result.groupby('model')},valid.trade_code.nunique()

def main():
    search=search_eval()
    trade,n=trade_eval('estimated_footfall')
    sales,n_sales=trade_eval('sales_transactions')
    result={'search_three_month':search,'trade_footfall_next_quarter':trade,'trade_sales_next_quarter':sales,
            'complete_trade_footfall_series':n,'complete_trade_sales_series':n_sales,
            'warning':'Search forecasts are for 31 named terms; footfall forecasts are for trade codes. A joined hotspot forecast requires a separate held-out evaluation.'}
    (OUT/'model_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

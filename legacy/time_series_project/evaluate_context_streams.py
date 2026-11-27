"""Forecast each contextual stream independently, with observed release lags."""
from __future__ import annotations
import json,sys,warnings
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tmp/pythonlibs'))
from statsmodels.tsa.statespace.sarimax import SARIMAX
OUT=ROOT/'results/semester_pilot'
SERIES={'living':'living_weekend_day_night_ratio','mobility':'mobility_shop_tour_other_share',
        'subway':'subway_exits_daily'}
df=pd.read_csv(OUT/'named_place_context_monthly_2023_2025.csv')
assert df.month.nunique()==36
rows=[]
for stream,col in SERIES.items():
    for area,g in df.sort_values('month').groupby('search_area'):
        g=g.sort_values('month')
        y=g[col].to_numpy(float)
        months=g.month.tolist()
        if len(y)!=36 or not np.isfinite(y).all():continue
        for origin in (24,27,30,33):
            # At a quarter boundary, the last published monthly value is two months earlier.
            history=y[:origin-1]
            preds={'last_available_month':np.repeat(history[-1],3),
                   'same_month_last_year':y[origin-12:origin-9]}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fit=SARIMAX(history,order=(1,0,0),seasonal_order=(0,1,0,12),trend='n',
                                enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=100)
                    # Forecast December/Mar/Jun/Sep first, since that month is not published yet.
                    preds['sarimax_seasonal_ar1']=np.maximum(0,fit.forecast(4)[1:4])
            except Exception:pass
            for model,pred in preds.items():
                for h in range(3):
                    rows.append({'stream':stream,'search_area':area,'origin_month':months[origin-1],
                                 'target_month':months[origin+h],'horizon_months':h+1,
                                 'model':model,'actual':y[origin+h],'forecast':float(pred[h])})
pred=pd.DataFrame(rows)
pred.to_csv(OUT/'context_stream_2025_backtest_predictions.csv',index=False,encoding='utf-8-sig')
result={}
for (stream,model),g in pred.groupby(['stream','model']):
    a=g.actual.to_numpy();p=g.forecast.to_numpy()
    result.setdefault(stream,{})[model]={'n':len(g),'areas':g.search_area.nunique(),
        'mae':round(float(np.abs(a-p).mean()),4),'wape':round(float(np.abs(a-p).sum()/np.abs(a).sum()),4),
        'bias':round(float((p-a).mean()),4)}
(OUT/'context_stream_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))

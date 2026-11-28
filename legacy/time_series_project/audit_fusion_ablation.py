"""Post-hoc single-stream additions, for diagnosis only, not weight selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent/'results/semester_pilot'
x=pd.read_csv(OUT/'five_stream_observed_score_2024_2025.csv',dtype={'quarter':str})
x=x[x.sales_reporting_trades==x.trade_count].copy()
x['sales_per_trade']=x.sales_transactions/x.trade_count
cols={'living':'living_weekend_day_night_ratio_yoy_pct','mobility':'mobility_shop_tour_other_share_yoy_pct',
      'subway':'subway_exits_daily_yoy_pct'}
scores={'original_two':x.hotness_score,'all_three_context':x.five_stream_score}
for name,col in cols.items():scores['add_'+name]=.9*x.hotness_score+.1*x[col]
result={}
for year in ('2024','2025'):
    result[year]={}
    for name,s in scores.items():
        z=x[x.quarter.str[:4]==year].assign(score=s)
        corrs=[float(g.score.rank().corr(g.sales_per_trade.rank())) for _,g in z.groupby('quarter')]
        top=[]
        for _,g in z.groupby('quarter'):
            actual=set(g.nlargest(3,'sales_per_trade').search_area)
            pred=set(g.nlargest(3,'score').search_area)
            top.append(len(actual&pred)/3)
        result[year][name]={'n':len(z),'sales_mean_spearman':round(float(np.mean(corrs)),3),
                            'sales_precision_at_3':round(float(np.mean(top)),3)}
result['warning']='2024 can inform a pre-specified weight choice; 2025 is the held-out check. Small samples and provisional geography remain.'
(OUT/'five_stream_ablation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))

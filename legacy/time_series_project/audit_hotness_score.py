"""Independent sales sanity check and threshold/weight sensitivity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent/'results/semester_pilot'
x=pd.read_csv(OUT/'named_place_hotness_score_quarterly.csv',dtype={'quarter':str})
x=x[x.score_valid].copy()
sales=x[x.sales_reporting_trades==x.trade_count].copy()
sales['sales_per_trade']=sales.sales_transactions/sales.trade_count
corr={q:float(g.hotness_score.rank().corr(g.sales_per_trade.rank())) for q,g in sales.groupby('quarter') if len(g)>=5}
weights={}
for q,g in x.groupby('quarter'):
    base=set(g.nlargest(3,'hotness_score').search_area)
    weights[q]={col:len(base&set(g.nlargest(3,col).search_area))/3 for col in ('score_search_40','score_search_60')}
thresholds={}
for threshold in (75,80,85):
    z=x.sort_values(['search_area','quarter']).copy()
    z['hit']=(z.hotness_score>=threshold)&z.component_floor_ok
    z['persistent']=z.hit&z.groupby('search_area').hit.shift(1).fillna(False)
    thresholds[str(threshold)]={'quarters_above':int(z.hit.sum()),'persistent_quarters':int(z.persistent.sum()),
                                'persistent_2025':int(z[z.quarter.str[:4]=='2025'].persistent.sum())}
result={'sales_validation':{'n_place_quarters':len(sales),'mean_cross_section_spearman':round(float(np.mean(list(corr.values()))),3),
                            'quarter_correlations':{k:round(v,3) for k,v in corr.items()}},
        'search_weight_sensitivity':{'mean_top3_overlap_40pct':round(float(np.mean([v['score_search_40'] for v in weights.values()])),3),
                                     'mean_top3_overlap_60pct':round(float(np.mean([v['score_search_60'] for v in weights.values()])),3)},
        'thresholds':thresholds,
        'notes':'Sales is independent of score inputs but correlational, not a ground truth of popularity. Sensitivity is within the same provisional 10-area geography.'}
(OUT/'score_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))

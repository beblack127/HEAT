"""Area-weighted neighborhood context from dongs plus nearest station exits."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
IN=ROOT/'data/context_monthly'
OUT=ROOT/'results/semester_pilot'
bridge=pd.read_csv(ROOT/'data/place_dong_crosswalk/place_to_admin_dong_overlap.csv',dtype={'admin_dong_code_202506':str})
audit=pd.read_csv(ROOT/'data/place_dong_crosswalk/place_31_mapping_audit.csv')
valid=set(audit[audit.training_status!='exclude_shared_polygon'].search_area)
bridge=bridge[(bridge.selected_by_5pct_rule==1)&bridge.search_area.isin(valid)].copy()
bridge['place_weight']=bridge.place_area_share/bridge.groupby('search_area').place_area_share.transform('sum')
bridge=bridge.rename(columns={'admin_dong_code_202506':'admin_dong_code'})
living=pd.concat([pd.read_csv(p,dtype={'admin_dong_code':str}) for p in sorted(IN.glob('living_202*.csv'))],ignore_index=True)
mobility=pd.concat([pd.read_csv(p,dtype={'admin_dong_code':str}) for p in sorted(IN.glob('mobility_202*.csv'))],ignore_index=True)
assert living.month.nunique()==36 and mobility.month.nunique()==36
context=bridge[['search_area','admin_dong_code','place_weight']].merge(living,on='admin_dong_code',how='inner')
context=context.merge(mobility,on=['month','admin_dong_code'],how='inner',validate='many_to_one')
context['living_weekend_day_night_ratio']=context.living_weekend_day_mean/context.living_night_mean
context['living_day_night_ratio']=context.living_day_mean/context.living_night_mean
context['mobility_shop_tour_other_share']=context.mobility_shop_tour_other_daily/context.mobility_all_arrivals_daily
for col in ('living_weekend_day_night_ratio','living_day_night_ratio','mobility_shop_tour_daily','mobility_shop_tour_other_daily','mobility_all_arrivals_daily','mobility_shop_tour_other_share'):
    context[col+'_weighted']=context[col]*context.place_weight
area=context.groupby(['search_area','month'],as_index=False).agg(
    living_weekend_day_night_ratio=('living_weekend_day_night_ratio_weighted','sum'),
    living_day_night_ratio=('living_day_night_ratio_weighted','sum'),
    mobility_shop_tour_daily=('mobility_shop_tour_daily_weighted','sum'),
    mobility_shop_tour_other_daily=('mobility_shop_tour_other_daily_weighted','sum'),
    mobility_all_arrivals_daily=('mobility_all_arrivals_daily_weighted','sum'),
    mobility_shop_tour_other_share=('mobility_shop_tour_other_share_weighted','sum'),
    contributing_dongs=('admin_dong_code','nunique'))
subway=pd.read_csv(IN/'subway_place_monthly_provisional.csv')
subway['days_in_month']=pd.to_datetime(subway.month+'-01').dt.days_in_month
subway['subway_exits_daily']=subway.subway_exits/subway.days_in_month
subway['subway_entries_daily']=subway.subway_entries/subway.days_in_month
area=area.merge(subway[['search_area','month','subway_exits_daily','subway_entries_daily','station_lines']],on=['search_area','month'],how='left',validate='one_to_one')
area['quarter']=area.month.str[:4]+(((area.month.str[5:7].astype(int)-1)//3)+1).astype(str)
area=area.sort_values(['search_area','month'])
area.to_csv(OUT/'named_place_context_monthly_2023_2025.csv',index=False,encoding='utf-8-sig')
print({'places':area.search_area.nunique(),'months':area.month.nunique(),'rows':len(area),
       'subway_complete_places':int((area.groupby('search_area').subway_exits_daily.count()==36).sum())})

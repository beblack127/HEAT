"""Package raw summaries, spatial bridges, forecasts, and scores in SQLite."""
import json,sqlite3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/semester_pilot'
CTX=ROOT/'data/context_monthly'
db=OUT/'five_stream_pilot.sqlite'
sources=[
 ('naver_search','https://api.ncloud-docs.com/docs/naver-api-hub-search-trend','named place','month','relative keyword-search ratio'),
 ('commercial_footfall','https://data.seoul.go.kr/dataList/OA-15568/S/1/datasetView.do','trade area','quarter','estimated passerby count'),
 ('living_population','https://data.seoul.go.kr/dataList/OA-14991/F/1/datasetView.do','administrative dong','hour -> month','nearby population context; not visits'),
 ('living_mobility','https://data.seoul.go.kr/dataList/OA-22298/F/1/datasetView.do','destination administrative dong','day-hour -> month','purpose 4 shopping, 5 tourism, 7 other; broad context'),
 ('subway','https://data.seoul.go.kr/dataList/OA-12914/F/1/datasetView.do','station','day -> month','station exits are transit context; not place visits'),
 ('stores','https://data.seoul.go.kr/dataList/OA-15577/F/1/datasetView.do','trade area','quarter','food store counts/openings/closures'),
 ('sales','https://data.seoul.go.kr/dataList/OA-15572/F/1/datasetView.do','trade area','quarter','independent commercial activity check'),
]
tables={
 'naver_search_monthly':ROOT/'data/naver/areas_20261106T102026Z/monthly_search_interest_31areas.csv',
 'trade_food_quarterly':OUT/'trade_food_quarterly_2023_2025.csv',
 'search_forecasts_2025':OUT/'search_3month_backtest_predictions.csv',
 'trade_footfall_forecasts_2025':OUT/'trade_estimated_footfall_next_quarter_backtest_predictions.csv',
 'trade_sales_forecasts_2025':OUT/'trade_sales_transactions_next_quarter_backtest_predictions.csv',
 'place_trade_bridge':OUT/'place_trade_bridge_provisional.csv',
 'place_dong_bridge':ROOT/'data/place_dong_crosswalk/place_to_admin_dong_overlap.csv',
 'place_dong_mapping_audit':ROOT/'data/place_dong_crosswalk/place_31_mapping_audit.csv',
 'place_subway_bridge':CTX/'subway_place_bridge_provisional.csv',
 'living_dong_monthly':None,
 'mobility_dong_monthly':None,
 'subway_station_monthly':CTX/'subway_station_monthly_2023_2025.csv',
 'subway_place_monthly':CTX/'subway_place_monthly_provisional.csv',
 'place_context_monthly':OUT/'named_place_context_monthly_2023_2025.csv',
 'place_quarterly':OUT/'named_place_quarterly_provisional.csv',
 'hotness_two_stream':OUT/'named_place_hotness_score_quarterly.csv',
 'hotness_two_stream_forecasts_2025':OUT/'hotness_2025_backtest_predictions.csv',
 'hotness_five_stream':OUT/'five_stream_observed_score_2024_2025.csv',
 'context_forecasts_2025':OUT/'context_stream_2025_backtest_predictions.csv',
 'late_fusion_forecasts_2025':OUT/'five_stream_2025_late_fusion_predictions.csv',
}
code_cols={'trade_code':str,'admin_dong_code':str,'admin_dong_code_202506':str,'quarter':str}
with sqlite3.connect(db) as conn:
    for name,path in tables.items():
        if name=='living_dong_monthly':
            frame=pd.concat([pd.read_csv(p,dtype=code_cols) for p in sorted(CTX.glob('living_202*.csv'))],ignore_index=True)
        elif name=='mobility_dong_monthly':
            frame=pd.concat([pd.read_csv(p,dtype=code_cols) for p in sorted(CTX.glob('mobility_202*.csv'))],ignore_index=True)
        else:frame=pd.read_csv(path,dtype=code_cols)
        frame.to_sql(name,conn,if_exists='replace',index=False)
        print(name,len(frame))
    pd.DataFrame(sources,columns=['source_id','url','native_geography','native_time','use_note']).to_sql('source_manifest',conn,if_exists='replace',index=False)
    indexes={'trade_food_quarterly':'trade_code,quarter','place_quarterly':'search_area,quarter',
             'naver_search_monthly':'search_area,month',
             'place_context_monthly':'search_area,month','living_dong_monthly':'admin_dong_code,month',
             'mobility_dong_monthly':'admin_dong_code,month','subway_station_monthly':'station_name,line,month',
             'hotness_two_stream':'search_area,quarter','hotness_five_stream':'search_area,quarter'}
    for table,cols in indexes.items():conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table} ON {table} ({cols})')
    conn.commit()
    integrity=conn.execute('PRAGMA integrity_check').fetchone()[0]
    if integrity!='ok':raise RuntimeError(integrity)
print(db,db.stat().st_size)

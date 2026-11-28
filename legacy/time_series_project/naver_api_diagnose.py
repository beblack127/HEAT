"""Temporary interactive diagnosis. Credentials stay in process memory only."""
import copy
import http.client
import json
import threading
import tkinter as tk
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from datetime import datetime, timezone
from naver_search_test import API_URL, PAYLOAD

OUT = Path(__file__).resolve().parent / 'tmp' / 'naver_diagnosis.json'
OUT.parent.mkdir(exist_ok=True)
root = tk.Tk()
root.title('NAVER API 연결 진단')
root.geometry('560x300')
tk.Label(root, text='API HUB 인증 정보를 입력하세요. 키는 저장하지 않습니다.').pack(pady=12)
tk.Label(root, text='Client ID').pack()
cid = tk.Entry(root, width=65, show='*'); cid.pack()
tk.Label(root, text='Client Secret').pack()
secret = tk.Entry(root, width=65, show='*'); secret.pack()
status = tk.StringVar(value='최대 5회 요청을 비교합니다. 완료 후 이 창을 닫아도 됩니다.')
tk.Label(root, textvariable=status, wraplength=530).pack(pady=15)

def diagnose(a, b):
    results=[]
    def save():
        OUT.write_text(json.dumps({'timestamp':datetime.now(timezone.utc).isoformat(), 'results':results},ensure_ascii=False,indent=2),encoding='utf-8')
    def call(name, payload, transport):
        headers={'X-NCP-APIGW-API-KEY-ID':a,'X-NCP-APIGW-API-KEY':b,'Content-Type':'application/json'}
        body=json.dumps(payload,ensure_ascii=False).encode('utf-8')
        try:
            if transport=='http.client':
                conn=http.client.HTTPSConnection('naverapihub.apigw.ntruss.com',timeout=30)
                try:
                    conn.request('POST','/search-trend/v1/search',body=body,headers=headers)
                    res=conn.getresponse(); code=res.status; rh=dict(res.getheaders()); raw=res.read()
                finally: conn.close()
            else:
                try: res=urlopen(Request(API_URL,data=body,headers=headers,method='POST'),timeout=30)
                except HTTPError as e: res=e
                with res:
                    code=res.code; rh=dict(res.headers); raw=res.read()
            decoded=raw.decode('utf-8',errors='replace')
            for key in (a,b): decoded=decoded.replace(key,'[REDACTED]')
            safe_headers={k:v for k,v in rh.items() if k.lower() in ('server','content-type','content-length','x-ncp-trace-id','x-ncp-apigw-response-origin')}
            record={'name':name,'transport':transport,'status':code,'headers':safe_headers,'body':decoded[:12000]}
        except Exception as exc:
            msg=str(exc)
            for key in (a,b):msg=msg.replace(key,'[REDACTED]')
            record={'name':name,'error':msg}
        results.append(record);save()
        return record.get('status')==200
    try:
        call('full_original',PAYLOAD,'urllib')
        call('full_preserve_header_case',PAYLOAD,'http.client')
        short=copy.deepcopy(PAYLOAD);short['endDate']='2023-01-31';short['keywordGroups']=short['keywordGroups'][:1]
        call('one_month_one_group',short,'urllib')
        options=copy.deepcopy(short);options.update(device='pc',gender='f',ages=['3','4','5','6','7','8','9','10','11'])
        call('explicit_optional_fields',options,'urllib')
        empty=copy.deepcopy(PAYLOAD);empty.update(device='',gender='',ages=[])
        call('all_devices_explicit_empty',empty,'urllib')
        root.after(0,lambda:status.set('진단 완료. 키는 파일에 저장하지 않았습니다. 이 창을 닫아주세요.'))
    finally:
        a=b=None

def start():
    a,b=cid.get().strip(),secret.get().strip()
    if not a or not b or any(ord(c)<33 or ord(c)>126 for c in a+b):
        status.set('두 키에 공백·제어문자 없이 입력하세요.');return
    cid.delete(0,'end');secret.delete(0,'end');button.config(state='disabled')
    status.set('비교 요청 중입니다. 약 1분 정도 기다려주세요.')
    threading.Thread(target=diagnose,args=(a,b),daemon=True).start()

button=tk.Button(root,text='진단 시작',command=start);button.pack()
root.mainloop()

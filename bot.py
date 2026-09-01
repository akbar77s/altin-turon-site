"""
Altin Signal — ko'p bozorli bot (AQSh + Yevropa + Osiyo).
O'rta/kichik kompaniyalar (volatilroq). 30m/1h/4h. Har soat.
Belgi formatlari: AQSh oddiy, Yevropa .DE/.PA/.L/.AS/.MI/.MC/.SW, Osiyo .T/.HK/.KS/.TW/.NS/.AX.
Birinchi ishga tushganda har bozor bo'yicha nechta aksiya ma'lumot bergani ko'rinadi.
"""
import os
import io
import re
import json
import time
import requests
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass

TOKEN = os.environ.get("TG_TOKEN", "")
CHAT_ID = os.environ.get("TG_CHAT_ID", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

TFS = {"30m": ("30m", "15d", False),
       "1h":  ("60m", "60d", False),
       "4h":  ("60m", "120d", True)}
STRENGTH = 3
TP_MULT = 1.618
CHUNK = 60
SENT_FILE = "sent.json"

EUROPE = [
    "RHM.DE","PUM.DE","SY1.DE","ZAL.DE","HFG.DE","LHA.DE","EVK.DE","AIXA.DE","SDF.DE",
    "PSM.DE","WCH.DE","KGX.DE","BOSS.DE","NDA.DE","FRA.DE","DUE.DE","GXI.DE","NEM.DE",
    "ALO.PA","UBI.PA","ML.PA","ERF.PA","RCO.PA","SESG.PA","VLA.PA","AKE.PA","BEN.PA",
    "WEIR.L","HWDN.L","BME.L","MRO.L","IMI.L","BAB.L","QQ.L","CNA.L","ITV.L","EZJ.L",
    "TKWY.AS","BESI.AS","AMG.AS","FUR.AS","ARCAD.AS",
    "LDO.MI","PIRC.MI","BAMI.MI","BMED.MI","IG.MI",
    "SCYR.MC","CIE.MC","MAP.MC","GEST.MC",
    "TEMN.SW","GALE.SW","BARN.SW","SOON.SW",
]

ASIA = [
    "6857.T","6146.T","7741.T","6920.T","6981.T","6762.T","4519.T","4568.T","6098.T",
    "6367.T","9843.T","4661.T","7269.T","6503.T","6273.T","4063.T","6954.T","7011.T",
    "0175.HK","1810.HK","0968.HK","9868.HK","2015.HK","1024.HK","9618.HK","9888.HK",
    "000660.KS","035420.KS","051910.KS","006400.KS","207940.KS","068270.KS","003670.KS",
    "2454.TW","3008.TW","2379.TW","3711.TW","2308.TW","6669.TW",
    "DIXON.NS","POLYCAB.NS","COFORGE.NS","MPHASIS.NS","PERSISTENT.NS","TATAELXSI.NS",
    "LTTS.NS","CDSL.NS","LTIM.NS","BSE.NS",
    "PLS.AX","WTC.AX","XRO.AX","PME.AX","ALU.AX","ALL.AX","REA.AX",
]

FALLBACK_US = ["AAPL","MSFT","NVDA","AMD","AVGO","CRM","ADBE","QCOM","NFLX","UBER"]

@dataclass
class Signal:
    index: int; close: float
    peak_top_1: float; peak_top_2: float; peak_idx_1: int; peak_idx_2: int

def _tops(df):
    return df[["open", "close"]].max(axis=1).astype(float).tolist()

def find_peaks(tops, upto, s):
    return [m for m in range(s, upto - s)
            if all(tops[m] > tops[m - k] for k in range(1, s + 1))
            and all(tops[m] > tops[m + k] for k in range(1, s + 1))]

def check(df, i, s=3):
    n = len(df)
    if i < 0: i = n + i
    if i < 2 * s + 2: return None
    cur = df.iloc[i]
    if float(cur["close"]) <= float(cur["open"]): return None
    ci = float(cur["close"]); tops = _tops(df)
    pk = find_peaks(tops, i, s)
    if len(pk) < 2: return None
    k1, k2 = pk[-1], pk[-2]
    if not (tops[k2] > tops[k1]): return None
    if ci > tops[k1] and ci > tops[k2]:
        return Signal(i, ci, tops[k1], tops[k2], k1, k2)
    return None

def zero_for(df, i, floor_idx):
    r = None
    for j in range(i - 1, floor_idx - 1, -1):
        if float(df["close"].iloc[j]) < float(df["open"].iloc[j]):
            r = j; break
    if r is None: r = max(floor_idx, i - 3)
    lo = float(df["low"].iloc[r])
    for j in range(r + 1, i): lo = min(lo, float(df["low"].iloc[j]))
    return lo

def us_midsmall(limit=200):
    got = []
    for url in ["https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"]:
        try:
            for tbl in pd.read_html(url):
                col = next((c for c in tbl.columns if str(c).lower() in ("symbol", "ticker", "ticker symbol")), None)
                if col is not None:
                    got += [str(s).replace(".", "-").strip() for s in tbl[col].tolist()]; break
        except Exception as e: print("US ro'yxat xato:", e)
    got = [s for s in got if s and s != "nan"]
    return got[:limit] if got else FALLBACK_US

def build_universe():
    us = us_midsmall(200)
    return ([(t, "AQSh") for t in us] + [(t, "Yevropa") for t in EUROPE] + [(t, "Osiyo") for t in ASIA])

def normalize(raw):
    if raw is None or raw.empty: return None
    df = raw.reset_index(); df.columns = [str(c).lower() for c in df.columns]
    tcol = "datetime" if "datetime" in df.columns else "date"
    need = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in need): return None
    return df.rename(columns={tcol: "date"})[["date"] + need].dropna().reset_index(drop=True)

def to_4h(df):
    return (df.set_index("date").resample("4h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna().reset_index())

def batch_download(tickers, interval, period):
    out = {}
    for a in range(0, len(tickers), CHUNK):
        chunk = tickers[a:a+CHUNK]
        try: raw = yf.download(chunk, period=period, interval=interval, auto_adjust=True, progress=False, group_by="ticker", threads=True)
        except Exception as e: print(f"  batch xato: {e}"); continue
        for t in chunk:
            try:
                sub = raw[t] if len(chunk) > 1 else raw; df = normalize(sub)
                if df is not None and len(df) > 2 * STRENGTH + 5: out[t] = df
            except Exception: continue
        time.sleep(1)
    return out

def send(text):
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id":CHAT_ID,"text":text}, timeout=15)
        return r.status_code == 200
    except Exception as e: print("Telegram xato:", e); return False

def load_sent():
    try:
        with open(SENT_FILE) as f: return set(json.load(f))
    except Exception: return set()

def save_sent(sent):
    with open(SENT_FILE,"w") as f: json.dump(sorted(sent)[-8000:],f)

def fmt_no(n): return f"#{int(n):05d}" if n else "#—"

def _supa_headers():
    return {"apikey":SUPABASE_KEY,"Authorization":"Bearer "+SUPABASE_KEY,"Content-Type":"application/json"}

def save_supabase(row):
    if not SUPABASE_URL or not SUPABASE_KEY: return None
    try:
        r=requests.post(SUPABASE_URL.rstrip("/")+"/rest/v1/signals",headers={**_supa_headers(),"Prefer":"return=representation"},json=row,timeout=15)
        if r.status_code in (200,201):
            data=r.json(); return data[0].get("signal_no") if data else None
        print("Supabase xato:",r.status_code,r.text[:200])
    except Exception as e: print("Supabase xato:",e)
    return None

def has_open(ticker,timeframe):
    if not SUPABASE_URL or not SUPABASE_KEY: return False
    try:
        url=SUPABASE_URL.rstrip("/")+f"/rest/v1/signals?select=id&status=eq.open&ticker=eq.{ticker}&timeframe=eq.{timeframe}&limit=1"
        r=requests.get(url,headers=_supa_headers(),timeout=15)
        if r.status_code==200:return len(r.json())>0
    except Exception as e: print("has_open xato:",e)
    return False

def get_open_signals():
    if not SUPABASE_URL or not SUPABASE_KEY:return []
    try:
        url=SUPABASE_URL.rstrip("/")+"/rest/v1/signals?select=id,signal_no,ticker,timeframe,entry,tp,sl,region&status=eq.open&limit=1000"
        r=requests.get(url,headers=_supa_headers(),timeout=20)
        if r.status_code==200:return r.json()
    except Exception as e:print("get_open_signals xato:",e)
    return []

def update_status(sig_id,status):
    if not SUPABASE_URL or not SUPABASE_KEY:return
    try: requests.patch(SUPABASE_URL.rstrip("/")+f"/rest/v1/signals?id=eq.{sig_id}",headers={**_supa_headers(),"Prefer":"return=minimal"},json={"status":status},timeout=15)
    except Exception as e:print("update_status xato:",e)

def last_price(ticker):
    try:
        raw=yf.download(ticker,period="5d",interval="60m",auto_adjust=True,progress=False)
        if isinstance(raw.columns,pd.MultiIndex):raw.columns=raw.columns.get_level_values(0)
        raw.columns=[str(c).lower() for c in raw.columns]
        if "close" in raw.columns and len(raw)>0:return float(raw["close"].dropna().iloc[-1])
    except Exception:pass
    return None

def monitor_open_signals():
    opens=get_open_signals()
    if not opens:print("Kuzatuv: ochiq signal yo'q");return
    print(f"Kuzatuv: {len(opens)} ochiq signal tekshirilmoqda");changed=0
    for s in opens:
        price=last_price(s["ticker"])
        if price is None:continue
        entry=float(s["entry"]);tp=float(s["tp"]);sl=float(s["sl"]);new_status=None
        if price>=tp:new_status="tp"
        elif price<=sl:new_status="sl"
        if new_status:
            update_status(s["id"],new_status);changed+=1
            emoji="🎯" if new_status=="tp" else "🛑";label="FOYDA (TP)" if new_status=="tp" else "STOP (SL)"
            send(f"{emoji} {fmt_no(s.get('signal_no'))} {s['ticker']} [{s['timeframe']}] — {label}\nKirish: {entry:.2f} → hozir: {price:.2f}")
        time.sleep(.3)
    print(f"Kuzatuv tugadi: {changed} ta signal holati o'zgardi")

def weekly_report():
    if not SUPABASE_URL or not SUPABASE_KEY:print("Hisobot: Supabase yo'q");return
    try:
        from datetime import datetime,timedelta,timezone
        week_ago=(datetime.now(timezone.utc)-timedelta(days=7)).isoformat()
        url=SUPABASE_URL.rstrip("/")+f"/rest/v1/signals?select=status,rr&created_at=gte.{week_ago}&limit=5000"
        r=requests.get(url,headers=_supa_headers(),timeout=20);rows=r.json() if r.status_code==200 else []
    except Exception as e:print("Hisobot xato:",e);rows=[]
    total=len(rows);tp=sum(1 for x in rows if x.get("status")=="tp");sl=sum(1 for x in rows if x.get("status")=="sl");still_open=sum(1 for x in rows if x.get("status")=="open");closed=tp+sl;winrate=round(tp/closed*100) if closed else 0;net_r=sum(float(x.get("rr") or 0) for x in rows if x.get("status")=="tp")-sl
    msg=f"📊 HAFTALIK HISOBOT\n\nJami signal: {total}\n🎯 Foyda (TP): {tp}\n🛑 Zarar (SL): {sl}\n⏳ Ochiq: {still_open}\nG'alaba: {winrate}%\nTaxminiy natija: {net_r:+.1f}R"
    send(msg);print("Hisobot yuborildi:",msg.replace("\n"," | "))

def render_chart(df,i):
    sub=df.iloc[max(0,i-14):i+1].reset_index(drop=True);fig,ax=plt.subplots(figsize=(4.2,2.6),dpi=100);fig.patch.set_facecolor("#0C1830");ax.set_facecolor("#0C1830")
    for x in range(len(sub)):
        o=float(sub["open"].iloc[x]);c=float(sub["close"].iloc[x]);h=float(sub["high"].iloc[x]);l=float(sub["low"].iloc[x]);col="#16A0A8" if c>=o else "#E2726E";ax.plot([x,x],[l,h],color=col,linewidth=.9);height=abs(c-o) or (h-l)*.002;ax.add_patch(plt.Rectangle((x-.3,min(o,c)),.6,height,color=col))
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    ax.margins(x=.03,y=.12);buf=io.BytesIO();plt.savefig(buf,format="png",facecolor=fig.get_facecolor(),bbox_inches="tight");plt.close(fig);return buf.getvalue()

def upload_chart(png,name):
    if not SUPABASE_URL or not SUPABASE_KEY:return None
    try:
        r=requests.post(SUPABASE_URL.rstrip("/")+"/storage/v1/object/charts/"+name,headers={"apikey":SUPABASE_KEY,"Authorization":"Bearer "+SUPABASE_KEY,"Content-Type":"image/png","x-upsert":"true"},data=png,timeout=25)
        if r.status_code in (200,201):return SUPABASE_URL.rstrip("/")+"/storage/v1/object/public/charts/"+name
        print("Storage xato:",r.status_code,r.text[:200])
    except Exception as e:print("Storage xato:",e)
    return None

def send_photo(png,caption):
    try:
        r=requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",data={"chat_id":CHAT_ID,"caption":caption},files={"photo":("chart.png",png,"image/png")},timeout=30);return r.status_code==200
    except Exception as e:print("Telegram photo xato:",e);return False

def main():
    if not TOKEN or not CHAT_ID:print("TG_TOKEN / TG_CHAT_ID yo'q.");return
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="report":print("Haftalik hisobot rejimi");weekly_report();return
    universe=build_universe();region_of={t:r for t,r in universe};tickers=[t for t,_ in universe]
    print(f"Jami {len(tickers)} aksiya: AQSh {sum(1 for _,r in universe if r=='AQSh')}, Yevropa {sum(1 for _,r in universe if r=='Yevropa')}, Osiyo {sum(1 for _,r in universe if r=='Osiyo')}")
    monitor_open_signals();sent=load_sent();new_count=0
    for tf,(interval,period,resample) in TFS.items():
        print(f"--- {tf} ---");data=batch_download(tickers,interval,period);by_reg={}
        for t in data:
            reg=region_of.get(t,"?");by_reg[reg]=by_reg.get(reg,0)+1
        print(f"  ma'lumot olindi: {len(data)} ({by_reg})")
        for ticker,df in data.items():
            if resample:df=to_4h(df)
            if len(df)<2*STRENGTH+5:continue
            i=len(df)-2;sig=check(df,i,STRENGTH)
            if sig is None:continue
            entry=float(sig.close);sl=zero_for(df,i,sig.peak_idx_2)
            if entry<=sl or has_open(ticker,tf):continue
            key=f"{ticker}_{tf}_{df['date'].iloc[i]}"
            if key in sent:continue
            tp=sl+TP_MULT*(entry-sl);rr=(tp-entry)/(entry-sl);reg=region_of.get(ticker,"");png=None;chart_url=None
            try:
                png=render_chart(df,i);fname=re.sub(r'[^A-Za-z0-9]','_',ticker)+f"_{tf}_{int(time.time()*1000)}.png";chart_url=upload_chart(png,fname)
            except Exception as e:print("Grafik xato:",e)
            signal_no=save_supabase({"ticker":ticker,"region":reg,"timeframe":tf,"entry":round(entry,2),"tp":round(tp,2),"sl":round(sl,2),"rr":round(rr,2),"status":"open","signal_time":str(df['date'].iloc[i]),"chart_url":chart_url})
            msg=f"🔔 SIGNAL {fmt_no(signal_no)} — {ticker} [{tf}] · {reg}\nRazvorot modeli\n\n➕ Entry: {entry:.2f}\n🎯 TP (1.618): {tp:.2f}\n🛑 SL (0): {sl:.2f}\nR:R {rr:.2f}\nVaqt: {df['date'].iloc[i]}"
            ok=send_photo(png,msg) if png else send(msg)
            if ok:sent.add(key);new_count+=1;print(f"  Yuborildi: {fmt_no(signal_no)} {ticker} [{tf}] {reg}")
    save_sent(sent);print(f"Tugadi. {new_count} ta yangi signal.")

if __name__=="__main__":main()

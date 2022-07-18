import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import datetime
from app.utils import DateUtil

## custom libs
import st_utils, utils

plt.rc('font', family='Malgun Gothic')

st.title(':chart_with_upwards_trend: MyData 기획 POC 대시보드')
st.markdown('---')

APPS = [
    '1. 환율 변동에 따른 원화자산/외화자산 균형',
    '2. 환율 변동에 따른 외화자산 직접투자 환차익/환손실 계산',
    '3. 주식 종목 간 상관관계에 기반한 유사종목 추천',
    '4. 기 보유한 펀드 포트폴리오와 상관관계 낮은 펀드 추천',
    # '5. 주식 종목 상관관계 클러스터링에 기반한 '
    ]

with st.sidebar:
    dropbox = st.selectbox('Select app', APPS)

today = datetime.date.today()
today_int = DateUtil.timestamp_2_intDate(today)
today_str = DateUtil.numdate2stddate(today_int)

if dropbox == APPS[0]:
    st.header(APPS[0])
    st.info('''
    - 고객이 가진 (달러)외화자산의 가치가 원달러 환율 변동에 의해 얼마나 변하는지 보여줍니다.
    - 고객 전체 자산에서 원화자산 대비 외화자산이 많을 수록 환율변동에 자산가치가 크게 노출됩니다.
    - 환율이 10원 변할 때 고객의 전체 자산가치가 몇%나 영향을 받는지 보여줍니다. 
    - 이를 통해 고객이 환율변동에 너무 큰 영향을 받지 않게 원화자산/외화자산 비중을 적절히 조절하도록 유도합니다.
    ''')
    
    fx_usdkrw = st_utils.get_fdr_data('USD/KRW', start=today_str, end=today_str)
    fx_usdkrw = utils.get_fdr_last(fx_usdkrw)

    st.subheader(f':dollar: 오늘의 환율: {fx_usdkrw} 원/달러')

    samsung_price = st_utils.get_fdr_data('005930', start=today_str, end=today_str)
    samsung_price = utils.get_fdr_last(samsung_price)
    st.write('원화자산을 입력해보세요 (입력 후 엔터)')
    st.write(f'현재 삼성전자 가격: {samsung_price} 원')
    samsung_vol = float(st.text_input('삼성전자 몇 주?', 1))
    

    apple_price = st_utils.get_fdr_data('AAPL', start=today_str, end=today_str)
    apple_price = utils.get_fdr_last(apple_price)
    st.write('외화자산을 입력해보세요 (입력 후 엔터)')
    st.write(f'현재 애플 가격: {apple_price} 달러')
    apple_vol = float(st.text_input('애플 몇 주?', 1))
    
    krw_asset = round(samsung_vol * samsung_price, 2)
    usd_asset = round(apple_vol * apple_price, 2)
    
    krw_total = round(krw_asset + usd_asset * fx_usdkrw, 2)
    usd_total = round(krw_asset / fx_usdkrw + usd_asset, 2)
    st.write(f'''
    현재 고객이 보유한 원화자산총합: {krw_asset} 원
    현재 고객이 보유한 외화자산총합: {usd_asset} 달러

    총 자산 (원화기준): {krw_total} 원
    총 자산 (달러기준): {usd_total} 달러
    ''')

    fx_change_krw = float(st.text_input('원달러 환율 변동값 (기본 50원)', 50))
    fx_usdkrw_changed = fx_usdkrw + fx_change_krw
    
    krw_total_after = round(krw_asset + usd_asset * fx_usdkrw_changed, 2)
    usd_total_after = round(krw_asset / fx_usdkrw_changed + usd_asset, 2)
    st.write(f'''
    원달러 환율이 {fx_change_krw}원 변한다면 고객님 자산은 이런 영향을 받아요.

    총 자산 (원화기준): {krw_total_after - krw_total} 원, {(krw_total_after - krw_total) / krw_total} % 변동
    총 자산 (달러기준): {usd_total_after - usd_total} 달러, {(usd_total_after - usd_total) / usd_total} % 변동
    ''')
    

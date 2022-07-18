import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

## custom libs
import st_utils

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

if dropbox == APPS[0]:
    pass
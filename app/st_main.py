import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import pandas as pd
import numpy as np
import datetime

## custom libs
from utils import SID2NAME, NAME2SID, SECTOR_DF
import st_utils, utils
import conf
from conf import PathConfig

plt.rc('font', family='Malgun Gothic')

QR_CODE = Image.open(PathConfig.ASSETS_PATH / 'img' / 'qr.png')
MYDATA_LOGO = Image.open(PathConfig.ASSETS_PATH / 'img' / 'mydatalogo.jpg')
WOORI_LOGO = Image.open(PathConfig.ASSETS_PATH / 'img' / 'wooribanklogo.jpg')

# st.image(WOORI_LOGO, width=200)
st.image(MYDATA_LOGO, width = 100)
st.title(':chart_with_upwards_trend: MyData 기획 POC 대시보드')
st.markdown('---')

APPS = [
    '투자종합리포트 POC',
    '1. 환율 변동에 따른 내 원화자산/외화자산 가치변화',
    '2. 환율 변동에 따른 외화자산 직접투자 환차익/환손실 계산',
    '3. 주식 종목 간 상관관계에 기반한 유사종목 보여주기 서비스',
    '4. 기 보유한 펀드 포트폴리오와 상관관계 낮은 펀드 추천',
    # '5. 주식 종목 상관관계 클러스터링에 기반한 '
    ]

with st.sidebar:
    dropbox = st.selectbox('Select app', APPS)

    st.markdown(f'''
    ### 현재 POC가 완성된 항목들:

    - {APPS[0]}
    - {APPS[1]}
    - {APPS[3]}
    ''')

    st.markdown('''
    ### 웹페이지 단축 주소:
    https://bit.ly/3OlfPO5
    ''')

    st.image(QR_CODE)

    st.write('해당 프로젝트 소스코드')
    st.markdown('[Source Code](https://github.com/jaepil-choi/st_woori)')
    st.write('제 깃허브 레포')
    st.markdown('[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/jaepil-choi)')

today = datetime.date.today()
today_int = utils.DateUtil.timestamp_2_intDate(today)
today_str = utils.DateUtil.numdate2stddate(today_int)

offset_day = today - datetime.timedelta(days=conf.OFFSET_DAYS)
offset_day_int = utils.DateUtil.timestamp_2_intDate(offset_day)
offset_day_str = utils.DateUtil.numdate2stddate(offset_day_int)

if dropbox == APPS[0]:
    st.header(APPS[0])
    stock_kind = st.radio('', ['국내주식', '해외주식']) # Does nothing

    START = 20210514
    END = 20220520

    PIXEL_WIDTH = 380

    # 지난 1년 (252일) 간 가격데이터가 모두 존재했던 종목만 남김
    # 즉, 1년 중 상폐 / 신규상장 되었던 기업들 모두 빠짐
    return_df = pd.read_pickle(PathConfig.DATA_PATH / 'recent252_return_df.pkl')

    sid_list = return_df.columns
    sidname_list = [utils.sid2name(sid) for sid in sid_list]
    sidname_list = [sidname for sidname in sidname_list if sidname is not None]
    sidname_list = sorted(sidname_list)

    # st.write(return_df)
    # st.write(len(return_df.columns))
    myportfolio = pd.DataFrame([
        {
            'sid': '005930', # 삼성전자005930
            'name': SID2NAME['005930'],
            'buy_price': utils.get_fdr_last(st_utils.get_fdr_data('005930', START-10, START)),
            'now_price': utils.get_fdr_last(st_utils.get_fdr_data('005930', END-10, END)),
            'volume': 10,
        },
        {
            'sid': '000660', # SK하이닉스000660
            'name': SID2NAME['000660'],
            'buy_price': utils.get_fdr_last(st_utils.get_fdr_data('000660', START-10, START)),
            'now_price': utils.get_fdr_last(st_utils.get_fdr_data('000660', END-10, END)),
            'volume': 5,
        },
        {
            'sid': '316140', # 우리금융지주316140
            'name': SID2NAME['316140'],
            'buy_price': utils.get_fdr_last(st_utils.get_fdr_data('316140', START-10, START)),
            'now_price': utils.get_fdr_last(st_utils.get_fdr_data('316140', END-10, END)),
            'volume': 100,
        },
        {
            'sid': '105560', # KB금융105560
            'name': SID2NAME['105560'],
            'buy_price': utils.get_fdr_last(st_utils.get_fdr_data('105560', START-10, START)),
            'now_price': utils.get_fdr_last(st_utils.get_fdr_data('105560', END-10, END)),
            'volume': 20,
        },
        {
            'sid': '017670', # SK텔레콤017670 
            'name': SID2NAME['017670'],
            'buy_price': utils.get_fdr_last(st_utils.get_fdr_data('017670', START-10, START)),
            'now_price': utils.get_fdr_last(st_utils.get_fdr_data('017670', END-10, END)),
            'volume': 3,
        },
    ])
    myportfolio['dollarvolume'] = myportfolio['buy_price'] * myportfolio['volume']
    st.write('내 포트폴리오')
    st.write(myportfolio[['name', 'volume']])

    ########### 벤치마크와 누적수익률 비교

    dollarvolume = np.array(myportfolio['dollarvolume'])
    weights = dollarvolume / np.sum(dollarvolume)
    
    ret_df = return_df[list(myportfolio['sid'])]
    cumret_df = (ret_df + 1).cumprod() - 1
    cumret_df = 100 * cumret_df

    portfolio_cumret = cumret_df.iloc[:, 0] * weights[0]
    for i, w in enumerate(weights[1:]):
        portfolio_cumret += cumret_df.iloc[:, i+1] * w
    portfolio_cumret.rename('포트폴리오', inplace=True)

    kospi = st_utils.get_fdr_data('KS11', start=START, end=END)
    kospi = kospi['Close']
    kospi_return = kospi.pct_change()
    kospi_cumret = (kospi_return + 1).cumprod() - 1
    kospi_cumret = 100 * kospi_cumret
    kospi_cumret.rename('KOSPI', inplace=True)

    merged_df = pd.concat([portfolio_cumret, kospi_cumret], axis=1)
    
    selected_fig = px.line(merged_df)   
    selected_fig.update_layout(
        title=f'KOSPI vs 내 포트폴리오',
        xaxis_title='날짜',
        yaxis_title='누적수익률',
        # legend_title='종목코드(클릭가능)',
        width=PIXEL_WIDTH,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            ),
    )
    st.plotly_chart(selected_fig, use_container_width=False)

    port_kospi_diff = portfolio_cumret.iloc[-1] - kospi_cumret.iloc[-1]
    port_last_ret = portfolio_cumret.iloc[-1]
    st.markdown(f'''
    오늘 내 주식포트폴리오는 <b style="color:DeepPink;">{port_last_ret:.2f}%</b>올랐어요. 

    <u style="color:Turquoise;">1년동안</u> 비교해보니 <u style="color:Turquoise;">KOSPI200</u>보다 
    <b style="color:DeepPink;">{port_kospi_diff:.2f}%</b> 높은 수익률을 기록했어요.
    ''', unsafe_allow_html=True) # 떨어질 땐 DeepSkyBlue

    st.markdown("""---""")

    ########## 섹터 비중 확인

    st.markdown('''
    연결한 모든 증권사의 주식을 모아 통합 포트폴리오를 만들었어요. 

    <b style="font-size: 20px; color:DodgerBlue;">내 포트폴리오는 얼마나 잘 분산되어 있을까요?</b>
    ''', unsafe_allow_html=True)

    st.warning('''
    한 번 확인해보세요
    - 너무 [전기전자], [금융업] 섹터에 편중되어 있어요.
    ''')

    myport_sector_df = myportfolio.merge(SECTOR_DF[['sid', 'sector', 'marketCap']], how='left', on='sid')
    myport_agg_df = myport_sector_df.groupby('sector', as_index=False).sum()
    
    # st.write(myport_sector_df)
    
    pie_fig = px.pie(myport_agg_df, values='dollarvolume', names='sector')
    pie_fig.update_layout(
        title=f'섹터별 분산투자 현황',
        width=PIXEL_WIDTH,
    )
    st.plotly_chart(pie_fig, )

    ######### 섹터별 대표 회사 보기
    SHOW_LINES = 10

    st.markdown('''
    <b style="font-size: 20px; color:DodgerBlue;">내 섹터의 대표 종목들을 살펴보세요</b>
    ''', unsafe_allow_html=True)
    sectors = myport_agg_df['sector'].tolist()
    selected_sector = st.radio('섹터를 선택하세요', sectors)
    filtered_df = SECTOR_DF[SECTOR_DF['sector'] == selected_sector].sort_values(by='marketCap', ascending=False)
    filtered_df = filtered_df.head(SHOW_LINES)
    filtered_df['rounded_mktcap'] = filtered_df['marketCap'] / 1e+8
    filtered_df = filtered_df[['sid', 'name', 'rounded_mktcap']]
    filtered_df.columns = ['종목코드', '종목명', '시가총액(억원)']

    st.write(filtered_df.style.format(precision=0, thousands=','))
    

if dropbox == APPS[1]:
    st.header(APPS[1])
    st.info('''
    - 고객이 가진 (달러)외화자산의 가치가 원달러 환율 변동에 의해 얼마나 변하는지 보여줍니다.
    - 고객 전체 자산에서 원화자산 대비 외화자산이 많을 수록 환율변동에 자산가치가 크게 노출됩니다.
    - 환율이 10원 변할 때 고객의 전체 자산가치가 몇%나 영향을 받는지 보여줍니다. 
    - 이를 통해 고객이 환율변동에 너무 큰 영향을 받지 않게 원화자산/외화자산 비중을 적절히 조절하도록 유도합니다.
    ''')
    
    fx_usdkrw = st_utils.get_fdr_data('USD/KRW', start=offset_day_str, end=today_str)
    fx_usdkrw = utils.get_fdr_last(fx_usdkrw)

    st.subheader(f':dollar: 오늘의 환율: {fx_usdkrw:,} 원/달러')

    samsung_logo = 'https://cdn.iconscout.com/icon/free/png-256/samsung-226432.png'
    st.image(samsung_logo, width=100)
    samsung_price = st_utils.get_fdr_data('005930', start=offset_day_str, end=today_str)
    samsung_price = utils.get_fdr_last(samsung_price)
    st.write('원화자산을 입력해보세요 (입력 후 엔터)')
    st.subheader(f'현재 삼성전자 가격: {samsung_price:,} 원')
    samsung_vol = float(st.text_input('삼성전자 몇 주?', 1000))
    
    apple_logo = 'http://alsanad.ae/wp-content/uploads/2016/10/apple-logo.png'
    st.image(apple_logo, width=100)
    apple_price = st_utils.get_fdr_data('AAPL', start=offset_day_str, end=today_str)
    apple_price = utils.get_fdr_last(apple_price)
    st.write('외화자산을 입력해보세요 (입력 후 엔터)')
    st.subheader(f'현재 애플 가격: {apple_price:,} 달러')
    apple_vol = float(st.text_input('애플 몇 주?', 1000))

    currency = st.radio('기준 통화를 선택해주세요', ['원', '달러'])
    
    krw_asset = round(samsung_vol * samsung_price, 2)
    usd_asset = round(apple_vol * apple_price, 2)
    
    krw_total = round(krw_asset + usd_asset * fx_usdkrw, 2)
    usd_total = round(krw_asset / fx_usdkrw + usd_asset, 2)
    
    if currency == '원':
        st.write(f'현재 고객이 보유한 원화자산총합:')
        st.subheader(f'{krw_asset:,} 원')

        st.write('현재 고객이 보유한 외화자산총합:') 
        st.subheader(f'{usd_asset:,} 달러')

        st.write('총 자산 (원화기준):') 
        st.subheader(f'{krw_total:,} 원')

        before_df = pd.DataFrame([
            {
                'asset_type': '원화자산(삼성)',
                'asset_value': krw_asset, 
            }, {
                'asset_type': '외화자산(애플)',
                'asset_value': usd_asset * fx_usdkrw, 
            }
            ])
        fig_before = px.pie(before_df, values='asset_value', names='asset_type')
        st.plotly_chart(fig_before, use_container_width=False)

    elif currency == '달러':
        st.write(f'현재 고객이 보유한 원화자산총합:')
        st.subheader(f'{krw_asset:,} 원')

        st.write('현재 고객이 보유한 외화자산총합:') 
        st.subheader(f'{usd_asset:,} 달러')

        st.write('총 자산 (원화기준):') 
        st.subheader(f'{usd_total:,} 달러')

        before_df = pd.DataFrame([
            {
                'asset_type': '원화자산(삼성)',
                'asset_value': krw_asset / fx_usdkrw, 
            }, {
                'asset_type': '외화자산(애플)',
                'asset_value': usd_asset, 
            }
            ])
        fig_before = px.pie(before_df, values='asset_value', names='asset_type')
        st.plotly_chart(fig_before, use_container_width=False)


    fx_change_krw = float(st.text_input('원달러 환율 변동값 (기본 50원)', 50))
    fx_usdkrw_changed = fx_usdkrw + fx_change_krw
    
    krw_total_after = round(krw_asset + usd_asset * fx_usdkrw_changed, 2)
    usd_total_after = round(krw_asset / fx_usdkrw_changed + usd_asset, 2)
    
    if currency == '원':
        st.write(f'''
        원달러 환율이 {fx_change_krw}원 변한다면 고객님 자산은 이런 영향을 받아요.

        환율이 {fx_usdkrw_changed:,} 원이 된다면...

        변동액 (원화기준):
        ''')
        st.subheader(f'{round(krw_total_after - krw_total, 2):,} 원, {round(100 * (krw_total_after - krw_total) / krw_total, 2)} % 변동')

        after_df = pd.DataFrame([
            {
                'asset_type': '원화자산(삼성)',
                'asset_value': krw_asset, 
            }, {
                'asset_type': '외화자산(애플)',
                'asset_value': usd_asset * fx_usdkrw_changed, 
            }
            ])
        fig_after = px.pie(after_df, values='asset_value', names='asset_type')
        st.plotly_chart(fig_after, use_container_width=False)

    elif currency == '달러':
        st.write(f'''
        원달러 환율이 {fx_change_krw}원 변한다면 고객님 자산은 이런 영향을 받아요.

        환율이 {fx_usdkrw_changed:,} 원이 된다면...
        
        변동액 (달러기준):
        ''')
        st.subheader(f'{round(usd_total_after - usd_total, 2):,} 달러, {round(100 * (usd_total_after - usd_total) / usd_total, 2)} % 변동')

        after_df = pd.DataFrame([
            {
                'asset_type': '원화자산(삼성)',
                'asset_value': krw_asset / fx_usdkrw_changed, 
            }, {
                'asset_type': '외화자산(애플)',
                'asset_value': usd_asset, 
            }
            ])
        fig_after = px.pie(after_df, values='asset_value', names='asset_type')
        st.plotly_chart(fig_after, use_container_width=False)
    
if dropbox == APPS[2]:
    DAYS = 252

    start_day = today - datetime.timedelta(days=DAYS)
    start_day_int = utils.DateUtil.timestamp_2_intDate(start_day)
    start_day_str = utils.DateUtil.numdate2stddate(start_day_int)

 
    default_buy_date = datetime.date(2021, 12, 28)
    default_sell_date = datetime.date(2022, 6, 17)

    buy_date = st.date_input(
        '매수일', 
        value=default_buy_date,
        min_value=start_day,
        max_value=today
        )
    buy_date = pd.to_datetime(buy_date)
    sell_date = st.date_input(
        '매도일', 
        value=default_sell_date,
        min_value=start_day,
        max_value=today
        )
    sell_date = pd.to_datetime(sell_date)
    
    
    fx_usdkrw = st_utils.get_fdr_data('USD/KRW', start=start_day_str, end=today_str)
    fx_usdkrw = fx_usdkrw['Close'].copy()
    fx_usdkrw.columns = ['FX_USDKRW']
    apple_price = st_utils.get_fdr_data('AAPL', start=start_day_str, end=today_str)
    apple_price = apple_price['Close'].copy()
    apple_price.columns = ['APPL']

    fx_fig = px.line(fx_usdkrw)
    fx_fig.update_layout(yaxis_range=[1100, 1400])
    fx_fig.add_vline(x=buy_date, line_color='yellow')
    fx_fig.add_vline(x=sell_date, line_color='cyan')
    st.plotly_chart(fx_fig)

    appl_fig = px.line(apple_price)
    appl_fig.update_layout(yaxis_range=[100, 200])
    appl_fig.update_traces(line_color='red')
    appl_fig.add_vline(x=buy_date, line_color='yellow')
    appl_fig.add_vline(x=sell_date, line_color='cyan')
    st.plotly_chart(appl_fig) # TODO: secondary y axes 넣어보기 둘이 겹치게.

    ## 매수/매도 시점의 가격들
    fx_buy = fx_usdkrw.loc[buy_date]
    fx_sell = fx_usdkrw.loc[sell_date]
    appl_buy = apple_price.loc[buy_date]
    appl_sell = apple_price.loc[sell_date]

    ## 환차익 환차손 계산
    st.subheader('원화환산 수익률 (AS-IS)')

    st.write('원화매수금액')
    buy_won = appl_buy * fx_buy
    round(buy_won, 2)
    st.write('원화매도금액')
    sell_won = appl_sell * fx_sell
    round(sell_won, 2)
    st.write('원화수익률')
    return_won = ((sell_won - buy_won) / buy_won) * 100
    f'{round(return_won, 2)} %'

    st.subheader('환차 & 시세차익 분리 수익률 (TO-BE)')

    st.write('환차익/환차손 (%)')
    fx_return = ((fx_sell - fx_buy) / fx_buy)
    f'{round(fx_return * 100, 2)} %'
    f'환율로 벌거나 잃은 돈은 {round((fx_buy * appl_buy) * fx_return, 2)} 원 이에요.'
    
    st.write('주식 시세차익 (%)')    
    stock_return = ((appl_sell - appl_buy) / appl_buy)
    f'{round(stock_return * 100, 2)} %'
    f'순수하게 주식 시세 차이로 벌거나 잃은 돈은 {round((fx_sell * appl_sell) * stock_return, 2)} 원 이에요.'

    return_all = (1+fx_return) * (1+stock_return) - 1
    f'{round(return_all * 100, 2)}%'

    today_fx = fx_usdkrw.iloc[-1]
    fx_range_offset = 100
    min_fx = min(fx_usdkrw) - fx_range_offset 
    max_fx = max(fx_usdkrw) + fx_range_offset
    sim_fx = st.slider('환율이 이렇게 변한다면?', min_fx, max_fx, float(today_fx))

    


if dropbox == APPS[3]:
    st.header(APPS[3])
    st.info('''
    - 고객이 가진 주식종목과 가장 유사한 수익률을 보였던 종목을 순서대로 보여줍니다. 
    - 그리고 고객이 본인 보유종목과 유사종목 간 누적수익률 차이를 쉽게 볼 수 있게 해줍니다.
    ''')
    st.warning('''
    - 수익률 자체는 너무 불규칙해 그래프로 그려도 아무 이득이 없으므로, 누적 수익률을 계산하여 보여줍니다.
    - 누적수익률은 말 그대로 일정기간 전부터(여기선 252 거래일 전) 보유했을 때 각 기간까지의 수익률을 보여줍니다.
    - 이 POC 데이터의 마지막 일자는 22/05/20기 때문에 최신 주가정보를 보여주진 않습니다.
    ''')

    LOOKBACK_PERIOD = 252 # Trading days of 1 year / Fixed value for now. (As of 20220520)
    TOP_N = 10
    CORR_THRESHOLD = 0.6

    # 지난 1년 (252일) 간 가격데이터가 모두 존재했던 종목만 남김
    # 즉, 1년 중 상폐 / 신규상장 되었던 기업들 모두 빠짐
    return_df = pd.read_pickle(PathConfig.DATA_PATH / 'recent252_return_df.pkl')
    original_cols = return_df.columns
    valid_cols = [sid for sid in original_cols if utils.sid2name(sid) is not None]
    return_df = return_df.loc[:, valid_cols]

    return_corr_df = pd.read_pickle(PathConfig.DATA_PATH / 'return_corr_df.pkl')
    return_corr_df = return_corr_df.loc[valid_cols, valid_cols]

    corr_rank_df = return_corr_df.rank(ascending=False)
    
    sid_list = return_corr_df.columns
    sidname_list = [utils.sid2name(sid) for sid in sid_list]
    sidname_list = [sidname for sidname in sidname_list if sidname is not None]
    sidname_list = sorted(sidname_list)

    # st.write(sidname_list.index('우리금융지주'))

    selected = st.selectbox(
        '보유한 종목을 고르세요', 
        options=sidname_list, 
        index=1439 # pre-selected option, 우리금융지주 index: 1439
        )
    selected_sid = utils.name2sid(selected)

    st.subheader('내가 보유한 주식의 지난 1년간의 누적수익률')

    selected_return_df = return_df.loc[:, selected_sid].copy()
    selected_cumret_df = (selected_return_df + 1).cumprod() - 1
    selected_cumret_df = selected_cumret_df * 100

    selected_fig = px.line(selected_cumret_df)   
    selected_fig.update_layout(
        title=f'{selected}의 {LOOKBACK_PERIOD} 거래일 전부터 지금까지의 누적수익률',
        xaxis_title='날짜',
        yaxis_title='누적수익률',
        legend_title='종목코드(클릭가능)',
    )
    st.plotly_chart(selected_fig)

    selected_corr_values = return_corr_df.loc[:, selected_sid]
    selected_corr_rank = corr_rank_df.loc[:, selected_sid]

    top_N_sid_list = selected_corr_rank.sort_values().index[1:TOP_N+1]
    top_N_sidname_list = [utils.sid2name(sid) for sid in top_N_sid_list]
    top_N_corr_list = selected_corr_values.sort_values(ascending=False)[1:TOP_N+1]
    
    bottom_N_sid_list = selected_corr_rank.sort_values().index[-TOP_N:]
    bottom_N_sidname_list = [utils.sid2name(sid) for sid in bottom_N_sid_list]
    bottom_N_corr_list = selected_corr_values.sort_values(ascending=False)[-TOP_N:]

    st.subheader('보유 종목과 가장 유사한 종목들이에요')
    top_N_df = pd.DataFrame(data=zip(top_N_sid_list, top_N_sidname_list, top_N_corr_list), columns=['종목코드', '종목명', '상관계수'])
    # top_N_df = top_N_df[top_N_df['상관계수'] >= CORR_THRESHOLD]
    bottom_N_df = pd.DataFrame(data=zip(bottom_N_sid_list, bottom_N_sidname_list, bottom_N_corr_list), columns=['종목코드', '종목명', '상관계수'])
    st.write('가장 상관관계 큰 종목들')
    st.write(top_N_df)
    st.write('가장 상관관계 작은 종목들')
    st.write(bottom_N_df)

    similar = st.selectbox('비교할 종목을 고르세요', top_N_sidname_list)
    similar_sid = utils.name2sid(similar)
    
    dissimilar = st.selectbox('비교할 종목을 고르세요', bottom_N_sidname_list)
    dissimilar_sid = utils.name2sid(dissimilar)

    similar_return_df = return_df.loc[:, similar_sid].copy()
    similar_cumret_df = (similar_return_df + 1).cumprod() - 1
    similar_cumret_df = similar_cumret_df * 100
    
    dissimilar_return_df = return_df.loc[:, dissimilar_sid].copy()
    dissimilar_cumret_df = (dissimilar_return_df + 1).cumprod() - 1
    dissimilar_cumret_df = dissimilar_cumret_df * 100
    
    concat_df = pd.concat(
        objs=[
            selected_cumret_df,
            similar_cumret_df,
            dissimilar_cumret_df,
        ],
        axis=1,
    )
    concat_df.columns = [utils.sid2name(c) for c in concat_df.columns]

    similar_fig = px.line(concat_df)   
    similar_fig.update_layout(
        title=f'{concat_df.columns[0]}와 {concat_df.columns[1]}의 {LOOKBACK_PERIOD} 거래일 전부터 지금까지의 누적수익률',
        xaxis_title='날짜',
        yaxis_title='누적수익률',
        legend_title='종목명(클릭가능)',
    )
    st.plotly_chart(similar_fig)

    st.subheader('dev: 전 종목 상관계수의 분포')
    corr_values = return_corr_df.to_numpy().flatten()
    st.write(f'mean: {np.mean(corr_values)}')
    st.write(f'std: {np.std(corr_values)}')
    corr_hist_fig = px.histogram(corr_values, nbins=50)
    st.plotly_chart(corr_hist_fig)

# if dropbox == APPS[4]:
#     pass
# if dropbox == APPS[4]:
#     pass
# if dropbox == APPS[5]:
#     pass
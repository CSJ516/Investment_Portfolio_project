import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
import re
from selenium import webdriver
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

import warnings
warnings.filterwarnings('ignore')

# 시각화 한글 처리
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)

class Portfolio:
    def __init__(self):
        propensity = input('안정형(장기)의 경우 1, 공격투자형(단기)의 경우 2를 입력하세요. -> ')
        if propensity == '1':
            self.BlueChip()
        elif propensity == '2':
            self.ThemeStock()
        else:
            self.__init__()
        
    # 종목별 PER, PBR 수집
    def DataSet(self):
        krx_list = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13')[0]
        krx_list.종목코드 = krx_list.종목코드.map('{:06d}'.format)

        per_pbr = []
        for code in tqdm(krx_list.종목코드):
            n_url = f'https://finance.naver.com/item/main.nhn?code={code}'
            html = BeautifulSoup(requests.get(n_url, headers={'User-agent': 'Mozilla/5.0'}).text, "lxml")
            try:
                group = html.find('h4', class_='h_sub sub_tit7')
                업종 = group.find('a').text
            except:
                업종 = 'N/A'
            try:
                per = html.find('em', id='_per').text
                pbr = html.find('em', id='_pbr').text
            except:
                per, pbr = 'N/A', 'N/A'
            try:
                table = html.find('table', summary='동일업종 PER 정보')
                업종per = table.find('em').text
            except:
                업종per = 'N/A'
            per_pbr.append([업종, 업종per, per, pbr])
        per_pbr_df = pd.DataFrame(per_pbr)
        per_pbr_df.columns = ['업종2', '업종PER', 'PER', 'PBR']
        
        krx_df = krx_list[['회사명','종목코드','업종']]
        df1 = pd.concat([krx_df, per_pbr_df], axis=1)
        df1.replace('N/A', np.nan, inplace=True)
        return df1
    
    # 저평가 된 기업 찾기
    def LowValue(self):
        df1 = self.DataSet()
        df1['종목코드'] = df1['종목코드'].astype('object')
        df1['업종PER'] = df1['업종PER'].astype('float')
        df1['PER'] = df1['PER'].str.replace(',','')
        df1['PER'] = df1['PER'].astype('float')
        df1['PBR'] = df1['PBR'].apply(pd.to_numeric, errors='coerce')
        df2 = df1[(df1['업종PER'] > df1['PER']) & (df1['PBR'] <=1)].reset_index(drop=True)
        return df2
    
    # 저평가 된 우량주를 찾아 포트폴리오 구현
    def BlueChip(self): 
        """시가총액 수집"""
        df3 = self.LowValue()
        df3['시가총액(억)'] = np.nan
        for i, code in tqdm(enumerate(df3.종목코드)):
            cg_url = f'https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A{code}&cID=&MenuYn=Y&ReportGB=&NewMenuID=11&stkGb=701'
            html = BeautifulSoup(requests.get(cg_url, headers={'User-agent': 'Mozilla/5.0'}).text, 'lxml')
            table = html.find_all("tr")[4]
            price = table.find("td", class_="r").text
            df3['시가총액(억)'][i] = price
        df3['시가총액(억)'] = df3['시가총액(억)'].str.replace(',','')
        df3['시가총액(억)'] = df3['시가총액(억)'].astype(float)
        df3 = df3.loc[df3.groupby('업종2')['시가총액(억)'].idxmax()].sort_values(by='시가총액(억)', ascending=False).reset_index(drop=True)
        
        """저평가 된 우량주 찾기"""
        stocks = df3.종목코드[:5].tolist()
        bluechip = pd.DataFrame()
        for s in stocks:
            bluechip[s] = pdr.get_data_yahoo(f'{s}.KS', '2020-01-01')['Close'].values
        bluechip.index = pdr.get_data_yahoo(f'{stocks[0]}.KS', '2020-01-01').index
        bluechip.columns = [i for i in df3.회사명[:5]]

        daily_ret = bluechip.pct_change()
        annual_ret = daily_ret.mean() * 252
        daily_cov = daily_ret.cov()
        annual_cov = daily_cov * 252
        
        """20000개의 포트폴리오 생성"""
        rets, risks, weights, sharpe = [], [], [], []
        for _ in range(20000):
            weight = np.random.random(len(stocks))
            weight /= np.sum(weight)
            ret = np.dot(weight, annual_ret)
            risk = np.sqrt(np.dot(weight.T, np.dot(annual_cov, weight)))
            rets.append(ret)
            risks.append(risk)
            weights.append(weight)
            sharpe.append(ret/risk)

        portfolio = {'Returns': rets, 'Risk': risks, 'Sharpe': sharpe}
        for i, s in enumerate(stocks):
            portfolio[s] = [weight[i] for weight in weights]
        blue_port = pd.DataFrame(portfolio)
        blue_port.columns = ['Returns','Risk','Sharpe'] + [i for i in df3.회사명[:5]]
        
        max_sharpe = blue_port.loc[blue_port['Sharpe'] == blue_port['Sharpe'].max()]
        min_risk = blue_port.loc[blue_port['Risk'] == blue_port['Risk'].min()]

        blue_port.plot.scatter(x='Risk', y='Returns', c='Sharpe', cmap='viridis', edgecolors='k', figsize=(10,7), grid=True)
        plt.scatter(x=max_sharpe['Risk'], y=max_sharpe['Returns'], c='r', marker='*', s=300)
        plt.scatter(x=min_risk['Risk'], y=min_risk['Returns'], c='b', marker='*', s=200)
        plt.title('포트폴리오 최적화')
        plt.xlabel('Risk')
        plt.ylabel('Expected Returns')
        plt.show()
        print('예상되는 수익률과 위험률, 추천하는 우량주 종목별 비중은 다음과 같습니다.')
        print(max_sharpe)
    
    # 저평가 된 테마주를 찾아 포트폴리오 구현
    def ThemeStock(self):
        """거래량, 거래대금 수집"""
        chromedriver = 'C:\\Users\\dalgo\\OneDrive\\바탕 화면\\08_Investment_Portfolio\\chromedriver.exe'
        driver = webdriver.Chrome(chromedriver)
        driver.get('https://finance.naver.com/sise/sise_quant.nhn?sosok=0')

        driver.find_element_by_xpath('//*[@id="option2"]').click()
        driver.find_element_by_xpath('//*[@id="option8"]').click()
        driver.find_element_by_xpath('//*[@id="option9"]').click()
        driver.implicitly_wait(2)
        driver.find_element_by_xpath('//*[@id="option12"]').click()
        driver.find_element_by_xpath('//*[@id="option24"]').click()
        driver.find_element_by_xpath('//*[@id="contentarea_left"]/div[2]/form/div/div/div/a[1]/img').click()

        html1 = driver.page_source
        table1 = pd.read_html(html1)[1]
        table1 = table1.dropna(subset=['종목명']).drop(['N','Unnamed: 11'], axis=1).reset_index(drop=True)

        driver.implicitly_wait(2)
        driver.find_element_by_xpath('//*[@id="contentarea"]/div[3]/div/div[2]/a').click()
        html2 = driver.page_source
        table2 = pd.read_html(html2)[1]
        table2 = table2.dropna(subset=['종목명']).drop(['N','Unnamed: 11'], axis=1).reset_index(drop=True)
        df4 = pd.concat([table1, table2], axis=0).sort_values(by='거래량', ascending=False).reset_index(drop=True)
        
        """저평가 된 테마주 찾기"""
        df4 = df4[~df4.등락률.str.contains('-')].reset_index(drop=True)
        df4['평균거래량'] = (df4['거래량'] + df4['전일거래량'])/2
        df4 = df4.sort_values(by=['평균거래량','거래대금'], ascending=False).reset_index(drop=True)        
        df4 = df4.rename(columns={'거래대금': '거래대금(백만)'})
        df4 = df4[['종목명','등락률','평균거래량','거래대금(백만)']]
        
        df3 = self.LowValue()
        l, idx = [], []
        for i in df4['종목명']:
            if i in df3['회사명'].unique():
                l.append(i)
        for i in range(len(df3)):
            if df3['회사명'][i] in l:
                idx.append(i)
        m_df = pd.merge(df4, df3.loc[idx], how='right', left_on='종목명', right_on='회사명')
        m_df = m_df[['회사명','종목코드','업종','업종2','업종PER','PER','PBR','등락률','평균거래량','거래대금(백만)']]
        m_df['순위'] = m_df['평균거래량'].rank() + m_df['거래대금(백만)'].rank()
        m_df = m_df.sort_values(by='순위', ascending=False).reset_index(drop=True)
        g_df = m_df.groupby(['업종2'])['평균거래량','거래대금(백만)','순위'].sum()
        g_df = g_df.sort_values(by='순위', ascending=False)
        
        idx2 = []
        for i in g_df.index[:5]:
            for j in m_df.index:
                if m_df['업종2'][j] == i:
                    idx2.append(j)
                    break
        print('최근 이틀간 주목받은 업종은 {}이며, 추천하는 테마주 종목별 비중은 {} 순서입니다.'.format([i for i in m_df.loc[idx2]['업종2'].values][:5], [i for i in m_df.loc[idx2]['회사명'].values][:5]))
        print(m_df.loc[idx2].reset_index(drop=True).drop('순위', axis=1))




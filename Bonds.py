import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dateutil import relativedelta
#streamlit run E:\Mengqi\Streamlit\Bonds.py

def remove_0(x, y):
    if y == 0:
        return np.nan
    else:
        return x/y


def q1(x):
    return x.quantile(0.01)
def q5(x):
    return x.quantile(0.05)
def q10(x):
    return x.quantile(0.1)
def q25(x):
    return x.quantile(0.25)
def q50(x):
    return x.quantile(0.5)
def q75(x):
    return x.quantile(0.75)
def q90(x):
    return x.quantile(0.9)
def q95(x):
    return x.quantile(0.95)
def q99(x):
    return x.quantile(0.99)
def count_comp(x):
    return x.nunique()

def generate_type_matrix(tmp_df):
    df_all = pd.DataFrame()
    df = cal_matrix_num(tmp_df[(tmp_df.Rating1_x == tmp_df.Rating1_y)&(tmp_df.PDiR1_x == tmp_df.PDiR1_y)])
    df['Fix_Type'] = 'Rating&PDiR'
    df_all = pd.concat([df_all, df], axis = 0)
    df = cal_matrix_num(tmp_df[(tmp_df.Rating1_x == tmp_df.Rating1_y)])
    df['Fix_Type'] = 'Rating'
    df_all = pd.concat([df_all, df], axis = 0)
    df = cal_matrix_num(tmp_df[(tmp_df.PDiR1_x == tmp_df.PDiR1_y)])
    df['Fix_Type'] = 'PDiR'
    df_all = pd.concat([df_all, df], axis = 0)
    df = cal_matrix_num(tmp_df)
    df['Fix_Type'] = 'N/A'
    df_all = pd.concat([df_all, df], axis = 0)
    df_all = df_all.set_index(['Fix_Type', 'Type'])
    return df_all
        
        

def cal_matrix_num(tmp_df):
    total_num = tmp_df.shape[0]
    df = pd.DataFrame(columns = ['<Q25', 'Q25_Q50', 'Q50_Q75', '>Q75'])
    df['Type'] = ['<Q25', 'Q25_Q50', 'Q50_Q75', '>Q75']
    for i in range(4):
        for j in range(4):
            aa = tmp_df[(tmp_df.Flag_x==i)&(tmp_df.Flag_y==j)]
            df.iloc[i,j] = "{rate:.1f}% ({num:.0f}, {std:.4f})".format(rate = remove_0(aa.shape[0]*100, total_num), num = aa.shape[0], std = aa.Yield.std())
    return df




st.write("Bond Analysis")

# bonddata = pd.read_csv(r'E:\Mengqi\CDS\DAS\DAS\Analysis\Domestic bond\bonddata.csv')
# bonddata = pd.read_csv(r'bonddata.csv')

bonddata = pd.DataFrame()

for i in range(11):
    print(i)
    tmp = pd.read_csv('bonddata{}.csv'.format(i))
    bonddata = pd.concat([bonddata, tmp], axis = 0)
bonddata = bonddata.reset_index(drop = True)
# for i in range(10):
#     print(i)
#     a = 250000*i
#     b = 250000*(i+1)
#     bonddata.iloc[a:b, :].to_csv(r'E:\Mengqi\Streamlit\bonds_analysis\bonddata{}.csv'.format(i), index = False)

# bonddata.iloc[2500000:, :].to_csv(r'E:\Mengqi\Streamlit\bonds_analysis\bonddata{}.csv'.format(10), index = False)

print(bonddata.columns)
bonddata["YYYYMMDD"] = pd.to_datetime(bonddata["YYYYMMDD"])
bonddata_bk = bonddata.copy(deep = True)
# df = pd.DataFrame(columns = ['Model', 'Type', 'Yield'])
# df['Model'] = ['LGFV', 'CORP','LGFV', 'CORP','LGFV', 'CORP']
# df['Type'] = ['1', '2','3', '3','4', '1']
# df['Yield'] = [1,2,3,4,5,6]
st.sidebar.title("Please select the category")
# Model = st.sidebar.multiselect('choose the Model',options = list(bonddata_bk['Model'].unique())+['All'], default=["All"])
Period = st.sidebar.multiselect('choose the Period',options = list(bonddata_bk['Period'].unique())+['All'], default=["All"])
isDomestic = st.sidebar.multiselect('choose the isDomestic',options = list(bonddata_bk['isDomestic'].unique())+['All'], default=["All"])
OptionType = st.sidebar.multiselect('choose the OptionType',options = list(bonddata_bk['OptionType'].unique())+['All'], default=["All"])
Rating = st.sidebar.multiselect('choose the Rating',options = list(bonddata_bk['Rating'].unique())+['All'], default=["All"])
PDiR = st.sidebar.multiselect('choose the PDiR',options = list(bonddata_bk['PDiR'].unique())+['All'], default=["All"])
TimeToMaturity = st.sidebar.multiselect('choose the TimeToMaturity',options = list(bonddata_bk['TimeToMaturity'].unique())+['All'], default=["All"])

CouponChg = st.sidebar.multiselect('choose the CouponChg',options = list(bonddata_bk['CouponChg'].unique())+['All'], default=["All"])
Is_Public_Issued = st.sidebar.multiselect('choose the Is_Public_Issued',options = list(bonddata_bk['Is_Public_Issued'].unique())+['All'], default=["All"])
multiValuation = st.sidebar.multiselect('choose the multiValuation',options = list(bonddata_bk['multiValuation'].unique())+['All'], default=["All"])
valEndDate = st.sidebar.multiselect('choose the valEndDate>Maturity',options = list(bonddata_bk['valEndDate>Maturity'].unique())+['All'], default=["All"])

imonth = st.sidebar.multiselect('choose the Month',options = [3,6,12], default=[3])
Fix_Type = st.sidebar.multiselect('choose the Fix_Type',options = ['Rating&PDiR', 'Rating', 'PDiR', 'N/A'], default=['Rating&PDiR'])

bonddata['M_later'] = bonddata["YYYYMMDD"].apply(lambda x: (x + relativedelta.relativedelta(months = imonth[0])).strftime('%Y%m') )
bonddata['M_later'] = bonddata["M_later"].apply(lambda x: int(x))
itype = ['All']
# if 'All' not in Model:
#     bonddata = bonddata[bonddata['Model'].isin(Model)]
#     itype.append("Model")
if 'All' not in Period:
    bonddata = bonddata[bonddata['Period'].isin(Period)]
    itype.append("Period")
if 'All' not in isDomestic:
    bonddata = bonddata[bonddata['isDomestic'].isin(isDomestic)]
    itype.append("isDomestic")
if 'All' not in OptionType:
    bonddata = bonddata[bonddata['OptionType'].isin(OptionType)]
    itype.append("OptionType")
if 'All' not in Rating:
    bonddata = bonddata[bonddata['Rating'].isin(Rating)]
    itype.append("Rating")
if 'All' not in PDiR:
    bonddata = bonddata[bonddata['PDiR'].isin(PDiR)]
    itype.append("PDiR")
if 'All' not in TimeToMaturity:
    bonddata = bonddata[bonddata['TimeToMaturity'].isin(TimeToMaturity)]
    itype.append("TimeToMaturity")
if 'All' not in CouponChg:
    bonddata = bonddata[bonddata['CouponChg'].isin(CouponChg)]
    itype.append("CouponChg")
if 'All' not in Is_Public_Issued:
    bonddata = bonddata[bonddata['Is_Public_Issued'].isin(Is_Public_Issued)]
    itype.append("Is_Public_Issued")
if 'All' not in multiValuation:
    bonddata = bonddata[bonddata['multiValuation'].isin(multiValuation)]
    itype.append("multiValuation")
if 'All' not in valEndDate:
    bonddata = bonddata[bonddata['valEndDate>Maturity'].isin(valEndDate)]
    itype.append("valEndDate>Maturity")

col = 'Yield'
cut_b = bonddata.groupby(itype + ['yearmonth']).agg({f'{col}': [q25, q50, q75]}).reset_index()
cut_b.columns = [''.join(col) for col in cut_b.columns]

tmp = bonddata[itype + ['yearmonth', 'Yield', 'bondID', 'YYYYMMDD', 'valEndDate']].merge(cut_b, on = itype + ['yearmonth'], how = 'left')
tmp['Q25'] = tmp['Yield'] - tmp['Yieldq25']
tmp['Q50'] = tmp['Yield'] - tmp['Yieldq50']
tmp['Q75'] = tmp['Yield'] - tmp['Yieldq75']

tmp['Flag'] = (tmp[['Q25', 'Q50', 'Q75']]>0).sum(axis = 1)
tmp = tmp[itype + ['yearmonth', 'Flag', 'bondID', 'YYYYMMDD','valEndDate', 'Yield']]

tmp = tmp.merge(bonddata[['multiValuation', 'bondID','YYYYMMDD','M_later', 'valEndDate', 'Rating', 'PDiR']].rename(columns = {'Rating':'Rating1', 'PDiR':'PDiR1', 'multiValuation': 'multiValuation1'}), how = 'left', on = ['bondID','YYYYMMDD', 'valEndDate'])
# tmp = tmp.merge(bonddata[itype+ ['bondID','YYYYMMDD', 'valEndDate']], how = 'left', on = ['bondID','YYYYMMDD', 'valEndDate']+itype)

tmp1 = tmp.loc[~tmp.multiValuation1, itype + ['bondID','yearmonth', 'M_later', 'valEndDate', 'Rating1', 'PDiR1', 'Flag', 'Yield']].merge(tmp.loc[~tmp.multiValuation1,['bondID', 'yearmonth', 'Rating1', 'PDiR1', 'Flag']].rename(columns = {'yearmonth':'M_later'}), on = ['bondID', 'M_later'], how = 'inner')
tmp2 = tmp.loc[tmp.multiValuation1, itype + [ 'bondID','yearmonth', 'M_later', 'valEndDate', 'Rating1', 'PDiR1' , 'Flag', 'Yield']].merge(tmp.loc[tmp.multiValuation1,['bondID', 'yearmonth', 'valEndDate', 'Rating1', 'PDiR1', 'Flag']].rename(columns = {'yearmonth':'M_later'}), on = ['bondID', 'M_later', 'valEndDate'], how = 'inner')
tmp_df = pd.concat([tmp1, tmp2], axis = 0)

df = tmp_df.groupby(itype).apply(lambda x: generate_type_matrix(x)).reset_index()

            # num_orig = tmp[itype + ['Yield', 'Flag']].groupby(itype).count().reset_index().rename(columns = {'Yield': 'Original_Obs', 'Flag':'Type'})
            # num_orig.loc[num_orig['Type']==0, 'Type'] = '<Q25'
            # num_orig.loc[num_orig['Type']==1, 'Type'] = 'Q25_Q50'
            # num_orig.loc[num_orig['Type']==2, 'Type'] = 'Q50_Q75'
            # num_orig.loc[num_orig['Type']==3, 'Type'] = '>Q75'
st.dataframe(df[df.Fix_Type.isin(Fix_Type)]) 












# will display the dataframe
# fig = px.box(bonddata['Yield'])
# st.plotly_chart(fig)



        # state_total_graph = px.bar(
        #     state_total,
        #     x='病例分类',
        #     y='病例数',
        #     labels={'病例数': '%s 国家的总病例数' % (select)},
        #     color='病例分类')
        # st.plotly_chart(state_total_graph)

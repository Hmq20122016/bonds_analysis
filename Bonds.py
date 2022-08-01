import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dateutil import relativedelta
import gc
#streamlit run E:\Mengqi\Streamlit\Bonds.py
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

def generate_type_matrix(tmp_df, Fix_Type):
    df_all = pd.DataFrame()
    for iType in Fix_Type:
        if iType == 'Rating&PDiR':
            df = cal_matrix_num(tmp_df[(tmp_df.Rating1_x == tmp_df.Rating1_y)&(tmp_df.PDiR1_x == tmp_df.PDiR1_y)])
            df['Fix_Type'] = 'Rating&PDiR'
        elif iType == 'Rating':
            df = cal_matrix_num(tmp_df[(tmp_df.Rating1_x == tmp_df.Rating1_y)])
            df['Fix_Type'] = 'Rating'
        elif iType == 'PDiR':
            df = cal_matrix_num(tmp_df[(tmp_df.PDiR1_x == tmp_df.PDiR1_y)])
            df['Fix_Type'] = 'PDiR'
        elif iType == 'All':
            df = cal_matrix_num(tmp_df)
            df['Fix_Type'] = 'All'
        df_all = pd.concat([df_all, df], axis = 0)
    df_all = df_all.set_index(['Fix_Type', 'Type'])
    return df_all
        
        

def cal_matrix_num(tmp_df):
    total_num = tmp_df.shape[0]
    df = pd.DataFrame(columns = ['<Q25 (obs, std)', 'Q25_Q50 (obs, std)', 'Q50_Q75 (obs, std)', '>Q75 (obs, std)', 'Sum_obs'])
    df['Type'] = ['<Q25', 'Q25_Q50', 'Q50_Q75', '>Q75']
    for i in range(4):
        num_obs = 0
        for j in range(4):
            aa = tmp_df[(tmp_df.Flag_x==i)&(tmp_df.Flag_y==j)]
            num_obs = num_obs + aa.shape[0]
            df.iloc[i,j] = "{rate:.1f}% ({num:.0f}, {std:.4f})".format(rate = remove_0(aa.shape[0]*100, total_num), num = aa.shape[0], std = aa.Yield.std())
        df.iloc[i,4] = num_obs
    return df

def generate_result(bonddata, imonth, Model, Period, isDomestic, OptionType, Rating, PDiR, TimeToMaturity, CouponChg, Is_Public_Issued, multiValuation, valEndDate, Fix_Type):

    itype = ['All']
    if 'All' not in Model:
        bonddata = bonddata[bonddata['Model'].isin(Model)]
        itype.append("Model")
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
    cut_b = []
    tmp['Q25'] = tmp['Yield'] - tmp['Yieldq25']
    tmp['Q50'] = tmp['Yield'] - tmp['Yieldq50']
    tmp['Q75'] = tmp['Yield'] - tmp['Yieldq75']

    tmp['Flag'] = (tmp[['Q25', 'Q50', 'Q75']]>0).sum(axis = 1)
    tmp = tmp[itype + ['yearmonth', 'Flag', 'bondID', 'YYYYMMDD','valEndDate', 'Yield']]
    # tmp = tmp.merge(bonddata[itype+ ['bondID','YYYYMMDD', 'valEndDate']], how = 'left', on = ['bondID','YYYYMMDD', 'valEndDate']+itype)
    num_orig = tmp[itype + ['Yield', 'Flag']].groupby(itype + ['Flag']).count().reset_index().rename(columns = {'Yield': 'Original_Obs', 'Flag':'Type'})
    num_orig.loc[num_orig['Type']==0, 'Type'] = '<Q25'
    num_orig.loc[num_orig['Type']==1, 'Type'] = 'Q25_Q50'
    num_orig.loc[num_orig['Type']==2, 'Type'] = 'Q50_Q75'
    num_orig.loc[num_orig['Type']==3, 'Type'] = '>Q75'
    print(num_orig)

    tmp = tmp.merge(bonddata[['multiValuation', 'bondID','YYYYMMDD',f'{imonth}M_later', 'valEndDate', 'Rating', 'PDiR']].rename(columns = {'Rating':'Rating1', 'PDiR':'PDiR1', 'multiValuation': 'multiValuation1'}), how = 'left', on = ['bondID','YYYYMMDD', 'valEndDate'])


    tmp1 = tmp.loc[~tmp.multiValuation1, itype + ['bondID','yearmonth', f'{imonth}M_later', 'valEndDate', 'Rating1', 'PDiR1', 'Flag', 'Yield']].merge(tmp.loc[~tmp.multiValuation1,['bondID', 'yearmonth', 'Rating1', 'PDiR1', 'Flag']].rename(columns = {'yearmonth':f'{imonth}M_later'}), on = ['bondID', f'{imonth}M_later'], how = 'inner')
    tmp2 = tmp.loc[tmp.multiValuation1, itype + [ 'bondID','yearmonth', f'{imonth}M_later', 'valEndDate', 'Rating1', 'PDiR1' , 'Flag', 'Yield']].merge(tmp.loc[tmp.multiValuation1,['bondID', 'yearmonth', 'valEndDate', 'Rating1', 'PDiR1', 'Flag']].rename(columns = {'yearmonth':f'{imonth}M_later'}), on = ['bondID', f'{imonth}M_later', 'valEndDate'], how = 'inner')
    tmp = pd.concat([tmp1, tmp2], axis = 0)

    df = tmp.groupby(itype).apply(lambda x: generate_type_matrix(x, Fix_Type)).reset_index()
    df = df.merge(num_orig, on = itype + ['Type'], how = 'left')


    st.dataframe(df)
    gc.collect()

@st.cache
def get_bonddata():
    bonddata = pd.DataFrame()
    for i in range(11):
        print(i)
        tmp = pd.read_parquet(r'E:\Mengqi\bonds_analysis\bonddata{}.parquet.gzip'.format(i))
        # tmp = pd.read_csv(r'E:\Mengqi\bonds_analysis\bonddata{}.csv'.format(i))
        # tmp["YYYYMMDD"] = pd.to_datetime(tmp["YYYYMMDD"])
        # for j in [3, 6, 12]:
        #     tmp[f'{j}M_later'] = tmp["YYYYMMDD"].apply(lambda x: (x + relativedelta.relativedelta(months = j)).strftime('%Y%m') )
        #     tmp[f'{j}M_later'] = tmp[f'{j}M_later'].apply(lambda x: int(x))
        # tmp.to_parquet(r'E:\Mengqi\bonds_analysis\bonddata{}.parquet.gzip'.format(i),
        #     compression='gzip')  

        bonddata = pd.concat([bonddata, tmp], axis = 0)
    bonddata = bonddata.reset_index(drop = True)

    list_fill = ['OptionType', 'isDomestic', 'Model', 'PDiR', 'Rating', 'TimeToMaturity', 'All', 'Period' , 'CouponChg', 'Is_Public_Issued', 'multiValuation', 'valEndDate>Maturity']
    bonddata[list_fill] = bonddata[list_fill].fillna('N/A')
    return bonddata


st.write("Bond Analysis")

# bonddata = pd.read_csv(r'E:\Mengqi\CDS\DAS\DAS\Analysis\Domestic bond\bonddata.csv')
# bonddata = pd.read_csv(r'bonddata.csv')


# for i in range(10):
#     print(i)
#     a = 250000*i
#     b = 250000*(i+1)
#     bonddata.iloc[a:b, :].to_csv(r'E:\Mengqi\Streamlit\bonds_analysis\bonddata{}.csv'.format(i), index = False)

# bonddata.iloc[2500000:, :].to_csv(r'E:\Mengqi\Streamlit\bonds_analysis\bonddata{}.csv'.format(10), index = False)


# df = pd.DataFrame(columns = ['Model', 'Type', 'Yield'])
# df['Model'] = ['LGFV', 'CORP','LGFV', 'CORP','LGFV', 'CORP']
# df['Type'] = ['1', '2','3', '3','4', '1']
# df['Yield'] = [1,2,3,4,5,6]
st.sidebar.title("Please select the category")
imonth = st.sidebar.multiselect('choose the Month',options = [3,6,12], default=[3]) 
imonth = imonth[0]
# Model = st.sidebar.multiselect('choose the Model',options = list(bonddata['Model'].unique())+['All'], default=['All'])
Model = st.sidebar.multiselect('choose the Period',options = ['CORP', 'LGFV', 'FINA', 'N/A', 'All'], default=['All'])
Period = st.sidebar.multiselect('choose the Period',options = ['Before2018', 'After2018', 'All'], default=['All'])
isDomestic = st.sidebar.multiselect('choose the isDomestic',options = ['Domestic', 'NonDomestic','All'], default=['All'])
OptionType = st.sidebar.multiselect('choose the OptionType',options = ['Vanilla', 'Others', 'Puttable', 'Callable', 'Callable&Puttable', 'All'], default=['All'])
Rating = st.sidebar.multiselect('choose the Rating',options = ['A_andBetter', 'N/A', 'BBB', 'B_andWorse', 'BB', 'All'], default=['All'])
PDiR = st.sidebar.multiselect('choose the PDiR',options = ['A_andBetter', 'BBB', 'BB', 'B_andWorse', 'N/A', 'All'], default=['All'])
TimeToMaturity = st.sidebar.multiselect('choose the TimeToMaturity',options = ['4To12M', 'LessThan3M', '1-3Y', '3-5Y', 'MoreThan5Y', 'N/A', 'All'], default=['All'])

CouponChg = st.sidebar.multiselect('choose the CouponChg',options = ['N/A', 'Change', 'All'], default=['All'])
Is_Public_Issued = st.sidebar.multiselect('choose the Is_Public_Issued',options = [True, False, 'All'], default=['All'])
multiValuation = st.sidebar.multiselect('choose the multiValuation',options = [False, True, 'All'], default=['All'])
valEndDate = st.sidebar.multiselect('choose the valEndDate>Maturity',options = [False, True, 'All'], default=['All'])

Fix_Type = st.sidebar.multiselect('choose the Fix_Type',options = ['Rating&PDiR', 'Rating', 'PDiR', 'All'], default=['Rating&PDiR'])
run = st.sidebar.button('Run')
bonddata = get_bonddata()
if run:
    print(bonddata.columns)
    generate_result(bonddata, imonth, Model, Period, isDomestic, OptionType, Rating, PDiR, TimeToMaturity, CouponChg, Is_Public_Issued, multiValuation, valEndDate, Fix_Type)






                # num_orig = tmp[itype + ['Yield', 'Flag']].groupby(itype).count().reset_index().rename(columns = {'Yield': 'Original_Obs', 'Flag':'Type'})
                # num_orig.loc[num_orig['Type']==0, 'Type'] = '<Q25'
                # num_orig.loc[num_orig['Type']==1, 'Type'] = 'Q25_Q50'
                # num_orig.loc[num_orig['Type']==2, 'Type'] = 'Q50_Q75'
                # num_orig.loc[num_orig['Type']==3, 'Type'] = '>Q75'












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

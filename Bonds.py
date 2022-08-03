import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dateutil import relativedelta
import gc
import math
import seaborn as sns
#streamlit run E:\Mengqi\Streamlit\Bonds.py
#streamlit run E:\Mengqi\Streamlit\Bonds.py
PAGE_CONFIG = {"page_title": "Hello",
               "page_icon": ":smiley:", "layout": "centered"}

st.set_page_config(**PAGE_CONFIG)

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

def generate_type_matrix(tmp_df, Fix_Type, col):
    df_all = pd.DataFrame()
    for iType in Fix_Type:
        if iType == 'Rating&PDiR':
            df = cal_matrix_num(tmp_df[(tmp_df.Rating1_x == tmp_df.Rating1_y)&(tmp_df.PDiR1_x == tmp_df.PDiR1_y)], col)
            df['Fix_Type'] = 'Rating&PDiR'
        elif iType == 'Rating':
            df = cal_matrix_num(tmp_df[(tmp_df.Rating1_x == tmp_df.Rating1_y)], col)
            df['Fix_Type'] = 'Rating'
        elif iType == 'PDiR':
            df = cal_matrix_num(tmp_df[(tmp_df.PDiR1_x == tmp_df.PDiR1_y)], col)
            df['Fix_Type'] = 'PDiR'
        elif iType == 'All':
            df = cal_matrix_num(tmp_df, col)
            df['Fix_Type'] = 'All'
        df_all = pd.concat([df_all, df], axis = 0)
    df_all = df_all.set_index(['Fix_Type', 'Type'])
    return df_all
        
        

def cal_matrix_num(tmp_df, col):
    total_num = tmp_df.shape[0]
    df = pd.DataFrame(columns = ['<Q25 (obs, std)', 'Q25_Q50 (obs, std)', 'Q50_Q75 (obs, std)', '>Q75 (obs, std)', 'Sum_obs'])
    df['Type'] = ['<Q25', 'Q25_Q50', 'Q50_Q75', '>Q75']
    for i in range(4):
        num_obs = 0
        for j in range(4):
            aa = tmp_df[(tmp_df.Flag_x==i)&(tmp_df.Flag_y==j)]
            num_obs = num_obs + aa.shape[0]
        for j in range(4):
            aa = tmp_df[(tmp_df.Flag_x==i)&(tmp_df.Flag_y==j)]
            df.iloc[i,j] = "{rate:.1f}% ({num:.0f}, {std:.4f})".format(rate = remove_0(aa.shape[0]*100, num_obs), num = aa.shape[0], std = aa[col].std())
        df.iloc[i,4] = num_obs
    return df

def generate_type(bonddata, Model, Period, isDomestic, OptionType, Rating, PDiR, TimeToMaturity, CouponChg, Is_Public_Issued, multiValuation, valEndDate):
    itype = ['All']
    sort_dict = {}
    if 'All' not in Model:
        bonddata = bonddata[bonddata['Model'].isin(Model)]
        itype.append("Model")
        sort_dict["Model"] = Model
    if 'All' not in Period:
        bonddata = bonddata[bonddata['Period'].isin(Period)]
        itype.append("Period")
        sort_dict["Period"] = Period
    if 'All' not in isDomestic:
        bonddata = bonddata[bonddata['isDomestic'].isin(isDomestic)]
        itype.append("isDomestic")
        sort_dict["isDomestic"] = isDomestic
    if 'All' not in OptionType:
        bonddata = bonddata[bonddata['OptionType'].isin(OptionType)]
        itype.append("OptionType")
        sort_dict["OptionType"] = OptionType
    if 'All' not in Rating:
        bonddata = bonddata[bonddata['Rating'].isin(Rating)]
        itype.append("Rating")
        sort_dict["Rating"] = Rating
    if 'All' not in PDiR:
        bonddata = bonddata[bonddata['PDiR'].isin(PDiR)]
        itype.append("PDiR")
        sort_dict["PDiR"] = PDiR
    if 'All' not in TimeToMaturity:
        bonddata = bonddata[bonddata['TimeToMaturity'].isin(TimeToMaturity)]
        itype.append("TimeToMaturity")
        sort_dict["TimeToMaturity"] = TimeToMaturity
    if 'All' not in CouponChg:
        bonddata = bonddata[bonddata['CouponChg'].isin(CouponChg)]
        itype.append("CouponChg")
        sort_dict["CouponChg"] = CouponChg
    if 'All' not in Is_Public_Issued:
        bonddata = bonddata[bonddata['Is_Public_Issued'].isin(Is_Public_Issued)]
        itype.append("Is_Public_Issued")
        sort_dict["Is_Public_Issued"] = Is_Public_Issued
    if 'All' not in multiValuation:
        bonddata = bonddata[bonddata['multiValuation'].isin(multiValuation)]
        itype.append("multiValuation")
        sort_dict["multiValuation"] = multiValuation
    if 'All' not in valEndDate:
        bonddata = bonddata[bonddata['valEndDate>Maturity'].isin(valEndDate)]
        itype.append("valEndDate>Maturity")
        sort_dict["valEndDate>Maturity"] = valEndDate
    return itype, bonddata, sort_dict

def generate_result(col, bonddata, imonth, itype, Fix_Type, sort_dict):

    cut_b = bonddata.groupby(itype + ['yearmonth']).agg({f'{col}': [q25, q50, q75]}).reset_index()
    cut_b.columns = [''.join(col) for col in cut_b.columns]

    tmp = bonddata[itype + ['yearmonth', col, 'bondID', 'YYYYMMDD', 'valEndDate']].merge(cut_b, on = itype + ['yearmonth'], how = 'left')
    cut_b = []
    tmp['Q25'] = tmp[col] - tmp[f'{col}q25']
    tmp['Q50'] = tmp[col] - tmp[f'{col}q50']
    tmp['Q75'] = tmp[col] - tmp[f'{col}q75']

    tmp['Flag'] = (tmp[['Q25', 'Q50', 'Q75']]>0).sum(axis = 1)
    tmp = tmp[itype + ['yearmonth', 'Flag', 'bondID', 'YYYYMMDD','valEndDate', col]]
    # tmp = tmp.merge(bonddata[itype+ ['bondID','YYYYMMDD', 'valEndDate']], how = 'left', on = ['bondID','YYYYMMDD', 'valEndDate']+itype)
    num_orig = tmp[itype + [col, 'Flag']].groupby(itype + ['Flag']).count().reset_index().rename(columns = {col: 'Original_obs', 'Flag':'Type'})
    num_orig.loc[num_orig['Type']==0, 'Type'] = '<Q25'
    num_orig.loc[num_orig['Type']==1, 'Type'] = 'Q25_Q50'
    num_orig.loc[num_orig['Type']==2, 'Type'] = 'Q50_Q75'
    num_orig.loc[num_orig['Type']==3, 'Type'] = '>Q75'
    print(num_orig)

    tmp = tmp.merge(bonddata[['multiValuation', 'bondID','YYYYMMDD',f'{imonth}M_later', 'valEndDate', 'Rating', 'PDiR']].rename(columns = {'Rating':'Rating1', 'PDiR':'PDiR1', 'multiValuation': 'multiValuation1'}), how = 'left', on = ['bondID','YYYYMMDD', 'valEndDate'])


    tmp1 = tmp.loc[~tmp.multiValuation1, itype + ['bondID','yearmonth', f'{imonth}M_later', 'valEndDate', 'Rating1', 'PDiR1', 'Flag', col]].merge(tmp.loc[~tmp.multiValuation1,['bondID', 'yearmonth', 'Rating1', 'PDiR1', 'Flag']].rename(columns = {'yearmonth':f'{imonth}M_later'}), on = ['bondID', f'{imonth}M_later'], how = 'inner')
    tmp2 = tmp.loc[tmp.multiValuation1, itype + [ 'bondID','yearmonth', f'{imonth}M_later', 'valEndDate', 'Rating1', 'PDiR1' , 'Flag', col]].merge(tmp.loc[tmp.multiValuation1,['bondID', 'yearmonth', 'valEndDate', 'Rating1', 'PDiR1', 'Flag']].rename(columns = {'yearmonth':f'{imonth}M_later'}), on = ['bondID', f'{imonth}M_later', 'valEndDate'], how = 'inner')
    tmp = pd.concat([tmp1, tmp2], axis = 0)
    if 'Rating&PDiR' not in Fix_Type:
        tmp = tmp[tmp.Rating1_x.isin(['A_andBetter', 'BBB', 'BB', 'B_andWorse'])]
        tmp = tmp[tmp.PDiR1_x.isin(['A_andBetter', 'BBB', 'BB', 'B_andWorse'])]
    if 'Rating' not in Fix_Type:
        tmp = tmp[tmp.Rating1_x.isin(['A_andBetter', 'BBB', 'BB', 'B_andWorse'])]
    if 'PDiR' not in Fix_Type:
        tmp = tmp[tmp.PDiR1_x.isin(['A_andBetter', 'BBB', 'BB', 'B_andWorse'])]

    df = tmp.groupby(itype).apply(lambda x: generate_type_matrix(x, Fix_Type, col)).reset_index()
    df = df.merge(num_orig, on = itype + ['Type'], how = 'left')

    index = []
    for ikey in sort_dict.keys():
        tmp = {}
        i = 0
        for ivalues in sort_dict[ikey]:
            tmp[ivalues] = i
            i = i + 1
        print(tmp)
        df[f'index_{ikey}'] = df[ikey].apply(lambda x: tmp[x])
        index.append(f'index_{ikey}')

    tmp = {'<Q25':0, 'Q25_Q50':1, 'Q50_Q75':2, '>Q75':3}
    df['index_sort1'] = df['Type'].apply(lambda x: tmp[x])
    df = df.sort_values(index + ['index_sort1'])
    df = df.drop(index + ['index_sort1', 'All'], axis = 1).reset_index(drop = True)
    df['Sum_obs/ori_obs(%)'] = (df['Sum_obs']/ df['Original_obs']).apply(lambda x: '{rate:.2f}%'.format(rate = x*100))
    st.dataframe(df)
    gc.collect()

def generate_stat(col, bonddata, itype, sort_dict, total_num):
    # pd.set_option('display.float_format', '{:.2g}'.format)
    print(itype)
    tmp_df = bonddata[[f'{col}', 'CompanyID', 'bondID']+itype].groupby(itype).agg({f'{col}': ['min', q1, q5, q10, q25, q50, q75, q90, q95, q99, 'max', 'mean', 'std'], 'CompanyID': ['nunique'],
    'bondID': ['nunique', 'count']}).reset_index()
    tmp_df.columns = ['_'.join(col) for col in tmp_df.columns]
    # tmp_df.to_csv(r'E:\Mengqi\CDS\DAS\DAS\Analysis\Domestic bond\tmp_df.csv', index = False)
    # tmp_df = pd.read_csv(r'E:\Mengqi\CDS\DAS\DAS\Analysis\Domestic bond\tmp_df.csv')
    for i in itype:
        if i + '_' in tmp_df.columns:
            tmp_df = tmp_df.rename(columns = {i + '_':i})
    tmp_df[itype] = tmp_df[itype].fillna('All')
    
    print(tmp_df)
    # tmp_df.loc[tmp_df.isDomestic_==1, 'isDomestic_'] = 
    # tmp_df = pd.read_csv(r'E:\Mengqi\CDS\DAS\DAS\Analysis\Domestic bond\tmp_df.csv')
    # tmp_df.to_csv(r'E:\Mengqi\CDS\DAS\DAS\Analysis\Domestic bond\tmp_df_fill.csv', index = False)
    tmp_df['%Obs(byAll)'] =  (tmp_df['bondID_count']/total_num).apply(lambda x: "{rate:.1f}%".format(rate = x*100))
    tmp_df = tmp_df.rename(columns = { 'CompanyID_nunique':'#Firm','bondID_nunique':'#Bond', 'bondID_count': '#Obs'})

    index = []
    for ikey in sort_dict.keys():
        tmp = {}
        i = 0
        for ivalues in sort_dict[ikey]:
            tmp[ivalues] = i
            i = i + 1
        print(tmp)
        tmp_df[f'index_{ikey}'] = tmp_df[ikey].apply(lambda x: tmp[x])
        index.append(f'index_{ikey}')
    
    tmp_df = tmp_df.sort_values(index + ['All'])

    tmp_df = tmp_df.drop(index + ['All'], axis = 1).reset_index(drop = True)
    # st.dataframe(tmp_df.style.highlight_max(axis=0))'
    for icol in  [f'{col}_q1', f'{col}_q5', f'{col}_q10', f'{col}_q25', f'{col}_q50', f'{col}_q75', f'{col}_q90', f'{col}_q95', f'{col}_q99', f'{col}_min', f'{col}_max', f'{col}_mean']:
        tmp_df.loc[:,icol] = (tmp_df.loc[:, icol]*100)
        tmp_df = tmp_df.rename(columns = {icol: icol+'(%)'})
    
    # st.dataframe(tmp_df.style.format('{:.3f}', na_rep="", subset = [f'{col}_q1'])\
    #      .bar(align=0, vmin=-2.5, vmax=2.5, cmap="bwr", height=50,
    #           width=60, props="width: 120px; border-right: 1px solid black;")\
    #      .text_gradient(cmap="bwr", vmin=-2.5, vmax=2.5))
    # st.dataframe(tmp_df.style.bar( color='#d65f5f'))
    # AgGrid(tmp_df)
    # df2 = pd.DataFrame(np.random.randn(10,4), columns=['A','B','C','D'])
    cm = sns.light_palette((260, 75, 60), input="husl", as_cmap=True)

    st.dataframe(tmp_df.style.background_gradient(subset = [f'{col}_q1(%)', f'{col}_q5(%)', f'{col}_q10(%)', f'{col}_q25(%)', f'{col}_q50(%)', f'{col}_q75(%)', f'{col}_q90(%)', f'{col}_q95(%)', f'{col}_q99(%)'], cmap=cm)\
        .format('{:.2f}', na_rep="", subset = [f'{col}_q1(%)', f'{col}_q5(%)', f'{col}_q10(%)', f'{col}_q25(%)', f'{col}_q50(%)', f'{col}_q75(%)', f'{col}_q90(%)', f'{col}_q95(%)', f'{col}_q99(%)', 
        f'{col}_min(%)', f'{col}_max(%)', f'{col}_mean(%)', f'{col}_std']))
    gc.collect()

def generate_plot(bonddata, col, itype):
    if len(itype)>1:
        itype.remove("All")
    groups = bonddata[[f'{col}', 'CompanyID', 'bondID']+itype].groupby(itype)
    rowlength = math.ceil(groups.ngroups/5)
    fig = make_subplots(rows = rowlength , cols = 5)
    icol = 1
    irow = 1
    for key in groups.groups.keys():
        print(key, icol, irow)
        tmp = groups.get_group(key)
        print(tmp)
        if len(itype)>1:
            fig.add_trace(go.Box(y = tmp['Yield'], name = '_'.join(key)), row = irow, col = icol)
        else:
            fig.add_trace(go.Box(y = tmp['Yield'], name = key), row = irow, col = icol)
        if icol==5:
            irow = irow + 1
            icol = 0
        icol = icol + 1

    fig.update_layout(title = 'Boxplot for {}'.format(col))
    fig.update_yaxes(title_text = col)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    # fig.write_html(r'E:\Mengqi\CDS\DAS\DAS\Analysis\P3_{}.html'.format('type'), auto_open = True)

    
@st.cache
def get_bonddata():
    bonddata = pd.DataFrame()
    for i in range(11):
        print(i)
        tmp = pd.read_parquet('bonddata{}.parquet.gzip'.format(i))
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
run = st.sidebar.button('Click and Run', help = 'If you finish selection below, please click run button')
col = st.sidebar.radio('Choose the values',options = ['Yield', 'GS', 'ZS'])
if col == 'ZS':
    col = 'Spread'
imonth = st.sidebar.selectbox('Choose the Month',options = [3,6,12]) 
# imonth = imonth[0]
# Model = st.sidebar.multiselect('Choose the Model',options = list(bonddata['Model'].unique())+['All'], default=['All'])
Model = st.sidebar.multiselect('Choose the Model',options = ['CORP', 'LGFV', 'FINA', 'N/A', 'All'], default=['All'])
Period = st.sidebar.multiselect('Choose the Period',options = ['Before2018', 'After2018', 'All'], default=['All'])
isDomestic = st.sidebar.multiselect('Choose the isDomestic',options = ['Domestic', 'NonDomestic','All'], default=['All'])
OptionType = st.sidebar.multiselect('Choose the OptionType',options = ['Vanilla', 'Others', 'Puttable', 'Callable', 'Callable&Puttable', 'All'], default=['All'])
Rating = st.sidebar.multiselect('Choose the Rating',options = ['A_andBetter', 'N/A', 'BBB', 'B_andWorse', 'BB', 'All'], default=['All'], help = 'This is the external rating from S&P, Moody, Fitch')
PDiR = st.sidebar.multiselect('Choose the PDiR',options = ['A_andBetter', 'BBB', 'BB', 'B_andWorse', 'N/A', 'All'], default=['All'], help = 'This is PDiR from irap China')
TimeToMaturity = st.sidebar.multiselect('Choose the TimeToMaturity',options = ['4To12M', 'LessThan3M', '1-3Y', '3-5Y', 'MoreThan5Y', 'N/A', 'All'], default=['All'])

CouponChg = st.sidebar.multiselect('Choose the CouponChg',options = ['N/A', 'Change', 'All'], default=['All'], help = 'Whether issuers have the right to change the coupon rate')
Is_Public_Issued = st.sidebar.multiselect('Choose the Is_Public_Issued',options = [True, False, 'All'], default=['All'], help = 'Whether the bonds are public issued')
multiValuation = st.sidebar.multiselect('Choose the multiValuation',options = [False, True, 'All'], default=['All'], help = 'Whether the bonds has multiple valuation on same date')
valEndDate = st.sidebar.multiselect('Choose the valEndDate>Maturity',options = [False, True, 'All'], default=['All'], help = 'Whether the valEndDate of bonds are larger than Maturity date')

Fix_Type = st.sidebar.multiselect('Choose the Fix_Type',options = ['Rating&PDiR', 'Rating', 'PDiR', 'All'], default=['Rating&PDiR'])

bonddata = get_bonddata()
total_num = bonddata.shape[0]
if run:
    print(bonddata.columns)
    itype, bonddata, sort_dict = generate_type(bonddata, Model, Period, isDomestic, OptionType, Rating, PDiR, TimeToMaturity, CouponChg, Is_Public_Issued, multiValuation, valEndDate)
    generate_result(col, bonddata, imonth, itype, Fix_Type, sort_dict)
    generate_stat(col, bonddata, itype, sort_dict, total_num)
    # generate_plot(bonddata, col, itype)





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

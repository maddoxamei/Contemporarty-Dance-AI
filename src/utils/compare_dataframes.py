def _get_df_differences(df1, df2, threshold=None):
    indices = []
    diff = df1-df2
    if threshold:
        diff = diff[(abs(diff)>threshold).any(axis='columns')]
    else:
        diff = diff[diff.fillna(0).astype(bool).any(axis='columns')]
    """for i in range(len(df1)):
        row = df1.iloc[i] - df2.iloc[i]
        if(row.any()):
            indices.append(i)
            #print(i,"==========",sum(row),"\n", row)
    #diff = df1.merge(df2,indicator = True, how='left').loc[lambda x : x['_merge']!='both'].drop('_merge', axis=1)
    if len(indices)!=0:
        diff.index = indices"""
    print("============ The differences are as follows ============")
    print(diff)
    return diff
"""df1=pd.DataFrame({'A':[1.00007,2.0,.02,4.0],'B':[2.0,3.0,4.0,3.0],'C':[-2.7,8.0,1.3,5.6]})
df2=pd.DataFrame({'A':[1.0,.7,.02,4.0],'B':[2.0,3.0,4.0,4.0]})
get_df_differences(df1,df2)
get_df_differences(df1,df2,0.0005)"""

def print_header(header):
    side = "======================="
    filler = "="*(len(header)+2)
    print(side+filler+side)
    print(side,header,side)
    print(side+filler+side)

def _print_sub_header(sub_header):
    side = "***********************"
    print(side,sub_header,side)
    
def compute_differences(df1, df2, sub_header, thresh=0.0005):
    _print_sub_header(sub_header)
    _get_df_differences(df1, df2, thresh)
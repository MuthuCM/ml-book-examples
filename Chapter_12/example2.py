# Example 13.1
# Load Packages
# load dataset
dataset = pd.read_csv('LoansData.csv.gz', compression='gzip', 
                                                low_memory=True)
#dataset.shape
dataset[ 'loan_status' ].value_counts(dropna=False)
dataset = dataset.loc[dataset [ 'loan_status' ].isin([ 'Fully Paid',  
                                                    'Charged Off' ]) ]
dataset[ 'loan_status' ].value_counts(normalize=True, dropna=False)

dataset[ 'charged_off' ] = (dataset[ 'loan_status' ] ==
                                'Charged Off').apply(np.uint8)
dataset.drop('loan_status', axis=1, inplace=True)

missing_fractions = dataset.isnull().mean( ).sort_values(ascending=False)
drop_list=sorted(list(missing_fractions[missing_fractions> 0.3].index))
dataset.drop(labels=drop_list, axis=1, inplace=True)
dataset.shape

keep_list = [ 'charged_off', 'funded_amnt', 'addr_state', 'annual_inc',    
             'application_type', 'dti', 'earliest_cr_line',   
             'emp_length', 'emp_title', 'fico_range_high', 
             'fico_range_low','grade','home_ownership','id',        
             'initial_list_status', 'installment',
             'int_rate', 'loan_amnt', 'loan_status', 'mort_acc',  
             'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 
             'purpose', 'revol_bal', 'revol_util', 
             'sub_grade', 'term', 'title', 'total_acc', 
             'verification_status', 'zip_code', 'last_pymnt_amnt', 
             'num_actv_rev_tl', 'mo_sin_rcnt_rev_tl_op', 
             'mo_sin_old_rev_tl_op', "bc_util", "bc_open_to_buy", 
             "avg_cur_bal", "acc_open_past_24mths" ]
drop_list = [col for col in dataset.columns if col not in keep_list]
dataset.drop(labels=drop_list, axis=1, inplace=True)
dataset.shape

correlation = dataset.corr( )
correlation_chargeOff = abs(correlation[ 'charged_off' ])
drop_list_corr = sorted (list[correlation_chargeOff < 0.03] . index) 
print( drop_list_corr)

dataset.drop(labels=drop_list_corr, axis=1, inplace=True)
dataset[ [ 'id', 'emp_title', 'title', 'zip_code' ] ] .describe( )
dataset.drop([ 'id', 'employment', 'title', 'zip_code' ], axis=1,            
                                                         inplace=True)

dataset[ 'term' ] = dataset[ 'term' ].apply(lambda s: 
                                        np.int8(s.split( )[0] ) )
dataset.groupby('term')['charged_off'].value_counts(normalize=True).loc[ :,1 ]

dataset['emp_length'].replace(to_replace='10+ years', 
                              value= '10 years', inplace=True )
	
dataset['emp_length'].replace('< 1 year' , '0 years', inplace=True )
def emp_length_to_int( s ) :
	if pd.isnull( s ) : 
		return s
	else:
		return np.int8( s.split( ) [ 0 ] )
dataset['emp_length'] = dataset[ 'emp_length' ].apply(emp_length_to_int)
charge_off_rates=dataset.groupby('emp_length')['charged_off'].value_counts(normalize=True).loc[ :,1 ]
sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values )

dataset.drop(['emp_length'], axis=1, inplace=True)
charge_off_rates = dataset.groupby('sub_grade')['charged_off'].value_counts(normalize=True).loc[ :,1 ]                                                                                              
sns.barplot( x=charge_off_rates.index, y=charge_off_rates.values )

dataset[ [ 'annual_inc' ] ] .describe( )
dataset['log_annual_inc']=dataset['annual_inc'].apply(lambda x: np.log10(x+1) )
dataset.drop( 'annual_inc', axis=1, inplace=True )

dataset[ [ 'fico_range_low', 'fico_range_high'] ].corr( )
dataset[ 'fico_score' ] = 0.5*dataset[ 'fico_range_low' ] + 0.5*dataset[ 'fico_range_high' ]
dataset.drop(['fico_range_high, fico_range_low'],axis=1,inplace=True )

From import LabelEncoder
# Create Categorical boolean mask
categorical_feature_mask = dataset.dtypes==object
# Filter categorical columns using mask and turn it into a list
categorical_cols = dataset.columns[categorical_feature_mask].tolist()

loanstatus_0 = dataset[dataset[ "charged_off"]==0 ]
loanstatus_1 = dataset[dataset[ "charged_off"]==1 ]
subset_of_loanstatus_0 = loanstatus_0.smple(n=5500)
subset_of_loanstatus_1 = loanstatus_1.smple(n=5500)
dataset = pd.concat([subset_of_loanstatus_1, subset_of_loanstatus_0 ])
dataset = dataset.sample(frac=1).reset_index(drop=True)
print( "Current shape of dataset :", dataset.shape )


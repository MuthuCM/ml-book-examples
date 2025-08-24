# Example 16.2
# Association Rule Mining - FP Growth Method
# Load Packages
# Step 1: import the package
import pandas as pd
# read the dataset
dataset = pd.read_csv( "Market_Basket_Optimisation.csv" )
dataset.shape  # (7500, 20)
#dataset.head( )

# Step 2: Gather all items of each Transaction into Numpy Array
transaction = [ ]
for i in range (0, dataset.shape[0] ) :
	for j in range (0, dataset.shape[ i ]) : 
		transaction.append (dataset.values [ i, j ] )
transactions = np.array( transaction )
print (transaction)

# Step 3: Transform Numpy Array into a Pandas DataFrame
df=pd.DataFrame ( transaction, colums = [ "items" ] )
# Put 1 to each item for making Countable Table, to be able to perform
# Group By operation
df[ "incident_count" ] = 1
# Delete NaN items from Dataset
indexNames = df[ df [ "items" ] == "nan" ] .index
df.drop( indexNames, inplace = True)

# Step 4:Making a New Data Frame (for visualizations)
df_table = df.groupby( "items" ).sum( ).sort_values( "incident_count",           
                       ascending=False). reset_index ( )
	
# Step 5:Intial Visualization
df_table.head(5).style.background_gradient( cmap = 'Blues' )
# Plot the Tree map
import plotly.express as px
df_table [ "all" ] = "Top 50 items"
fig = px.treemap( df_table(50), path=[ 'all', "items" ], 
				values = 'incident_count', 
				colour = df_table[ "incident_count" ].head(50), 
				hover_data = [ 'items' ],
				colour_continuous_scale = 'Blues',
			   ) 	
fig.show( )

# Step 6: Transform Every Transaction to a separate List & Gather them
# into a NumpyArray
transaction = [ ]
for i in range (dataset.shape[o] ):
	   transaction.append ( [structure(dataset.values [i, j] ) for j in                     
                                             range(dataset.shape[1]) ]) 
transaction = np.array( transaction )
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder( )
dataset = pd.DataFrame( te_ary, columns = te,columns_ )
dataset.head( )      # dataset after encoding is done

# Step 7 
# After encoding, there are 121 items/features
# Extracting the frequent items from 121 is difficult
# So, we will select the top 30 items & apply the FP-growth algorithm
# to find the most frequent items
# Select top 30 items
df_table =
first 30 = df_table [ "items" ].head(30).values
# Extract Top 30
dataset = dataset.loc[ :, first30 ]
# Display shape of the dataset
dataset.shape # (7500, 30)

# Step 8: Implement FP Growth algorithm 
from mlxtend.frequent_patterns import fpgrowth
res = fpgrowth(dataset, min_support = 0.05, use_colnames = True)
# print top 10
res.head( 10 )

# Step 9
#Create different Association Rules from these frequently occurring items
from mlxtend.frequent_patterns import association_rules
res = association_rules (res, metric = "lift", min_threshold=1)
res 

# Step 10
# Display Association Rules
# Sort values based on confidence to know which items are more related
res.sort-values ("confidence", ascending=False)
# {spaghetti}  {mineral water} has the highest confidence
# These two items are more related to each other
# FP-growth is an improved version of the Apriori algorithm
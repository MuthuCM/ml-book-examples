# Example 16.1
# Association Rule Mining - Apriori Method
# Load Packages
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
# Define the Dataset
dataset = [	['Milk', 'Onion', 'Bread', 'Beans', 'Eggs', 'Yoghurt'],
	['Fish', 'Onion', 'Bread', 'Beans', 'Eggs', 'Yoghurt'],
	['Milk', 'Apples', 'Beans', 'Eggs'],
	['Milk', 'Sugar', 'Tea Leaves', 'Beans', 'Yoghurt'],
	['Tea Leaves', 'Onion', 'Beans','Ice Cream', 'Eggs']]
# One_Hot Encoding of data
tr = TransactionEncoder ()
tr_arr = tr.fit (dataset).transform (dataset)
df = pd.DataFrame (tr_arr, columns = tr.columns_)
df

# Applying Apriori
from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
frequent_itemsets

# Displaying Association Rules 
from mlxtend.frequent_patterns import association_rules
result = association_rules (frequent_itemsets, metric='lift',
                            min_threshold=1)
print (result)

# Sort the Association Rules based on Confidence
result.sort_values ("confidence", ascending = False)

# Sort the Association Rules based on Lift

result.sort_values ("lift", ascending=False)

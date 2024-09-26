import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Exercise 1: Concatenate
# creating the data frames: 
df1 = pd.DataFrame([['a', 1], ['b', 2]],
                   columns=['letter', 'number'])
df2 = pd.DataFrame([['c', 1], ['d', 2]],
                   columns=['letter', 'number'])

# concatenate the data frames concatenate along the rows (axis = 0)
result = pd.concat([df1,df2], axis=0)

# re-setting the indices: 
# drop = True means that the old index will be removed and not added as a new column
# inplace = True means that the changes will be made directly to the result DataFrame without creating a new DataFrame.
result.reset_index(drop=True, inplace=True)

print(result)

# Exercise 2 : Merge
# Step 1: Creating the dictionaries 
#df1
df1_dict = {
    'id': ['1', '2', '3', '4', '5'],
    'Feature1': ['A', 'C', 'E', 'G', 'I'],
    'Feature2': ['B', 'D', 'F', 'H', 'J']
}

df1 = pd.DataFrame(df1_dict, columns = ['id', 'Feature1', 'Feature2'])

#df2
df2_dict = {
        'id': ['1', '2', '6', '7', '8'],
        'Feature1': ['K', 'M', 'O', 'Q', 'S'],
        'Feature2': ['L', 'N', 'P', 'R', 'T']}

df2 = pd.DataFrame(df2_dict, columns = ['id', 'Feature1', 'Feature2'])

# Step 2: Merge for the first output (inner merge)
inner_merge_result = pd.merge(df1,df2, on="id",suffixes=('_x','_y'))
print("Inner Merge Result:")
print(inner_merge_result)

# Step 3: Merge for the second output (outer merge)
outer_merge_result = pd.merge(df1,df2, on='id',how='outer',suffixes=('_df1','_df2'))
print("Outer Merge Result:")
print(outer_merge_result)


# Exercise 3: Merge MultiIndex
#generate days
all_dates = pd.date_range('2021-01-01', '2021-12-15')
business_dates = pd.bdate_range('2021-01-01', '2021-12-31')

#generate tickers
tickers = ['AAPL', 'FB', 'GE', 'AMZN', 'DAI']

#create indexs
index_alt = pd.MultiIndex.from_product([all_dates, tickers], names=['Date', 'Ticker'])
index = pd.MultiIndex.from_product([business_dates, tickers], names=['Date', 'Ticker'])

# create DFs
market_data = pd.DataFrame(index=index,
                        data=np.random.randn(len(index), 3),
                        columns=['Open','Close','Close_Adjusted'])

alternative_data = pd.DataFrame(index=index_alt,
                                data=np.random.randn(len(index_alt), 2),
                                columns=['Twitter','Reddit'])

# merge data frames: 
merged_data = pd.merge(market_data,alternative_data, how='left',left_index=True,right_index=True)

#Fill missing values with 0
merged_data.fillna(0, inplace=True)

print(merged_data)

# Exercise 4: 
def winsorize(df, quantiles):
    lower_bound = np.percentile(df, quantiles[0] * 100)  # Calculate lower percentile
    upper_bound = np.percentile(df, quantiles[1] * 100)  # Calculate upper percentile
    return df.clip(lower=lower_bound, upper=upper_bound)  # Clip values

df = pd.DataFrame(range(1, 11), columns=['sequence'])
print(winsorize(df['sequence'], [0.20, 0.80]).to_markdown())


groups = np.concatenate([np.ones(10), np.ones(10)+1,  np.ones(10)+2, np.ones(10)+3, np.ones(10)+4])

df = pd.DataFrame(data= zip(groups,
                            range(1,51)),
                columns=["group", "sequence"])

result = df.groupby('group')['sequence'].apply(lambda x: winsorize(x, [0.05, 0.95]))

print(result.to_markdown())

# Exercise 5
# Define the data
data = {
    'value': [20.45, 22.89, 32.12, 111.22, 33.22, 100.00, 99.99],
    'product': ['table', 'chair', 'chair', 'mobile phone', 'table', 'mobile phone', 'table']
}

# Create the DataFrame
df = pd.DataFrame(data)
# Compute min, max, and mean without MultiIndex columns
result = df.groupby('product')['value'].agg(min_value='min', max_value='max', mean_value='mean').reset_index()

# Display the result
print(result)

# Exercise 6
# Step 1: Generate business dates and tickers
business_dates = pd.bdate_range('2021-01-01', '2021-12-31')
tickers = ['AAPL', 'FB', 'GE', 'AMZN', 'DAI']

# Step 2: Create MultiIndex
index = pd.MultiIndex.from_product([business_dates, tickers], names=['Date', 'Ticker'])

# Step 3: Create DataFrame
market_data = pd.DataFrame(index=index,
                           data=np.random.randn(len(index), 1),
                           columns=['Prediction'])

# Step 4: Unstack the DataFrame
unstacked_data = market_data.unstack()

# Step 5: Display the first 3 rows
print(unstacked_data.head(3))

# Step 6: Plot the time series
unstacked_data.plot(title='Predictions Over Time')
plt.show()
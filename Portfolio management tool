#Import numpy and pandas for calculations
import pandas as pd
import numpy as np

#import matplotlib and seaborn for charts
import matplotlib.pyplot as plt
import seaborn as sns

#upload files with input data
from google.colab import files
uploaded = files.upload()

#save unpoaded file in pandas dataframe
df1 = pd.read_excel("analisi corr Colab.xlsx", sheet_name = "dati_2")

#show columns of dataframe
df1.columns

#select input variables you want to include in the analysis
#comprised index and target variable
df = df1[['Index', 'EG', 'OAS', 'OAS_change', 'EUSA2', 'EUSA2_change',
       'Steepeness', 'Steepeness_change', 'Curvature', 'Curvature_change',
       'US_PMI', 'US_PMI_change', 'CPI', 'CPI_change', 'RealR','RealR_change']]

#set index 
df = df.set_index("Index")

#drop null values if present
df = df.dropna()

#define target variable
#in this case in the Euro Govt Index 1-3y (historical prices)
df_target = df[["EG"]]

#this code calculates the future period's perfromance of the target index (forward looking)
#what I need to know is the future performance of my target variable, given the current state of my input variables
df_target["shifted"] = df_target["EG"].shift(-1)  #-3
df_target[f"perf"] = ((df_target["shifted"] / df_target["EG"]) - 1) * 100

#drop target variable from df that now contains only input variables
df = df.drop("EG", axis=1)

#add the previous index performance to the input variables
df["shifted_input"] = df_target[["perf"]].shift(1)
#fillna for one shifted day
df=df.fillna(0)

#Should remove the last value becouse it is considered in future perf, and gives 0 for last values
last_day_to_remove = df.tail(1).index  ##ind last value
df = df.drop(last_day_to_remove)
df_target = df_target.drop(last_day_to_remove)

#show last 5 rows of dataframe
df.tail()

#print correlation of input variables with target
pd.merge(df, df_target["perf"], left_index = True, right_index = True).corr()[["perf"]].sort_values(by="perf", ascending=False)

#create function that divides the input variables into "n" parts
def define_categories(df, perc):

  #create a df for indices
  df_ind = df.copy()

  #select the variable
  for name_col in df.columns:

    #create new df for each loop
    new_df = df.copy()

    print(f"start with {name_col}")

    col = np.percentile(new_df[name_col], np.arange(20, 100, perc))
    print(f"col for {name_col} is {col}")

    for a,b in list(zip(col, np.arange(0,len(col)))):
      
      #step 0
      ind_to_change = new_df[new_df[name_col] < a].index

      #step 1
      df_ind.loc[ind_to_change,name_col] = b

      #step 2
      #drop previosly completed indexes
      new_df = new_df.drop(ind_to_change)


    #finish with last values
    df_ind.loc[new_df.index, name_col] = len(col)


  lenght = len(set(df_ind[df_ind.columns[0]]))
  return df_ind, lenght

#For our purpose we are going to divide the input dataframe in 3 parts, passing to the "perc" input 50 (50% of dataframe)
#the function will print values used to divide input variables.
new_df, lenght = define_categories(df, 50)

#create function that calculates the future performance of target variable for different states of each input variable in my dataframe

#the funtion will return 3 outputs:
# a) future performance;
# b) count of occurrences used in future performance calculations
# c) standard deviation of future performance

def calculate_future_return_1_variable(variable, df_target, lenght):

  #create new df
  df_results = pd.DataFrame(index = range(0,lenght),  #11
                          columns = [variable])
  
  df_count = pd.DataFrame(index = range(0,lenght),
                          columns = [variable])

  df_std = pd.DataFrame(index = range(0,lenght),
                          columns = [variable])


  for ind in range(0,lenght):

    try:
      
      ind_for_perf = new_df.reset_index().set_index(variable).loc[ind]["Index"]
      #attach mean returns
      df_results.loc[ind,variable] = np.mean(df_target.loc[ind_for_perf,"perf"])  #Here I can choose whether to caluculate the mean or the median of future returns
      #df_results.loc[ind,variable] = np.median(df_target.loc[ind_for_perf,"perf"]) #Here I can choose whether to caluculate the mean or the median of future returns
      df_count.loc[ind,variable] = len(np.unique(df_target.loc[ind_for_perf,"perf"]))
      df_std.loc[ind,variable] = np.std(df_target.loc[ind_for_perf,"perf"])

    except:
      pass

  #substitute zeros and round
  df_results = df_results.astype("float").round(3).fillna(0)
  df_std = df_std.astype("float").round(3).fillna(0)

  return df_results, df_count, df_std

#in a loop, call the function for all input variables in my dataframe
#the function will also print the "single rating factor", that is the sum of absolute values of future performance for each current state of the input variable
#the input variables with sum close to zero is probably not explanatory of our target variable
rating_single_factor = []

for variable in df.columns:
  a, b, c = calculate_future_return_1_variable(variable, df_target, lenght)
  rating_single_factor.append(abs(a).mean())
  print(a)
  print(b)

  a.plot(figsize=(8,4))
  plt.title(f"{variable} - returns")
  plt.gca().set_ylim(-0.5,0.5)
  plt.grid()

  #CAN PLOT COUNTS AND STD AS WELL

  # b.plot(figsize=(8,4))
  # plt.title(f"{variable} - count")
  # plt.grid()

  # c.plot(figsize=(8,4))
  # plt.title(f"{variable} - std")
  # plt.grid()

  plt.show();

rating_single_factor

#show last five values of divided df
#number 2 indicates high level/increase of the variable, 0 indicates low level/decrease. 1 indicates mean value
new_df.tail()

#calculates future performance considering 2 variables jointly

def calculate_future_return(new_df, df_target, lenght):

  #create new df
  df_results = pd.DataFrame(index = range(0,lenght),  #11
                          columns = range(0,lenght))
  
  df_count = pd.DataFrame(index = range(0,lenght),
                          columns = range(0,lenght))

  df_std = pd.DataFrame(index = range(0,lenght),
                          columns = range(0,lenght))


  var_1 = new_df.columns[0]
  print(var_1)
  var_2 = new_df.columns[1]
  print(var_2)

  for ind in range(0,lenght):

    for col in range(0,lenght):
      
      new_df.reset_index(inplace=True)

      try:
        ind_for_perf = new_df.set_index(var_1).loc[ind].set_index(var_2).loc[col]["Index"]

        #attach mean returns
        df_results.loc[ind,col] = np.mean(df_target.loc[ind_for_perf,"perf"])  #MEAN
        #df_results.loc[ind,col] = np.median(df_target.loc[ind_for_perf,"perf"])  #MEDIAN
        df_count.loc[ind,col] = len(np.unique(df_target.loc[ind_for_perf,"perf"]))
        df_std.loc[ind,col] = np.std(df_target.loc[ind_for_perf,"perf"])

      except:
        pass

      new_df.set_index("Index", inplace=True)


  #substitute zeros and round
  df_results = df_results.astype("float").round(3).fillna(0)
  df_std = df_std.astype("float").round(3).fillna(0)

  return df_results, df_count, df_std

expected_perf = []

for ind in range(len(new_df.columns)):

  for next in range(ind+1, len(new_df.columns)):

    print(ind, next)

    df_results, df_count, df_std = calculate_future_return(new_df[[new_df.columns[ind], new_df.columns[next]]], df_target, lenght)

    print(df_results)
    print(df_count)
    print(f"last value for {new_df.columns[ind]} is {list(new_df.tail(1)[new_df.columns[ind]])}")
    print(f"last value for {new_df.columns[next]} is {list(new_df.tail(1)[new_df.columns[next]])}")
    print(f"expected perf is {df_results.loc[list(new_df.tail(1)[new_df.columns[ind]]), list(new_df.tail(1)[new_df.columns[next]])]}")


    expected_perf.append(np.array(df_results.loc[list(new_df.tail(1)[new_df.columns[ind]]), list(new_df.tail(1)[new_df.columns[next]])]).item())
    
    # plotting the heatmap
    hm = sns.heatmap(data=df_results,
                    annot=True)
      
    # displaying the plotted heatmap
    plt.show()

#print resutls
print(f"print all results {expected_perf}")
print(f"the mean value of mean future performance is {(np.array(expected_perf).mean()).round(2)}")

#print last current states of input variables
new_df.tail(1)

pip install XlsxWriter

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('df_results.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df.to_excel(writer, sheet_name='data')
new_df.to_excel(writer, sheet_name='percentile')
df_target.to_excel(writer, sheet_name='returns')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
files.download('df_results.xlsx')


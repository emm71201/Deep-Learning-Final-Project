# EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def make_bar_plot(parameter):
    table = df.pivot_table(index=parameter,
                    aggfunc={parameter:'count'})

    table.columns = ['count']
    table.reset_index(inplace=True)

    sns.catplot(table, x=parameter, y="count", hue="target", kind="bar")
    return plt.show()

def make_displot(parameter, bin):
    sns.displot(df, x=parameter, bins = bin, hue="target", kde=False)
    
    return plt.show()

def make_histplot(parameter):
    sns.histplot(binwidth=0.5, x=parameter, hue="target", data=df, stat="count", multiple="stack")

    return plt.show()

FILE_NAME = 'data.csv'
df = pd.read_csv(FILE_NAME)

convert_dict = {'M/F': 'category'}
 
df = df.astype(convert_dict)

# Restrict to patients over 60
df = df[df['Age']>=60]

# Target plot
make_bar_plot('target')

# Sex plot
make_histplot('M/F')

# Age plot
make_displot('Age', 10)

# Education plot
make_histplot('Educ')

# Socioeconomic plot
make_histplot('SES')

# Mini-mental state evaluation plot
make_displot('MMSE', 10)

# Estimated total intracranial volume plot
make_displot('eTIV', 10)

# Normalized whole brain volume plot
make_displot('nWBV', 10)

#  Atlas scaling factor plot
make_displot('ASF', 10)

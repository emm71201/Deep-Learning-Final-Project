# EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_bar_plot(parameter):
    table = df.pivot_table(index=parameter,
                    aggfunc={parameter:'count'})

    table.columns = ['count']
    table.reset_index(inplace=True)

    sns.catplot(table, x=parameter, y="count", hue="target", kind="bar", palette=colors)
    return plt.show()

def make_displot(parameter):
    sns.displot(df, x=parameter, hue="target", kind="kde", palette=colors)
    #sns.displot(df, x=df[df['target']==0.5][parameter], palette=colors, kind="kde")
    #sns.displot(df, x=df[df['target']==1][parameter], palette=colors, kind="kde")

    return plt.show()

def make_histplot(parameter):
    sns.histplot(binwidth=0.5, x=parameter, hue="target", data=df, stat="count", multiple="stack", palette=colors)

    return plt.show()

colors = sns.color_palette("Set2")

FILE_NAME = 'data.csv'
df = pd.read_csv(FILE_NAME)

convert_dict = {'M/F': 'category'}
 
df = df.astype(convert_dict)

# Restrict to patients over 60
df = df[df['Age']>=60]

# Remove patients with target type 2
df = df[df['target']!=2]

# Target plot
make_bar_plot('target')

# Sex plot
make_histplot('M/F')

# Age plot
make_displot('Age')

# Education plot
make_histplot('Educ')

# Socioeconomic plot
make_histplot('SES')

# Mini-mental state evaluation plot
make_displot('MMSE')

# Estimated total intracranial volume plot
make_displot('eTIV')

# Normalized whole brain volume plot
make_displot('nWBV')

#  Atlas scaling factor plot
make_displot('ASF')

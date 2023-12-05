# EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# My setup of matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('text', usetex=False)
plt.rc('font', family='times')
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

def make_bar_plot(parameter, filename="Figures/bar_plot.png"):
    table = df.pivot_table(index=parameter,
                           aggfunc={parameter: 'count'})

    table.columns = ['count']
    table.reset_index(inplace=True)

    table['class'] = ['Non Dem.', 'Very Mild Dem.', 'Mild. Dem.']

    sns.catplot(table, x="class", y="count", hue="target", kind="bar", legend=False)
    plt.savefig(filename, bbox_inches='tight')
    return plt.show()

# def make_displot(parameter):
#     sns.displot(df, x=parameter, hue="target", kind="kde", palette=colors)
#     # sns.displot(df, x=df[df['target']==0.5][parameter], palette=colors, kind="kde")
#     # sns.displot(df, x=df[df['target']==1][parameter], palette=colors, kind="kde")
#
#     return plt.show()

def make_displot(parameter, filename=None, labels=['Non Dem.', 'Very Mild Dem.', 'Mild. Dem.']):
    sns.displot(df, x=parameter, hue="target", kind="kde", palette=colors, legend=False)
    plt.legend(labels=labels[::-1])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

    return plt.show()


def make_histplot(parameter):
    sns.histplot(binwidth=0.5, x=parameter, hue="target", data=df, stat="count", multiple="stack", palette=colors)

    return plt.show()

def make_histplot(parameter, filename=None, labels=['Non Dem.', 'Very Mild Dem.', 'Mild. Dem.']):
    #sns.histplot(binwidth=0.5, x=parameter, hue="target", data=df, stat="count", multiple="stack")
    sns.histplot(binwidth=0.5, x=parameter, hue="target", data=df, stat="count", multiple="stack", palette=colors,\
                 legend=False)
    plt.legend(labels=labels[::-1])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    return plt.show()


colors = sns.color_palette("Set2")

FILE_NAME = 'data.csv'
df = pd.read_csv(FILE_NAME)

convert_dict = {'M/F': 'category'}

df = df.astype(convert_dict)

# Restrict to patients over 60
df = df[df['Age'] >= 60]

# Remove patients with target type 2
df = df[df['target'] != 2]

# Target plot
make_bar_plot('target')

# Sex plot
make_histplot('M/F', filename='Figures/gender_dist.png')

# Age plot
make_displot('Age', filename='Figures/age_dist.png')

# Education plot
make_histplot('Educ', filename="Figures/educ_dist.png")

# Socioeconomic plot
make_histplot('SES', filename="Figures/ses.png")

# Mini-mental state evaluation plot
make_displot('MMSE', filename="Figures/mmse_dist.png")

# Estimated total intracranial volume plot
make_displot('eTIV',  filename="Figures/eTIV_dist.png")

# Normalized whole brain volume plot
make_displot('nWBV', filename="Figures/nWBV_dist")

#  Atlas scaling factor plot
make_displot('ASF',  filename="Figures/asf_dist.png")
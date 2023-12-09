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

label_encode = {'Non_Demented':0, 'Very_mild_Dementia':0.5, 'Moderate_Dementia':2, 'Mild_Dementia':1}

def make_bar_plot(parameter, filename="Figures/bar_plot.png"):
    table = df.pivot_table(index=parameter,
                           aggfunc={parameter: 'count'})

    table.columns = ['count']
    table.reset_index(inplace=True)

    table['class'] = ['Non Dem.', 'Very Mild Dem.', 'Mild. Dem.', 'Mod. Dem.']

    sns.catplot(table, x="class", y="count", hue="target", kind="bar", legend=False)
    plt.savefig(filename, bbox_inches='tight')
    return plt.show()


def make_displot(parameter, bin, filename=None, labels=['Non Dem.', 'Very Mild Dem.', 'Mild. Dem.', 'Mod. Dem.']):
    sns.displot(df, x=parameter, bins=bin, hue="target", kde=False, legend=False)
    plt.legend(labels=labels[::-1])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

    return plt.show()


def make_histplot(parameter, filename=None, labels=['Non Dem.', 'Very Mild Dem.', 'Mild. Dem.', 'Mod. Dem.']):
    #sns.histplot(binwidth=0.5, x=parameter, hue="target", data=df, stat="count", multiple="stack")
    sns.histplot(binwidth=0.5, x=parameter, hue="target", data=df, stat="count", multiple="stack", legend=False)
    plt.legend(labels=labels[::-1])

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    return plt.show()


FILE_NAME = 'data.csv'
df = pd.read_csv(FILE_NAME)

convert_dict = {'M/F': 'category'}

df = df.astype(convert_dict)

# Restrict to patients over 60
df = df[df['Age'] >= 60]

# Target plot
make_bar_plot('target')

# Sex plot
make_histplot('M/F', filename='Figures/gender_dist.png')

# Age plot
make_displot('Age', 10, filename='Figures/age_dist.png')

# Education plot
make_histplot('Educ', filename="Figures/educ_dist.png")

# Socioeconomic plot
make_histplot('SES', filename="Figures/ses.png")

# Mini-mental state evaluation plot
make_displot('MMSE', 10, filename="Figures/mmse_dist.png")

# Estimated total intracranial volume plot
make_displot('eTIV', 10, filename="Figures/eTIV_dist.png")

# Normalized whole brain volume plot
make_displot('nWBV', 10, filename="Figures/nWBV_dist")

#  Atlas scaling factor plot
make_displot('ASF', 10, filename="Figures/asf_dist.png")
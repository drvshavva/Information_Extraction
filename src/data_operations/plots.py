import matplotlib.pyplot as plt
from pandas import DataFrame
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import numpy as np

from os.path import dirname
import warnings

warnings.filterwarnings(action='ignore')
plt.style.use('ggplot')

RESULTS_PATH = dirname(dirname(dirname(__file__))) + "/results/"


def plot_wordcloud(corpus: list, max_words: int = 200, title: str = None) -> WordCloud:
    """
    This method generates wordcloud for given corpus

    :param corpus: list of sentences, :type str
    :param max_words: maximum word count, :type int
    :param title: label for sentences, :type str
    :return: WordCloud
    """
    comment_words = ''
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=100,
                          random_state=42,
                          width=800,
                          height=400,
                          mask=None)

    for sent in corpus:
        comment_words += " " + sent

    wordcloud.generate(str(comment_words))

    plt.figure(figsize=(24.0, 16.0))
    plt.imshow(wordcloud);
    plt.title(title, fontdict={'size': 40, 'color': 'black',
                               'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}word_cloud_{title}")
    return plt


def plot_bar_chart(labels: list, values: list, title: str):
    """
    This method plot bar chart

    :param labels: list of labels, :type list
    :param values: count of each label values, :type list
    :param title: title of plot
    :return: plot
    """
    y_pos = np.arange(len(labels))
    plt.figure(figsize=(24.0, 16.0))
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig(f"{RESULTS_PATH}bar_chart_{title}")
    return plt


def plot_pie_chart(labels: list, values: list, title: str):
    """
    This method plot pie chart

    :param labels: list of labels, :type list
    :param values: count of each label values, :type list
    :param title: title of plot
    :return: plot
    """
    plt.figure(figsize=(24.0, 16.0))
    plt.pie(values, labels=labels, startangle=90, autopct='%.1f%%')
    plt.title(title)
    plt.savefig(f"{RESULTS_PATH}pie_chart_{title}")
    return plt


def plot_count_plot(label_name: str, data: DataFrame, title: str):
    """
    This method returns count plot of the dataset

    :param label_name: name of the class, :type str
    :param data: input dataFrame, :type DataFrame
    :param title: title of plot
    :return plt
    """
    plt.figure(figsize=(24.0, 16.0))
    sns.countplot(x=label_name, data=data)
    plt.title(title)
    plt.savefig(f"{RESULTS_PATH}plot_count_{title}")
    return plt

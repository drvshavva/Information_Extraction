import streamlit as st
import base64
from sqlalchemy import create_engine
from pandas import DataFrame
import pandas as pd
from os.path import dirname
from sqlalchemy import inspect
from nltk.corpus import stopwords as stop
from sqlalchemy import MetaData, Table, Column, String, Integer
import regex
from gensim.models import KeyedVectors
import numpy as np

DATA_PATH = dirname(dirname(__file__)) + "/data/"
MODEL_PATH = dirname(dirname(__file__)) + "/word2vec_model/trmodel"
stop_words = set(stop.words('turkish'))


class Regex:
    VERB = r"(\b[a-z]{1,20}m[ıiuü][sş][a-z]*\b|\b[a-z]{1,20}[ae]c[ae][kgğ][a-z]*\b|\b[a-z]{1,20}m[ae]l[iı][a-z]*\b|\b[a-z]{1,20}yor[a-z]*\b|\b[a-z]{1,20}[dt][iı][nmgk\s][iı]\b)"
    PLACE = r"(\b[a-z]{1,20}[dt][ea][\sn]\b)"  # bulunma/ ayrılma/ çıkma hal eki
    WHEN = r"(\b[a-z]{1,20}[dt][iı][ğg][ıi]nd[ae][a-z]*\b|\b[a-z]{1,20}[dt][iı][ğg][ıi]\szaman*\b|\b[a-z]{1,20}yorken*\b|\b[a-z]{1,20}[ıi]nc[ae]*\b)"
    WHY = r"(\b[a-z]{1,20}[dt][iı][gğ][iı]\sicin*\b|\b[a-z]{1,20}[dt][ea]n\sdolayi*\b|\b[a-z]{1,20}[dt][ea]n\soturu*\b|\b[a-z]{1,20}[dt][ıi][gğ][iı]\sicin*\b|\b[a-z]{1,20}\sdiye*\b|\b[a-z]{1,20}[dt][ıi][ğg][iı]nd[ae]n*\b|\b[a-z]{1,20}\sicin*\b)"


class Default:
    ADJ = ["küçük", "büyük", "mutlu", "kirli", "uzun", "kısa", "temiz", "mutsuz", "yüksek", "alçak"]
    WHO = ["anne", "baba", "kahraman", "kardeş", "yiğit", "o", "bu", "onlar", "şu", "ben", "sen", "biz", "siz", "oğlan",
           "kız", "kadın", "erkek", "bay", "bayan"]
    PLACE = ["ev", "otel", "okul", "restoran", "market", "mekan", "cafe", "iş"]
    WHAT = ["uyur", "düşer", "bayılır", "oturur", "olur", "sıkılır"]


class Tables:
    ADJ = "sıfatlar"
    WHO = "kim"
    WHEN = "ne_zaman"
    WHAT = "ne_olur"
    VERBS = "fiiler"
    PLACE = "yer"
    WHY = "neden"


class Word2vec:
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
        self.model.init_sims(replace=True)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self._model.init_sims(replace=True)

    def get_most_similar(self, word: str):
        return self.model.most_similar([word])

    def words_closer_than(self, word1: str, word2: str):
        return self.model.words_closer_than(word1, word2)

    def get_doesnt_match(self, word_list: list):
        return self.model.doesnt_match(word_list)

    def distance(self, word1: str, word2: str):
        return self.model.distance(word1, word2)

    def closer_than(self, word1: str, word2: str):
        return self.model.closer_than(word1, word2)

    def get_vocab(self):
        return self.model.vocab

    def distances(self, word: str, list_of_words: list):
        return self.model.distances(word, list_of_words)

    def get_vector(self, word: str):
        return self.model.get_vector(word)

    def get_most_similar_to_given(self, word: str, list_of_words: list):
        return self.model.most_similar_to_given(word, list_of_words)

    def cosine_similarity(self, word: str, list_of_words: list):
        wc_list = []
        wc = self.get_vector(word)
        for w in list_of_words:
            _wc = self.get_vector(w)
            wc_list.append(_wc)

        return self.model.cosine_similarities(wc, wc_list)

    def similar_by_word(self, word: str, top_n: int = 10):
        return self.model.similar_by_word(word, top_n)

    def similarity(self, w1: str, w2: str):
        return self.model.similarity(w1, w2)

    def get_document_similarity(self, doc1, doc2):
        d1 = [w for w in doc1.split()]
        d2 = [w for w in doc2.split()]
        return self.model.wmdistance(d1, d2)


def apply_operation(corpus, operation, **kwargs):
    """
    This method applies one operation and returns the result

    Args:
         corpus: list of sentences, :type list
         operation: image operation
         kwargs: (optional) params to apply operations,
                  for stemmer stemmer operation and for remove stopwords stopwords list

    Returns:
         operation applied result
    """
    data_precessed = []
    for sentence in corpus:
        data_precessed.append(operation(sentence, **kwargs))
    return data_precessed


def apply_preprocess_operations_to_corpus(corpus: list, operations: list, **kwargs) -> list:
    """
    This method applies list of operations to given corpus

    Args:
         corpus: list of sentences, :type list
         operations: list of operations, :type list
       operations:
           - remove_less_than_two
           - apply_lemmatizer
           - apply_stemmer
           - remove_stopwords
           - replace_special_chars
           - remove_whitespace
           - remove_punctuation
           - remove_number
           - to_lower
           - remove_hyperlink
         kwargs:(optional) params to apply operations,
                  for stemmer stemmer operation and for remove stopwords stopwords list
    Returns:
         preprocessed sentences, :type list
    """
    for operation in operations:

        if operation == remove_stopwords:
            if "stopwords" in kwargs:
                corpus = apply_operation(corpus, remove_stopwords, kwargs.get("stopwords"))
            else:
                corpus = apply_operation(corpus, remove_stopwords)
        else:
            corpus = apply_operation(corpus, operation)
    return corpus


def verbs(add_select_box):
    with col1:
        try:
            col1.subheader("Kim/Ne ile Yapılır ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[0][add_select_box])))
        except:
            pass

    with col2:
        try:
            col2.subheader("Nerede Yapılır ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[3][add_select_box])))
        except:
            pass

    with col3:
        try:
            col3.subheader("Tanımı Nedir ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[4][add_select_box])))
        except:
            pass

    with col4:
        try:
            col4.subheader("Niçin Yapılır ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[5][add_select_box])))
        except:
            pass

    with col5:
        try:
            col5.subheader("Kim/Ne Yapar ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[6][add_select_box])))
        except:
            pass

    with col6:
        try:
            col6.subheader("Kim/Ne ile Yapılır ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[7][add_select_box])))
        except:
            pass

    with col7:
        try:
            col7.subheader("Neyi/Kimi Yapılır ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[8][add_select_box])))
        except:
            pass

    with col8:
        try:
            col8.subheader("Nasıl Yapılır ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[9][add_select_box])))
        except:
            pass

    with col9:
        try:
            col9.subheader("Ne Olunca Yapılır ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[10][add_select_box])))
        except:
            pass

    with col10:
        try:
            col10.subheader("Neye/Kime Yapılır ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[1][add_select_box])))
        except:
            pass

    with col11:
        try:
            col11.subheader("Fiziksel/Zihinsel ?")
            st.dataframe(st.dataframe(pd.DataFrame.from_dict(isimler_dicts[2][add_select_box])))
        except:
            pass


def nouns(add_select_box):
    with col1:
        try:
            col1.subheader("Hammaddesi Nedir ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[0][add_select_box]))
        except:
            pass

    with col2:
        try:
            col2.subheader("Ağırlık kg ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[5][add_select_box]))
        except:
            pass

    with col3:
        try:
            col3.subheader("Kim Kullanır ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[8][add_select_box]))
        except:
            pass

    with col4:
        try:
            col4.subheader("Ne İşe Yarar ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[9][add_select_box]))
        except:
            pass

    with col5:
        try:
            col5.subheader("Rengi ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[10][add_select_box]))
        except:
            pass

    with col6:
        try:
            col6.subheader("Üst Kavramı Nedir ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[11][add_select_box]))
        except:
            pass

    with col7:
        try:
            col7.subheader("Tanımı Nedir ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[12][add_select_box]))
        except:
            pass

    with col8:
        try:
            col8.subheader("Yanında Neler Olur ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[13][add_select_box]))
        except:
            pass

    with col9:
        try:
            col9.subheader("Hacmi cm3/m3 ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[14][add_select_box]))
        except:
            pass

    with col10:
        try:
            col10.subheader("Nerede Bulunur ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[1][add_select_box]))
        except:
            pass

    with col11:
        try:
            col11.subheader("Şekli Nasıl ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[2][add_select_box]))
        except:
            pass

    with col12:
        try:
            col12.subheader("Canlı/Cansız ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[3][add_select_box]))
        except:
            pass

    with col13:
        try:
            col13.subheader("İçinde Neler Bulunur ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[4][add_select_box]))
        except:
            pass

    with col14:
        try:
            col14.subheader("Ağırlık gr/kg ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[5][add_select_box]))
        except:
            pass

    with col15:
        try:
            col15.subheader("Sıfatları ?")
            st.dataframe(pd.DataFrame.from_dict(isimler_dicts[6][add_select_box]))
        except:
            pass


def remove_stopwords(sentence: str, stopwords: list = stop_words) -> str:
    """
    This method removes stopwords from given sentence

    Args:
         sentence: sentence to remove stopwords, :type str
         stopwords: stopwords list, :type list
    Returns:
         cleaned sentence
    """
    tokens = sentence.split()
    filtered_tokens = [token for token in tokens if token not in stopwords]
    sentence = ' '.join(filtered_tokens)
    return sentence


def to_lower(sentence: str) -> str:
    """
    This method lowers sentence

    Args:
         sentence: input sentence file, :type str
    Returns:
         lower cased sentence
    """
    result = sentence.lower()
    return result


def replace_turkish_chars(sentence: str) -> str:
    """
    This method normalizes turkish characters

    Args:
        sentence: sentence to normalize

    """
    sentence = sentence.replace("ü", "u")
    sentence = sentence.replace("ı", "i")
    sentence = sentence.replace("ö", "o")
    sentence = sentence.replace("ü", "u")
    sentence = sentence.replace("ş", "s")
    sentence = sentence.replace("ç", "c")
    sentence = sentence.replace("ğ", "g")

    return sentence


def apply_list_of_operations_to_data_frame(operations: list, data: DataFrame) -> DataFrame:
    """
    This method takes list of operations to apply preprocess to given data frame

         operations: list of operations
         data: Data frame
    Returns:
         Preprocessed data frame
    """
    # start = time.time()
    for operation in operations:
        data = data.apply(operation)
    # print(f"Processed {len(data)} samples.\n")
    # print(f"It's took {(time.time() - start) / 60} seconds.")
    return data


def get_db_connection(db_name: str):
    """
    This is for to connect db

    Params:
        db_name: Name of the db to connect

    Returns:
        connection
    """
    return create_engine(f'sqlite:///{DATA_PATH}{db_name}').connect()


def read_from_db(table_name: str = "relation_1", db_name: str = "fiiller_relations.db") -> DataFrame:
    con = create_engine(f'sqlite:///{DATA_PATH}{db_name}').connect()
    df = pd.read_sql_table(f"{table_name}", con)
    df = df.drop(["index"], axis=1)
    return df


def get_related_dict(data_frame: DataFrame) -> dict:
    """
    This method provides each item result in table

    Example: akbil - demir, akbil - kart -> {'akbil': [demir, kart]}

    Args:
        data_frame: data frame pandas
    Returns:
        dictionary
    """
    objects = data_frame.values
    objs = set([obj.split("-")[0] for obj in objects])

    objs_dict = dict()
    for obj in list(objs):
        objs_dict[obj] = []

    for obj in objects:
        obj_list = obj.split("-")
        objs_dict[obj_list[0]].append(obj_list[1])

    return objs_dict


def get_relations_df_and_dict(db_name: str):
    connection = get_db_connection(db_name)
    inspector = inspect(connection)
    tables = inspector.get_table_names()

    relations_df_list = []
    for table in tables:
        relations_df_list.append(read_from_db(table_name=table, db_name=db_name))

    # apply preprocess operations
    preprocess_operations = [to_lower, replace_turkish_chars]
    relations_df_list_pr = []
    for df in relations_df_list:
        relations_df_list_pr.append(
            apply_list_of_operations_to_data_frame(operations=preprocess_operations, data=df.iloc[:, 0]))

    relations_dicts = []
    for df in relations_df_list_pr:
        relations_dicts.append(get_related_dict(df))

    return relations_df_list_pr, relations_dicts


WORD2VEC = Word2vec()


def find_regex_results(regex_str: str, data_list: list) -> list:
    """
    This method search regex in given data and returns the results

    Args:
         regex: regular expression, :type str
         data_list: data list to search regex, :type list

    Returns:
         list of result, :rtype list
    """
    matched = list()
    for data in data_list:
        data = data.strip()
        match = regex.findall(regex_str, data, ignore_unused=regex.IGNORECASE)
        if match:
            matched += match
    return matched


def extract_verb(data: str):
    """
    This method extract verbs from given text, which is in db
    to extract regex operation will be applied

    Args:
         data: text to extract verbs
    Returns
         extracted verbs
    """
    verbs = read_from_db("fiiller", "vocabulary.db").iloc[:, 0]  # 487 tekil fiil var

    regex_verbs = []
    for verb in verbs:
        verb = verb.strip()
        regex_verbs.append(r"\b" + verb + r"\b")
    regex_ = '|'.join(regex_verbs)
    sentences = data.split(".")
    cleaned_sentences = apply_preprocess_operations_to_corpus(operations=[replace_turkish_chars, to_lower],
                                                              corpus=sentences)
    text_verbs = find_regex_results(regex_, cleaned_sentences)

    if text_verbs.__len__() < 1:
        __regex = Regex.VERB
        text_verbs = find_regex_results(__regex, cleaned_sentences)

    conn = get_db_connection("vocabulary.db")
    meta = MetaData()

    fiiller = Table(
        'fiiller', meta,
        # Column('index', Integer),
        Column('fiiller', String)
    )
    if isinstance(text_verbs, list) and text_verbs.__len__() >= 1:
        for _verb in text_verbs:
            if _verb not in verbs:
                ins = fiiller.insert().values(fiiller=_verb)
                conn.execute(ins)

    elif text_verbs.__len__() == 1 and text_verbs not in verbs:
        ins = conn.insert().values(fiiller=text_verbs)
        conn.execute(ins)
    return text_verbs


def extract(data: str,
            table_name: str,
            default: list = None,
            special_reg: str = '',
            use_cosine: bool = False):
    db = read_from_db(table_name, "vocabulary_lower.db").iloc[:, 0]

    regex_db = []
    for _d in db:
        _d = _d.strip()
        if _d:
            regex_db.append(r"\b" + _d + r"\b")

    _re = '|'.join(regex_db)
    sentences = data.split(".")
    cleaned_sentences = apply_preprocess_operations_to_corpus(operations=[replace_turkish_chars, to_lower],
                                                              corpus=sentences)
    text_res = find_regex_results(_re, cleaned_sentences)

    try:
        __condition = text_res[0][0].__len__()
        if special_reg != '':
            text_res += find_regex_results(special_reg, cleaned_sentences)
    except:
        __condition = text_res.__len__()
        if text_res.__len__() < 1 and special_reg != '':
            text_res = find_regex_results(special_reg, cleaned_sentences)

    if __condition < 1:
        model = WORD2VEC
        sim = list()
        __cleaned = apply_preprocess_operations_to_corpus(operations=[to_lower, remove_stopwords],
                                                          corpus=sentences)

        for _clean in __cleaned:
            for _s in db:
                if _s.split().__len__() < 2:
                    __split = _clean.split()
                    try:
                        if use_cosine:
                            __res = model.cosine_similarity(_s.strip(), __split)
                            __max = max(__res)
                            if __max >= 0.2:
                                sim.append(__split(np.argmax(__res)))
                        else:
                            sim.append(model.get_most_similar_to_given(_s.strip(), __split))
                    except:
                        continue

        text_res = list(filter(lambda x: len(x) > 3, list(set(sim))))
        if default is not None:
            _f = list()
            __new = list()
            for __res in text_res:
                __new.clear()
                __new = default.copy()
                __new.append(__res)

                _f.append(model.get_doesnt_match(__new))
            text_res = [_n for _n in text_res if _n not in _f]
    return text_res


st.title('Information Extraction from Text')

file_ = open("../io.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

page_bg_img = '''
<img src="../io.gif">
<style>
body {
    color: #fff;
    background-color: #FFFFFF;
}
.stButton>button {
    color: #4F8BF9;
}

.stTextInput>div>div>input {
    color: #4F8BF9;
}
</style>
'''
st.markdown(f'<img src="data:image/gif;base64,{data_url}">', unsafe_allow_html=True)

# Add a selectbox to the sidebar:
add_select_box_side = st.sidebar.selectbox(
    'What do you want to do?',
    ('See Nouns DB', 'See Verbs DB', 'See Vocabulary DB', 'Extract Information')
)

if add_select_box_side == 'See Verbs DB':
    _, isimler_dicts = get_relations_df_and_dict("fiiller_relations.db")
    col1, col2, col3 = st.beta_columns(3)
    col4, col5, col6 = st.beta_columns(3)
    col7, col8, col9 = st.beta_columns(3)
    col10, col11 = st.beta_columns(2)

    add_select_box = st.selectbox(
        'Which Verb?',
        list(isimler_dicts[0].keys())
    )
    if add_select_box == list(isimler_dicts[0].keys())[0]:
        verbs(list(isimler_dicts[0].keys())[0])

    if add_select_box == list(isimler_dicts[0].keys())[1]:
        verbs(list(isimler_dicts[0].keys())[1])

    if add_select_box == list(isimler_dicts[0].keys())[2]:
        verbs(list(isimler_dicts[0].keys())[2])

    if add_select_box == list(isimler_dicts[0].keys())[3]:
        verbs(list(isimler_dicts[0].keys())[3])

    if add_select_box == list(isimler_dicts[0].keys())[4]:
        verbs(list(isimler_dicts[0].keys())[4])

    if add_select_box == list(isimler_dicts[0].keys())[5]:
        verbs(list(isimler_dicts[0].keys())[5])

    if add_select_box == list(isimler_dicts[0].keys())[6]:
        verbs(list(isimler_dicts[0].keys())[6])

    if add_select_box == list(isimler_dicts[0].keys())[7]:
        verbs(list(isimler_dicts[0].keys())[7])

    if add_select_box == list(isimler_dicts[0].keys())[8]:
        verbs(list(isimler_dicts[0].keys())[8])

    if add_select_box == list(isimler_dicts[0].keys())[9]:
        verbs(list(isimler_dicts[0].keys())[9])

    if add_select_box == list(isimler_dicts[0].keys())[10]:
        verbs(list(isimler_dicts[0].keys())[10])

if add_select_box_side == 'See Nouns DB':
    _, isimler_dicts = get_relations_df_and_dict("isimler_ralations.db")
    col1, col2, col3 = st.beta_columns(3)
    col4, col5, col6 = st.beta_columns(3)
    col7, col8, col9 = st.beta_columns(3)
    col10, col11, col12 = st.beta_columns(3)
    col13, col14, col15 = st.beta_columns(3)
    add_select_box = st.selectbox(
        'Which Noun?',
        list(isimler_dicts[0].keys())
    )
    if add_select_box == list(isimler_dicts[0].keys())[0]:
        nouns(list(isimler_dicts[0].keys())[0])

    if add_select_box == list(isimler_dicts[0].keys())[1]:
        nouns(list(isimler_dicts[0].keys())[1])

    if add_select_box == list(isimler_dicts[0].keys())[2]:
        nouns(list(isimler_dicts[0].keys())[2])

    if add_select_box == list(isimler_dicts[0].keys())[3]:
        nouns(list(isimler_dicts[0].keys())[3])

    if add_select_box == list(isimler_dicts[0].keys())[4]:
        nouns(list(isimler_dicts[0].keys())[4])

    if add_select_box == list(isimler_dicts[0].keys())[5]:
        nouns(list(isimler_dicts[0].keys())[5])

    if add_select_box == list(isimler_dicts[0].keys())[6]:
        nouns(list(isimler_dicts[0].keys())[6])

    if add_select_box == list(isimler_dicts[0].keys())[7]:
        nouns(list(isimler_dicts[0].keys())[7])

    if add_select_box == list(isimler_dicts[0].keys())[8]:
        nouns(list(isimler_dicts[0].keys())[8])

    if add_select_box == list(isimler_dicts[0].keys())[9]:
        nouns(list(isimler_dicts[0].keys())[9])

    if add_select_box == list(isimler_dicts[0].keys())[10]:
        nouns(list(isimler_dicts[0].keys())[10])

    if add_select_box == list(isimler_dicts[0].keys())[11]:
        nouns(list(isimler_dicts[0].keys())[11])

    if add_select_box == list(isimler_dicts[0].keys())[12]:
        nouns(list(isimler_dicts[0].keys())[12])

    if add_select_box == list(isimler_dicts[0].keys())[13]:
        nouns(list(isimler_dicts[0].keys())[13])

    if add_select_box == list(isimler_dicts[0].keys())[14]:
        nouns(list(isimler_dicts[0].keys())[14])

if add_select_box_side == 'See Vocabulary DB':
    col1, col2 = st.beta_columns(2)
    col4, col5 = st.beta_columns(2)
    col7, col8 = st.beta_columns(2)
    col10, col11 = st.beta_columns(2)

    with col1:
        col1.subheader("Fiiller")
        st.dataframe(read_from_db("fiiller", "vocabulary_lower.db"))

    with col2:
        col2.subheader("Hammaddesi")
        st.dataframe(read_from_db("hammaddesi", "vocabulary_lower.db"))

    with col4:
        col4.subheader("Kim")
        st.dataframe(read_from_db("kim", "vocabulary_lower.db"))

    with col5:
        col5.subheader("Ne olur ?")
        st.dataframe(read_from_db("ne_olur", "vocabulary_lower.db"))

    with col7:
        col7.subheader("Ne zaman ?")
        st.dataframe(read_from_db("ne_zaman", "vocabulary_lower.db"))

    with col8:
        col8.subheader("Neden ?")
        st.dataframe(read_from_db("neden", "vocabulary_lower.db"))

    with col10:
        col10.subheader("Sıfatlar")
        st.dataframe(read_from_db("sıfatlar", "vocabulary_lower.db"))

    with col11:
        col11.subheader("Yer")
        st.dataframe(read_from_db("yer", "vocabulary_lower.db"))

if add_select_box_side == 'Extract Information':
    text = st.text_area("Text to extract information")
    df_how = pd.DataFrame.from_dict({"Nasıl": extract(text, Tables.ADJ, default=Default.ADJ)})
    df_what = pd.DataFrame.from_dict({"Ne Yaptı?": extract_verb(text)})
    df_where = pd.DataFrame.from_dict(
        {"Nerede?": extract(text, Tables.PLACE, default=Default.PLACE, special_reg=Regex.PLACE)})
    df_when = pd.DataFrame.from_dict({"Ne zaman ?": extract(text, Tables.WHEN, special_reg=Regex.WHEN)})
    df_why = pd.DataFrame.from_dict({"Neden ?": extract(text, Tables.WHY, special_reg=Regex.WHY)})
    df_happen = pd.DataFrame.from_dict({'Ne Yaptı?': extract(text, Tables.WHAT, default=Default.WHAT)})
    df_who = pd.DataFrame.from_dict({'Kim ?': extract(text, Tables.WHO, default=Default.WHO)})

    col1, col2 = st.beta_columns(2)
    col3, col4 = st.beta_columns(2)
    col5, col6, col7 = st.beta_columns(3)

    with col1:
        col1.subheader("Nasıl ?")
        st.dataframe(df_how)

    with col2:
        col2.subheader("Ne Yaptı ?")
        st.dataframe(df_what)

    with col3:
        col3.subheader("Nerede ?")
        st.dataframe(df_where)

    with col4:
        col4.subheader("Ne zaman ?")
        st.dataframe(df_when)

    with col5:
        col5.subheader("Neden ?")
        st.dataframe(df_why)

    with col6:
        col6.subheader('Ne Yaptı?')
        st.dataframe(df_happen)

    with col7:
        col7.subheader('Kim ?')
        st.dataframe(df_who)

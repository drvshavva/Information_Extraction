import pandas as pd
from sqlalchemy import MetaData, Table, Column, String, Integer
import regex
import numpy as np

from src.data_operations.rw_utils import read_from_db, get_db_connection
from src.word2vec.word2vec_operations import Word2vec
from src.preprocess.preprocess import apply_preprocess_operations_to_corpus, replace_turkish_chars, to_lower, \
    remove_stopwords, apply_stemmer
from src.info_extraction.__constants import Regex, Default, Tables

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

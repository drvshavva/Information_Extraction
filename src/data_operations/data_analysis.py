from pandas import DataFrame
import pandas as pd
from sqlalchemy import inspect

from src.data_operations.rw_utils import read_from_db, get_db_connection
from src.preprocess.preprocess import apply_list_of_operations_to_data_frame, replace_turkish_chars, to_lower


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
    """
    This method returns relations from db

    Args:
        db_name: database name
    Returns:
         list of preprocessed data frames
         list of key: [] relations dictionaries
    """
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


def create_db_from_dict(dict_to_save: dict, name: str):
    """
    This method creates db from dictionary

    Args:
        dict_to_save: data to save
        name: table name in db
    """
    df = pd.DataFrame.from_dict(dict_to_save)
    connection = get_db_connection("vocabulary_lower.db")

    df.to_sql(name, con=connection)

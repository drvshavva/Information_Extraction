from os.path import dirname
import warnings

from sqlalchemy import create_engine
from pandas import DataFrame
import pandas as pd
import re

warnings.filterwarnings(action='ignore')
DATA_PATH = dirname(dirname(dirname(__file__))) + "/data/"


def get_db_connection(db_name: str):
    """
    This is for to connect db

    Params:
        db_name: Name of the db to connect

    Returns:
        connection
    """
    return create_engine(f'sqlite:///{DATA_PATH}{db_name}').connect()


def read_from_db(table_name: str = "relation_1", db_name: str = "fiiler_relations.db") -> DataFrame:
    """
    Read data from db and returns dataFrame

    Args:
         table_name: table name, :type str
         db_name: db name, type: str

    Returns:
         DataFrame
    """
    con = create_engine(f'sqlite:///{DATA_PATH}{db_name}').connect()
    df = pd.read_sql_table(f"{table_name}", con)
    df = df.drop(["index"], axis=1)
    print(f"Data is read. Len of the data {len(df)} and columns {df.columns}")
    return df


def create_db_from_pl_file(path: str = "isim_fiil/isimler.pl", db_name: str = "isimler.db"):
    """
    Creates data base from isimler.pl or fiiller.pl text files

    Note: pl files should be includes '' to separate relation between two concept
    Example: one line should be like this to use this function
        iliski('ambalaj','hammaddesi nedir?','ağaç').\n

        with this method we will create a table like this
        ___________________
        |hammaddesi nedir?|
        |_________________|
        |ambalaj-kağıt    |
        ...
        ..

    Args:
         path: path to read text file
         db_name: db name to save table, :type str

    Returns:
        Message of the operation
    """
    try:
        con = create_engine(f'sqlite:///{DATA_PATH}{db_name}').connect()
    except:
        msg = f"Database connection failed: {db_name}"
        return msg

    try:
        with open(rf"{DATA_PATH}{path}") as file:
            lines = file.readlines()
    except:
        msg = f"File: {path} not found in path: {DATA_PATH}. Please check the data in path: {DATA_PATH}{path}."
        return msg

    try:
        # first find the relations between concepts
        relations = set([re.search(r'\'.*\'', line).group(0).replace(r"'", "").split(",")[1] for line in lines])
    except:
        msg = "Relations finding operation is not successful."
        return msg

    try:
        # then create a dict to save as relation : concept1 - concept2
        dict_data = dict()
        for data in list(relations):
            dict_data[data] = []

        for line in lines:
            line_data = re.search(r'\'.*\'', line).group(0).replace(r"'", "").split(",")
            dict_data[line_data[1]].append(f"{line_data[0]} - {line_data[2]}")

        num_relation = 1
        for key in dict_data:
            if len(key) > 3:
                data_frame = pd.DataFrame.from_dict({key: dict_data[key]})
                data_frame.to_sql(f'relation_{num_relation}', con=con)
                num_relation += 1

        msg = f"Data is wrote to path {DATA_PATH}, with name relation_<num>"
    except:
        msg = f"Create data frame from relation - concepts is not successful."
        return msg

    return msg


def read_from_csv(csv_name: str, sep: str = ",") -> DataFrame:
    """
    This method read data from csv file and  returns DataFrame

    Args:
         sep: csv seperator, :type str
         csv_name: name of the csv, :type str
    Returns:
         DataFrame
    """
    df = pd.read_csv(f"{DATA_PATH}{csv_name}", sep=sep)
    print(f"Data is read. Len of the data {len(df)} and columns {df.columns}")
    return df


def write_to_csv(csv_name: str, data: DataFrame):
    """
    This method write data from csv file and  returns DataFrame

    Args:
         data: data to save, :type str
         csv_name: name of the csv, :type str
    Returns:
         None
    """
    data.to_csv(f"{DATA_PATH}{csv_name}", index=False)
    print(f"Data is wrote to path {DATA_PATH}, with name {csv_name}")

import awkward as ak
import csv
import pandas
from typing import Union


def load_ak_from_csv(csv_file):
    '''
    从 csv 中载入为 awkward 数组
    :param csv_file:
    :return:
    '''
    if isinstance(csv_file, str):
        csv_file = open(csv_file, 'r', encoding='utf8')

    reader = csv.reader(csv_file)
    rows = []
    for row in reader:
        rows.append(row)

    arr = ak.Array(rows)
    return arr


def load_ak_from_pandas(table: pandas.DataFrame):
    '''
    从 pandas DataFrame 中载入为 awkward 数组
    :param table:
    :return:
    '''
    shape = table.shape

    rows = []
    rows.append(table.columns.tolist())

    for y in range(shape[0]):
        rows.append(table.iloc[y].tolist())

    arr = ak.Array(rows)
    return arr


def load_ak_from_excel(excel_file, sheet_name: Union[str,int]=0, dtype=str, **pandas_kwargs):
    '''
    从 excel 中载入为 awkward 数组
    :param excel_file: 
    :param sheet_name: 要载入的表号或表名
    :return:
    '''
    table = pandas.read_excel(excel_file, sheet_name=sheet_name, dtype=dtype, **pandas_kwargs)
    arr = load_ak_from_pandas(table)
    return arr


if __name__ == '__main__':
    f = r'a.xlsx'
    d = load_ak_from_excel(f)
    print(d)

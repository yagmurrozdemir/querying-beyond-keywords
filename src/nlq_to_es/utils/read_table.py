#!/usr/bin/env python3
import json
import pandas as pd
import os
import ast
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

types_file= f"{PROJECT_ROOT}/data/resources/types.csv"

master_csv = f"{PROJECT_ROOT}/data/resources/headers.csv"


def load_table_by_id(table_id):
    master_df = pd.read_csv(master_csv)
    types_df = pd.read_csv(types_file)

    table_row = master_df[master_df['Table ID'].str.strip() == table_id.strip()]
    if table_row.empty:
        print(f"⏩ Skipping: Table ID '{table_id}' not found in {master_csv}")
        return None, None

    type_row = types_df[types_df['Table ID'].str.strip() == table_id.strip()]
    if type_row.empty:
        print(f"⏩ Skipping: Table ID '{table_id}' not found in {types_file}")
        return None, None

    column_data = table_row.iloc[0]['Headers']
    type_data = type_row.iloc[0]['Types']
    
    return [col.strip() for col in column_data.split(';')], [type.strip() for type in type_data.split(';')]








                


#!/usr/bin/env python3
import json
import pandas as pd
from elasticsearch import Elasticsearch
import os
import ast
import sys
from pathlib import Path

# Elasticsearch connection
ES_HOST = os.getenv("ELASTIC_HOST", "http://localhost:9200")
es = Elasticsearch(ES_HOST)

# Relative CSV paths inside Docker
TYPES_FILE = "inputs/types.csv"
MASTER_CSV = "inputs/headers.csv"

class Query:
    agg_ops = ['', 'max', 'min', 'value_count', 'sum', 'avg']
    cond_ops = ['=', '>', '<', 'OP']
    agg_ops_sql = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

    def __init__(self, sel_column, agg_index, conditions=tuple()):
        self.sel_column = sel_column
        self.agg_index = agg_index
        self.conditions = list(conditions)

    def __repr__(self):
        rep = 'SELECT {agg} {sel} FROM table'.format(
            agg=self.agg_ops_sql[self.agg_index],
            sel=self.sel_column,
        )
        if self.conditions:
            rep += ' WHERE ' + ' AND '.join([
                '{} {} {}'.format(col, self.cond_ops[op], val)
                for col, op, val in self.conditions
            ])
        return rep

def load_table_by_id(master_csv, types_file, table_id):
    master_df = pd.read_csv(master_csv)
    types_df = pd.read_csv(types_file)

    table_row = master_df[master_df['Table ID'].str.strip() == table_id.strip()]
    if table_row.empty:
        print(f"⚠️ Table ID '{table_id}' not found in {master_csv}")
        return None, None

    type_row = types_df[types_df['Table ID'].str.strip() == table_id.strip()]
    if type_row.empty:
        print(f"⚠️ Table ID '{table_id}' not found in {types_file}")
        return None, None

    columns = table_row.iloc[0]['Headers'].split(';')
    types = type_row.iloc[0]['Types'].split(';')
    return [c.strip() for c in columns], [t.strip() for t in types]

def convert_to_elasticsearch_dsl(encoded_query, master_csv):
    table_id = encoded_query['table_id']
    question = encoded_query['question']
    sql = encoded_query['sql']
    agg_info = False
    index_name = f"table{table_id.replace('-', '_')[1:]}"
    columns, types = load_table_by_id(master_csv, TYPES_FILE, table_id)

    if not columns or not types:
        return None

    sel_col = columns[sql['sel']]
    sel_col_type = types[sql['sel']]
    agg_op = Query.agg_ops[sql['agg']]
    conditions = []

    for cond in sql['conds']:
        col_index, op_index, value = cond
        col_name = columns[col_index]
        operator = Query.cond_ops[op_index]
        col_type = types[col_index]

        if col_name.endswith("."):
            col_name = col_name[:-1]

        if col_type == 'text':
            conditions.append({
                "term": {f"{col_name}.keyword": {"value": value, "case_insensitive": True}}
            })
        elif col_type == 'dense_vector':
            conditions.append({
                "knn": {
                    "field": col_name,
                    "query_vector": value,
                    "k": 20,
                    "similarity": 0.98
                }
            })
        else:
            if isinstance(value, str):
                value = value.replace(',', '.')
            if operator == '=':
                conditions.append({"term": {col_name: {"value": value}}})
            elif operator == '>':
                conditions.append({"range": {col_name: {"gt": value}}})
            elif operator == '<':
                conditions.append({"range": {col_name: {"lt": value}}})

    query = {
        "query": {
            "bool": {
                "must": conditions
            }
        }
    }

    if sel_col.endswith("."):
        sel_col = sel_col[:-1]

    if agg_op:
        field = f"{sel_col}.keyword" if sel_col_type == "text" else sel_col
        query["aggs"] = {
            f"{agg_op}_{sel_col}": {
                agg_op: {"field": field}
            }
        }
        query["_source"] = False
        agg_info = True
    else:
        query["_source"] = [sel_col]

    return agg_info, index_name, query, question

def get_response(result):
    agg_info, index_name, dsl_query, question = result
    return es.search(index=index_name, body=dsl_query)

def main(encoded_query_str, output_file_path):
    encoded_query = json.loads(encoded_query_str)
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

    result = convert_to_elasticsearch_dsl(encoded_query, MASTER_CSV)
    if result is None:
        Path(output_file_path).write_text("Table ID not found.")
        return

    response = get_response(result)
    Path(output_file_path).write_text(json.dumps(response, indent=2))

# CLI mode for subprocess.run(...)
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python correct_result.py <query_json_string> <output_file_path>")
        sys.exit(1)

    json_str = sys.argv[1]
    output_file = sys.argv[2]
    main(json_str, output_file)

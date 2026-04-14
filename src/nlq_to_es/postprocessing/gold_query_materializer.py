#!/usr/bin/env python3
import json
import pandas as pd
import os
import ast
import sys
from pathlib import Path
from nlq_to_es.utils.read_table import load_table_by_id


class Query:
    agg_ops = ['', 'max', 'min', 'value_count', 'sum', 'avg']
    cond_ops = ['=', '>', '<', 'OP']
    agg_ops_sql = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']

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

    def to_dict(self):
        return {'sel': self.sel_column, 'agg': self.agg_index, 'conds': self.conditions}


def to_valid_json_string(query_text) -> str:
    if isinstance(query_text, (dict, list)):
        return json.dumps(query_text)

    cleaned = str(query_text).strip()

    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    parsed = ast.literal_eval(cleaned)
    return json.dumps(parsed)


def convert_to_elasticsearch_dsl_text(data):
    
    table_id = data['table_id']
    print("hii!", table_id)
    sql = data['sql']
    index_name = f"table{table_id.replace('-', '_')[1:]}"

    columns, types = load_table_by_id(table_id)
    if columns is None or types is None:
        return None


    sel_col = columns[sql['sel']]
    sel_col_type = types[sql['sel']]
    agg_op = Query.agg_ops[sql['agg']]

    conditions = []
    
    for cond in sql['conds']:
        col_index, op_index, value = cond
        if 0 <= col_index < len(columns):
            col_name = columns[col_index]
            operator = Query.cond_ops[op_index]
            col_type = types[col_index]

            if col_name.endswith("."):
                col_name = col_name[:-1]

            if col_type == 'text':
                conditions.append({"term": {f"{col_name}.keyword": {"value": value, "case_insensitive": True}}})
            elif col_type == 'dense_vector':
                conditions.append({"knn": {"field": col_name, "query_vector": value, "k": 20, "similarity": 0.98}})  # Increase k to 10 to get more results
            
                
            else:
                if type(value) not in [int, float]:
                    value = value.replace(',', '.')
                if operator == '=':
                    conditions.append({"term": {f"{col_name}": {"value": value}}})
                elif operator == '>':
                    conditions.append({"range": {col_name: {"gt": value}}})
                elif operator == '<':
                    conditions.append({"range": {col_name: {"lt": value}}})

    query_dsl = {
        "query": {
            "bool": {
                "must": conditions
            }
        }
    }



    # Handle aggregation if needed
    if sel_col.endswith("."):
        sel_col = sel_col[:-1]

    if agg_op and agg_op != '':

        field_name = f"{sel_col}.keyword" if sel_col_type == 'text' else f"{sel_col}"
        if "aggs" not in query_dsl:
            query_dsl["aggs"] = {}
        query_dsl["aggs"][f"{agg_op}_{sel_col}"] = {
            f"{agg_op}": {
                "field": field_name
            }
        }

        query_dsl["_source"] = False

    else:
        query_dsl["_source"] = [sel_col]

    return str(query_dsl)

def convert_to_elasticsearch_dsl(data):
    query_text = convert_to_elasticsearch_dsl_text(data)
    query_dsl = to_valid_json_string(query_text)
    return query_dsl




                


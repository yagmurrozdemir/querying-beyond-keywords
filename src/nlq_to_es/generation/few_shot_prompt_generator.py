#!/usr/bin/env python3
import openai
import re
import os
import sys
from elasticsearch import Elasticsearch, exceptions
import json


def generate_prompt(index_mapping: str, nl_query: str, template:str) -> str:

    example_nlq = "How many times did an incumbent, defined such as:\n\"Rubén Eloy Hinojosa ... get elected?\""

    example_output_query = """
    {
      "query": {
        "bool": {
          "must": [
            {
              "knn": {
                "field": "Incumbent",
                "query_vector": $vector$,
                "k": 20,
                "similarity": 0.98
              }
            }
          ]
        }
      }
    }
    """

    example_vector_content  = "Rubén Eloy Hinojosa ..."

    example_nlq_2 = "What is the original air date ..."

    example_output_query_2 = """
    {
      "query": {
        "term": {
          "Written by.keyword": {
            "value": "Karen Felix and Don Woodard",
            "case_insensitive": true
          }
        }
      }
    }
    """


    prompt_txt = template.format(
        example_nlq=example_nlq,
        example_output_query=example_output_query,
        example_nlq_2=example_nlq_2,
        example_output_query_2=example_output_query_2,
        example_vector_content=example_vector_content,
        index_mapping=index_mapping,
        nl_query=nl_query
    )

    return prompt_txt



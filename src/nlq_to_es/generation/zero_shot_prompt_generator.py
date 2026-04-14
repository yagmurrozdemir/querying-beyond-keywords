#!/usr/bin/env python3
import openai
import re
import os
import sys
from elasticsearch import Elasticsearch, exceptions
import json


def generate_prompt(index_mapping: str, nl_query: str, template: str) -> str:

    prompt_txt = template.format(index_mapping=index_mapping, nl_query=nl_query)
    return prompt_txt


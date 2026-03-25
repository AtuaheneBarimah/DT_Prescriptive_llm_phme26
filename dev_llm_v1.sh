#!/bin/bash

clear

litgpt download list

litgpt download microsoft/Phi-3-mini-128k-instruct

sleep 10

python llm_report.py

sleep 10

python llm_report_test.py


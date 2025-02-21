#!/usr/bin/env bash

uv run src/glycocalyx/data_preparation.py
uv run src/glycocalyx/fit_cmdstanpy.py
uv run src/glycocalyx/investigate.py

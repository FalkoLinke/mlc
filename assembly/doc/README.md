# Project Report - README

This directory contains the sources for the project report.

## Build

In order to build the project report, use the following command to set up a python virtual environment and install all required packages.

````
uv sync
````

Before building, make sure the virtual environment is activated.

````
source .venv/bin/activate
````

Afterwards the project report may be built using the following command:

````
make html
````

The output is then located in `_build/html`.





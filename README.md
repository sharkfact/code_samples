Sample Code
===

This repo contains a small collection of sample code written by Maddie Laethem. Any code related to my project work during my time at MassMutual related to my job there cannot be shared, so I hope that these non-proprietary samples will be sufficient. If there are additional examples that you would like to see, I do have more available, including dashboards, regression and classification with kernels, a Python text decompressor, etc... Just let me know! I have included files in R, Python, and SQL that demonstrate a small portion of my coding capabilities in these languages. The structure of the repo is as follows:

* `modular_shiny_code/:` Contains code for a toy dashboard that uses modularized code. These files were created to demonstrate the basics of modularizing code in Shiny dashboards for co-workers. Although this code related to my work somewhat, I have included it because it is not proprietary intellectual property. 
  * `app.R` 
  * `config.R`
  * `modules.R`
  * `wwww/`
    * `background.png`
    * `custom.css`
    * `massmutual_logo.png`
* `python_image_carver/:` A simple image carver written in Python to carve contiguous image files from the `dfrws-2006-challenge.raw` disk image used in the 2006 DFRWS forensic challenge.
  * `carve_JPEG.py`
  * `dfrws-2006-challenge.raw`
* `python_machine_learning/:` Python code for regression and classification tasks on various included datasets. Methods were validated with five-fold cross validation. The .npz data files are used for classification and the .txt data files for regression.
  * `classification.py`
  * `data_test.npz`
  * `data_test.txt`
  * `data_train.npz`
  * `data_train.txt`
  * `labels_train.npz`
  * `labels_train.txt`
  * `regression.py`
* `shiny_hello_world/:` A very basic single file Shiny dashboard of non-modularized code that runs simple data cleaning operations before building a dashboard that compares juvenile arrest records by race. 
  * `JAR`
  * `JAR_shiny.R`
* `sql/:` A collection of small queries on sample database schemas to highlight the fundamentals of writing SQL code.
  * `sql_queries.sql`
  
  

#!/bin/bash
echo "Download data from kaggle"
kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction
unzip -o real-or-fake-fake-jobposting-prediction.zip
echo "Save column titles"
head -n 1 fake_job_postings.csv > column_titles.csv
tail -n +2 fake_job_postings.csv > data_not_shuf.csv
echo "Create sets"
shuf data_not_shuf.csv > data_not_cutted.csv
head -n $1 data_not_cutted.csv > data.csv
sed -n '1,2500p'  data.csv > data_test.csv
sed -n '2501,5000p'  data.csv > data_dev.csv
tail -n +5001  data.csv > data_train.csv
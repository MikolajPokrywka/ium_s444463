# Run with docker
`docker build  -t ium .`

`docker run -e KAGGLE_USERNAME='your_kaggle_username' -e KAGGLE_KEY='<your_kaggle_key>' -e CUTOFF='1600' -it ium:latest`

Read models results:

`docker run -e KAGGLE_USERNAME='your_kaggle_username' -e KAGGLE_KEY='<your_kaggle_key>' -e CUTOFF='1600' -it ium:latest /bin/bash`

`python3 deepl.py`

`cat model_resutls.txt`
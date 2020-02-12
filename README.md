# SentimentAnalysis
Sentiment Analysis based on word-embeddings pre-trained on a twitter messages. It is RNN model using Attention layer.

1. You need to Download [word-embeddings-300d](https://mega.nz/#!u4hFAJpK!UeZ5ERYod-SwrekW-qsPSsl-GYwLFQkh06lPTR7K93I) and put it in embeddings/
2. Create weights/ inside sentiment_analysis/. Download [Weights](https://drive.google.com/file/d/1OPDocwIghXQq7G3BuZVnFS7H2YUT8mnD/view?usp=sharing) and place them in weights/

# Usage

## Docker
  
1) docker build -t [CONTAINER_NAME] [PATH_TO_DOCKERFILE](if it is in Your current directory use ".")  
2) docker run [CONTAINER_NAME]  
3) By default Your docker should run on 172.17.0.2:5001. But If it is not working You have to check  
it by using some additinal commands  
4) docker ps -> check Your's container name (they are random generated, last column) **
5) docker inspect (name from previous command)  **
6) find "IPadress". It should be something like 172.17.0.* **
7) Now You just need to paste this IP with proper port (5001 by default) in Your browser **


## Without Docker(python3, virtualenv recommended) 
1) git https://github.com/Piotrekszmel/SentimentAnalysis.git
2) cd SentimentAnalysis
1) pip3 install -r requirements
2) python3 app.py
3) by default app should start on http:0.0.0.0:5005, so just go to Your browser and paste this URL

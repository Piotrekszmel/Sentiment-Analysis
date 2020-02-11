from flask import Flask, render_template, request, url_for, redirect
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from sentiment_analysis.evaluate import predict_sentiment_single_tweet 
from sentiment_analysis.models.utils import create_model
from sentiment_analysis.utilities.data_loader import Loader


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
  return render_template("index.html")


@app.route("/Sentiment", methods=["GET", "POST"])
def sentiment():
    if request.method == "POST":
        f = request.form["data_file"]
        texts = []
        predictions = []
        message = request.form["text"]
        if message is not '':
            texts.append(message)
        if f is not '':
            with open(f) as file:
                for line in file:
                    line = line.replace("\r", "").replace("\n", "")
                    texts.append(line)
        
        with graph.as_default():
            set_session(sess)
            if message is not '':
                prediction = predict_sentiment_single_tweet(message, sentiment_model, loader.pipeline)
                predictions.append(prediction)
            if texts is not []:
                for text in texts[1:]:
                    prediction = predict_sentiment_single_tweet(text, sentiment_model, loader.pipeline)
                    predictions.append(prediction)
                print(texts, "\n")
                print(predictions)
                return render_template('sentiment.html', predictions=predictions, texts=texts)
            else:
                render_template("index.html")
    return render_template("index.html")


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    set_session(sess)
    global graph, sentiment_model
    sentiment_model, word_indices = create_model("datastories.twitter", 300, "sentiment_analysis/weights/bi_model_weights_1.h5")
    graph = tf.get_default_graph()
    
    loader = Loader(word_indices, text_lengths=50)
    
    app.run(debug=False, port=5005, host='0.0.0.0')
import os
from flask import Flask, render_template, request
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

app = Flask(__name__)

# Função para baixar dados NLTK necessários
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Usando a função de baixar dados
download_nltk_data()

# Função para pré-processar o texto
def preprocess_text(text):
    # Tokenizar o texto
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    text = ' '.join(tokens)
    text = text.lower()
    return text

# Diretório base (Usando o diretório atual)
base_directory = '.'


# Caminhos para o modelo treinado e vetorizador
model_path = os.path.join(base_directory, os.environ.get('MODEL_PATH', 'best_model.pkl'))
vectorizer_path = os.path.join(base_directory, os.environ.get('VECTORIZER_PATH', 'best_vectorizer.pkl'))

# Carregar o vetorizador (ele deve ter sido ajustado nos dados de treinamento)
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Carregar o modelo
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Função para pré-processar e prever o artigo de notícia
def predict_news(news_article):
    try:
        preprocessed_article = preprocess_text(news_article)
        X_news = vectorizer.transform([preprocessed_article])
        prediction = model.predict(X_news)[0]
        return "FAKE" if prediction == 0 else "REAL"
    except Exception as e:
        # Registrar o erro para fins de depuração
        app.logger.error(f"Erro na previsão: {e}")
        # Retornar uma mensagem de erro ou None para indicar um problema na previsão
        return None

# Rota para a página inicial
@app.route("/")
def home():
    return render_template("index.html")

# Rota para previsão
@app.route('/predict', methods=['POST'])
def predict():
    news_article = request.form['news_article']
    prediction = predict_news(news_article)
    return render_template('result.html', prediction=prediction)

# Iniciar o aplicativo Flask
if __name__ == '__main__':
    app.run(debug=True)
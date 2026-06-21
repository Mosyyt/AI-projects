from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import string

app = Flask(__name__)

# Load the model and dictionary
model = load_model('trained_model.keras')
model.make_predict_function()

# Generating a dictionary with letters of the alphabet as values
alphabet = string.ascii_uppercase
dic = {i: letter for i, letter in enumerate(alphabet)}

def predict_label(img_path):
    try:
        img = image.load_img(img_path, target_size=(64, 64))  # Resize to match model's expected input shape
        img = image.img_to_array(img) / 255.0
        img = img.reshape(1, 64, 64, 3)  # Reshape to match model's expected input shape
        prediction = model.predict(img)
        predicted_label_index = prediction.argmax()
        predicted_label = dic.get(predicted_label_index, 'Unknown')
        return predicted_label
    except Exception as e:
        print(f"Error processing image: {e}")
        return f'Error: {e}'

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        prediction = predict_label(img_path)
        return render_template("index.html", prediction=prediction, img_path=img_path)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

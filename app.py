from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('our_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        
        # Save the image to the static folder
        file.save('static/temp_image.jpg')

        # Load and preprocess the image
        img = image.load_img('static/temp_image.jpg', target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_data = preprocess_input(img_array)

        # Make predictions
        prediction = model.predict(img_data)

        # Determine the result based on the prediction
        if prediction[0][0] > prediction[0][1]:
            result = 'Person is safe.'
        else:
            pneumonia_percentage = prediction[0][1] * 100
            result = f'Person is affected with Pneumonia. Percentage of Pneumonia: {pneumonia_percentage:.2f}%'

        return render_template('result.html', result=result)

@app.route('/home')
def return_home():
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

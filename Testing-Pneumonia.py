from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = load_model('our_model.h5')  # Loading our model
img = image.load_img('D:/ML-Project/Pneumonia/chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg', target_size=(224, 224))
image_array = image.img_to_array(img)  # Converting the X-Ray into pixels
image_array = np.expand_dims(image_array, axis=0)
img_data = preprocess_input(image_array)
prediction = model.predict(img_data)

if prediction[0][0] > prediction[0][1]:
    print('Person is safe.')
else:
    print('Person is affected with Pneumonia.')
    pneumonia_percentage = prediction[0][1] * 100
    print(f'Percentage of Pneumonia: {pneumonia_percentage  :.2f}%')

print(f'Predictions: {prediction}')
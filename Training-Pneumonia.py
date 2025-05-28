from keras.models import Model 
from keras.layers import Flatten, Dense 
from keras.applications.vgg16 import VGG16
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.layers import Dropout

# Model Training
IMAGESHAPE = [224, 224, 3]
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)

for each_layer in vgg_model.layers[:-4]:  # Unfreeze last 4 layers
    each_layer.trainable = False 

classes = glob('chest_xray/train/*')
flatten_layer = Flatten()(vgg_model.output)
dropout_layer = Dropout(0.5)(flatten_layer)  # Adding dropout
prediction = Dense(len(classes), activation='softmax')(dropout_layer)
final_model = Model(inputs=vgg_model.input, outputs=prediction)
final_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
testing_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'chest_xray/train',
    target_size=(224, 224),
    batch_size=32,  # Increase batch size
    class_mode='categorical'
)

test_set = testing_datagen.flow_from_directory(
    'chest_xray/test',
    target_size=(224, 224),
    batch_size=32,  # Increase batch size
    class_mode='categorical'
)

fitted_model = final_model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=10,  # Increase epochs
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Displaying accuracy
accuracy = final_model.evaluate_generator(test_set, steps=len(test_set))
print("Accuracy: {:.2f}%".format(accuracy[1] * 100))

# Plotting the training history
plt.plot(fitted_model.history['loss'], label='Training Loss')
plt.plot(fitted_model.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(fitted_model.history['accuracy'], label='Training Accuracy')
plt.plot(fitted_model.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Saving the model
final_model.save('our_model.h5')
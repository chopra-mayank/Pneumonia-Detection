from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Load the saved model
model = load_model('our_model.h5')

# Define test data generator
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = test_datagen.flow_from_directory(
    'chest_xray/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate_generator(test_set, steps=len(test_set))

print("Test Loss: {:.4f}".format(loss))
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
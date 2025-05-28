# Pneumonia Detection Using Chest X-Ray Images

This project uses deep learning to detect pneumonia from chest X-ray images. It includes a trained Convolutional Neural Network (CNN), a Flask web app for live predictions, and visualizations of model performance.

## 🚀 Features

- CNN model trained on chest X-ray images
- Flask-based web interface for user interaction
- Visualizations of accuracy and loss during training
- Modular Python scripts for training, testing, and scoring

## 📁 Project Structure
├── app.py # Flask web application
├── Score.py # Model evaluation script
├── Training-Pneumonia.py # CNN training script
├── Testing-Pneumonia.py # Model testing script
├── templates/ # HTML templates for Flask
├── static/ # CSS, JS, images, etc.
├── Training and Validation Accuracy.png
├── Training and Validation Loss.png

## 🧠 Dataset

Dataset used: **Chest X-Ray Images (Pneumonia)** from Kaggle  
📎 [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## 💻 How to Run the Project

1. **Clone the repository**

   ```bash
   git clone https://github.com/chopra-mayank/Pneumonia-Detection.git
   cd Pneumonia-Detection

2. **Install dependencies**

  ```bash
  pip install -r requirements.txt
  ```

3. **Run the web application**
```bash
python app.py
```
4. **Open in browser**

Visit http://127.0.0.1:5000/ in your web browser.

📊 Model Performance
Training and validation performance visualizations:



📌 Notes
Model file (our_model.h5) is not included. You can train your own or download from an external source.

Dataset folder chest_xray/ must be manually downloaded if you wish to retrain the model.

Flask handles the front-end interface with support for local predictions.

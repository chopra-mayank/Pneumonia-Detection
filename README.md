Here is your **updated `README.md` file**, preserving your style and structure while adding the enhanced **Model Performance Summary** and updated **Features** as requested — without including code or deep explanations:

---

```markdown
# 🫁 Pneumonia Detection Using Chest X-Ray Images

This project leverages deep learning to detect pneumonia from chest X-ray images. It features a Convolutional Neural Network (CNN), a Flask web app for real-time predictions, and visualizations to track model performance.

---

## 🚀 Features

* ✅ VGG16-based CNN with transfer learning and dropout regularization
* 🔄 Last 4 convolutional layers unfrozen for fine-tuning
* 📊 Real-time visualization of accuracy and loss
* 🌐 Flask-based UI for image uploads and predictions
* 🧩 Modular structure for training, testing, evaluation, and deployment

---

## 📁 Project Structure

```

├── app.py                           # Flask web application
├── Score.py                         # Model evaluation script
├── Training-Pneumonia.py            # CNN training script
├── Testing-Pneumonia.py             # Model testing script
├── templates/                       # HTML templates for Flask
├── static/                          # CSS, JS, and images
├── Training and Validation Accuracy.png
├── Training and Validation Loss.png
├── requirements.txt                 # Python dependencies

````

---

## 🧠 Dataset

Dataset used: **Chest X-Ray Images (Pneumonia)** from Kaggle  
🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

> ⚠️ Note: The dataset (`chest_xray/`) must be downloaded manually from the link above.

---

## 💻 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/chopra-mayank/Pneumonia-Detection.git
cd Pneumonia-Detection
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the web application

```bash
python app.py
```

### 4. Open in your browser

Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the application.

---

## 📊 Model Performance

### 🔍 Accuracy

![Accuracy](Training%20and%20Validation%20Accuracy.png)

* Training accuracy peaked at **\~96.7%**
* Validation accuracy stabilized around **\~93%**
* Consistent trend shows good generalization capability

### 📉 Loss

![Loss](Training%20and%20Validation%20Loss.png)

* Training loss decreased steadily to **\~0.1**
* Validation loss fluctuated but ended near **0.3**
* Minimal overfitting observed

---

## 📌 Notes

* The trained model file (`our_model.h5`) is **not included** in the repository.
* You can either train the model yourself using the dataset or download a pre-trained version.
* Flask is used to serve the model and handle the front-end interface for local predictions.

---

```

Let me know if you'd like this exported as a `.md` file or committed via Git steps.
```

# Diabetic Retinopathy Detector using Ensemble Learning

This project is a **machine learning-based web application** designed to detect **diabetic retinopathy** by analyzing retinal images. The application uses **Ensemble Learning**, combining models like **ResNet** and **VGG** to enhance the accuracy of detection. The web interface allows users to upload retinal images and view diagnostic results.

## Features

- **Image Upload**: Users can upload retinal images for analysis.
- **Ensemble Learning Model**: Combines multiple pre-trained models (ResNet, VGG) to detect diabetic retinopathy.
- **Web-Based Interface**: User-friendly web interface built with Flask.
- **Results Display**: Shows whether diabetic retinopathy is present in the uploaded image.
  
## Technologies Used

- **Python**: Core programming language for machine learning and web development.
- **Flask**: Web framework for building the application interface.
- **TensorFlow / Keras**: Libraries for model building and training.
- **OpenCV**: Library for image preprocessing.
- **Pre-trained Models**: Includes **ResNet101**, **ResNet152**, **VGG16**, and **VGG19** models for image classification.

## File Structure

```
Diabetic-Retinopathy-Detector-using-Ensemble-Learning/
├── static/                             # Static files (CSS, images, etc.)
├── templates/                          # HTML templates for Flask
├── app.py                              # Flask app entry point
├── main - ResNet101.ipynb               # Jupyter Notebook for ResNet101 model
├── main - ResNet152.ipynb               # Jupyter Notebook for ResNet152 model
├── main - vgg16.ipynb                   # Jupyter Notebook for VGG16 model
├── main - vgg19.ipynb                   # Jupyter Notebook for VGG19 model
├── main-ensemble.ipynb                  # Jupyter Notebook for ensemble model
├── retina_weights.hdf5                  # Pre-trained model weights for the retina model
├── requirement.txt                      # Python dependencies
└── tempCodeRunnerFile.py                # Temporary file for code execution
```

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/RadhaPriyadarshan/Diabetic-Retinopathy-Detector-using-Ensemble-Learning.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd Diabetic-Retinopathy-Detector-using-Ensemble-Learning
    ```

3. **Create a virtual environment** (optional):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4. **Install the dependencies**:
    ```bash
    pip install -r requirement.txt
    ```

5. **Run the Flask app**:
    ```bash
    python app.py
    ```

6. **Access the application**:  
   Go to `http://127.0.0.1:5000/` in your web browser.

## How It Works

1. **Image Upload**: Users upload retinal images via the web interface.
2. **Preprocessing**: The images are preprocessed using **OpenCV**.
3. **Model Predictions**: The pre-trained models (ResNet101, ResNet152, VGG16, VGG19) are applied to classify the images.
4. **Ensemble Learning**: The individual model predictions are combined using an ensemble approach to provide a final diagnostic result.
5. **Result Display**: The result is shown on the web interface, indicating whether diabetic retinopathy is detected.

## Pre-trained Models

- **ResNet101**, **ResNet152**: These are deep residual networks known for their accuracy in image classification tasks.
- **VGG16**, **VGG19**: Popular convolutional neural networks that excel at feature extraction from images.
- **Ensemble Model**: The predictions from all models are combined to form a robust final prediction.

## Dataset

To train the models, you can use datasets such as:
- [Kaggle's Diabetic Retinopathy Detection Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection)

## Contributing

Contributions are welcome! If you'd like to improve this project:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


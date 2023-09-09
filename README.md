<!DOCTYPE html>
<html>

<head>
   
</head>

<body>

<h1>Smart Waste Segregation System - Neural Network Model</h1>

<h2>Overview</h2>

<p>This repository contains a Convolutional Neural Network (CNN) model for a Smart Waste Segregation System. The model is designed to classify waste items into three categories: organic, recyclable, and trash. This system can be integrated into smart waste bins to automate the waste segregation process, making waste management more efficient and environmentally friendly.</p>

<p>The neural network model uses computer vision techniques to analyze images of waste items and assign them to the appropriate category. It has been trained on a dataset of labeled waste images and achieved high accuracy in classifying waste items.</p>

<h2>Key Features</h2>

<ul>
    <li>Waste classification into three categories: organic, recyclable, and trash.</li>
    <li>Uses a Convolutional Neural Network (CNN) for image classification.</li>
    <li>High accuracy and efficiency in waste segregation.</li>
    <li>Easy integration with smart waste bins and cameras.</li>
    <li>Minimal false positives and false negatives.</li>
</ul>

<h2>Getting Started</h2>

<h3>Code Requirements</h3>

<p>The following Python libraries and packages are required to run the Smart Waste Segregation System:</p>

<ul>
    <li><strong>TensorFlow:</strong> You can install TensorFlow using pip:</li>
    <pre>pip install tensorflow</pre>

  <li><strong>NumPy:</strong> NumPy is used for numerical operations. Install it using pip:</li>
  <pre>pip install numpy</pre>

  <li><strong>Matplotlib:</strong> Matplotlib is used for data visualization. You can install it using pip:</li>
    <pre>pip install matplotlib</pre>

  <li><strong>Random:</strong> The random module is included in Python's standard library.</li>

  <li><strong>Math:</strong> The math module is included in Python's standard library.</li>

  <li><strong>OpenCV (cv2):</strong> OpenCV is used for image processing. Install it using pip:</li>
    <pre>pip install opencv-python</pre>

  <li><strong>Scikit-Learn:</strong> Scikit-Learn is used for machine learning tasks. Install it using pip:</li>
    <pre>pip install scikit-learn</pre>
</ul>

<h3>Usage</h3>

<ol>
    <li>Clone this repository to your local machine:</li>

<pre>
git clone https://github.com/devinchugh/smart_waste_segregation.git
</pre>

  <li>Navigate to the project directory:</li>

<pre>
cd smart_waste_segregation
</pre>

  <li>Place the waste images you want to classify in the <code>input_images</code> directory.</li>

  <li>Run the model to classify the waste items:</li>

<pre>
python classify_waste.py
</pre>

  <p>The model will process the images in the <code>input_images</code> directory and save the results in the <code>output_images</code> directory.</p>
</ol>

<h3>Model Training</h3>

<p>If you want to retrain the model or fine-tune it with your own dataset, you can do so by following these steps:</p>

<ol>
    <li>Prepare your dataset and organize it into three subdirectories: <code>organic</code>, <code>recyclable</code>, and <code>trash</code>.</li>

  <li>Update the paths to your dataset in the <code>WasteClassification.ipynb</code> script.</li>

  <li>Run the Notebook:</li>


  <li>The trained model will be saved in the <code>model.h5</code> and can be used for waste classification.</li>
</ol>

<h2>Contributing</h2>

<p>Contributions to this project are welcome. If you have any suggestions, improvements, or bug fixes, please submit a pull request or open an issue.</p>

<h3>Dataset Sources</h3>

<p>The dataset used for training and testing the Smart Waste Segregation System model was collected from various sources:</p>

<ol>
    <li>
        <a href="https://github.com/garythung/trashnet">TrashNet</a>: A dataset provided by Gary Thung and Mindy Yang, which contains labeled images of trash items in categories such as cardboard, glass, metal, paper, plastic, and trash.
    </li>
    <li>
        <a href="https://github.com/AgaMiko/waste-datasets-review">Waste Datasets Review</a>: This dataset contains a compilation of waste-related datasets from different sources, providing a diverse set of waste images for training and evaluation.
    </li>
    <li>
        N. V. Kumsetty, A. Bhat Nekkare, S. K. S. and A. Kumar M., "TrashBox: Trash Detection and Classification using Quantum Transfer Learning," 2022 31st Conference of Open Innovations Association (FRUCT), Helsinki, Finland, 2022, pp. 125-130, doi: 10.23919/FRUCT54823.2022.9770922.
    </li>
</ol>


<h2>Acknowledgments</h2>

<ul>
    <li>The model architecture and training process were inspired by various image classification tutorials and resources.</li>
    <li>Special thanks to the open-source community for their contributions to machine learning and computer vision.</li>
</ul>

<h2>Contact</h2>

<p>If you have any questions or need further assistance, please feel free to contact me at <a href="mailto:devinchugh.dc@gmail.com">devinchugh.dc@gmail.com</a>.</p>

<p>Happy waste segregation!</p>

</body>

</html>

# Image Forgery Detection using Custom CNN

This project demonstrates a custom-designed Convolutional Neural Network (CNN) for binary classification of **authentic** vs. **forged (spliced)** images. It draws architectural inspiration from **InceptionNet** and **ResNet**, implemented in **PyTorch**, and trained from scratch to work with spliced image detection problems.

---

## 🧠 Motivation

Most off-the-shelf deep learning models are optimized for **object detection**, not holistic image analysis. This project aims to detect **splicing forgeries** — where regions from one image are inserted into another — a challenge where **transfer learning** may be suboptimal.

A 2017 research paper [An Evaluation of Digital Image Forgery Detection Approaches](https://arxiv.org/pdf/1703.09968.pdf) highlighted **wavelet decomposition** as an effective technique. While this project doesn't implement wavelets, it aims to test an original CNN approach inspired by Inception and ResNet.

---

## 🏗️ Model Architecture

- Inspired by **InceptionNet**, the model applies multiple convolutions in parallel.
- One path inside the Inception block includes **two 3x3 convolutions with a skip connection**, inspired by **ResNet**.
- Later layers are **widened** (deeper filters) to capture global features as much as GPU limits allow.
- Weights are initialized using Xavier initialization or Glorot initialization:  
  `sqrt(2 / number of inputs to the layer)`

📌 **See architecture visual**:  
`ImageForgeryDetectionNetworkDiagram.pdf`  
[ImageForgeryDetectionNetworkDiagram (1).pdf](https://github.com/user-attachments/files/20328163/ImageForgeryDetectionNetworkDiagram.1.pdf)


📌 **Inception Block**: `inception.jpg`
![inception](https://github.com/user-attachments/assets/570b3543-ac68-4923-8f6d-d446072212f9)


📌 **ResNet Skip Block**: `ResNet.png`
![ResNet](https://github.com/user-attachments/assets/04e6f9eb-b342-4bbb-b45b-74be91263d2d)


📌 **Weight Initialization**: `InitialiseWeights.jpeg`
![InitialiseWeights](https://github.com/user-attachments/assets/1e567496-4f66-4109-bafa-2a61f2be9ab9)


---

## 🚀 Technology Stack

- **Language**: Python 3.x
- **Framework**: PyTorch
- **CUDA**: Used for GPU training on a personal laptop

---

## 📊 Training Results

- Training was conducted for ~600 epochs on ~1800 labeled images (not publicly shared).
- Loss and accuracy graphs demonstrate progressive learning and network convergence.

| Metric | Result |
|--------|--------|
| Training Loss ↓ | [Plot](https://plot.ly/~santhoshnumberone/9/) <img width="1419" alt="Screenshot 2025-05-20 at 1 17 08 PM" src="https://github.com/user-attachments/assets/63bba571-51f4-406c-a8ea-aa123195b163" /> |
| Training Accuracy ↑ | [Plot](https://plot.ly/~santhoshnumberone/11/) <img width="1421" alt="Screenshot 2025-05-20 at 1 19 14 PM" src="https://github.com/user-attachments/assets/5a9ecc5c-6a77-4866-bd49-dd346bc323a2" /> |
| Validation Metrics |  ![TerminalView](https://github.com/user-attachments/assets/2a169b12-0ba7-4e65-8c43-9694d48eabed) |


📌 **Bias & Variance explanation**: `BiasAndVariance.png`
![BiasAndVariance](https://github.com/user-attachments/assets/e76f13d2-8152-4975-b05a-c0e83c3cde80)

---

## ❌ Limitations

- Dataset is **not included** due to interview scope and license constraints.
- **Model weights** are not saved or uploaded.
- **Wavelet-based CNN** is mentioned but **not implemented**.

---

## 🤝 Acknowledgments

This project was part of an interview challenge to demonstrate architectural thinking, PyTorch proficiency, and a hands-on approach to forgery detection. (2018)

---


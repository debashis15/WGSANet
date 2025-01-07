# WGSANet: Wavelet-Gradient Synergistic Attentive Fusion Network

Welcome to the official repository for **WGSANet**, an innovative image denoising model that combines **Wavelet-Gradient Synergistic Attentive Fusion** to deliver state-of-the-art denoising performance.

---

## 🌟 Abstract

Image denoising is a fundamental task in computer vision with significant applications, including:

- Medical imaging
- Remote sensing
- Photography

Despite recent advancements in deep learning, key challenges persist:

1. **Training Complexity**: Many CNN-based denoisers rely on increasing network depth, which introduces training difficulties.  
2. **Gradient Neglect**: The role of gradient information in preserving edges and fine details remains underutilized.  
3. **Limited Wavelet Usage**: Existing models often fail to leverage wavelet-domain features effectively.

### Introducing **WGSANet**
**WGSANet**, or **Wavelet-Gradient Synergistic Attentive Fusion Network**, addresses these limitations by combining wavelet-domain and gradient-domain features through two distinct processing pipelines:

- **Wavelet-Domain Processing Pipeline (WDP)**:  
  Utilizes discrete wavelet transform (DWT) for multi-level decomposition, enhanced by a **Multi-Scale Attention Aggregator Block (MSAAB)** to process high-frequency sub-band features.  
- **Gradient-Based Processing Pipeline (GBPP)**:  
  Captures image gradients from multiple directions, ensuring superior edge preservation and detail retention.

**WGSANet** achieves remarkable denoising results, surpassing existing methods across multiple datasets. Comprehensive ablation studies validate the synergistic effectiveness of wavelet and gradient-based features.

---

## 🔬 Key Contributions

- **Wavelet-Gradient Synergy**: Leverages complementary wavelet and gradient features for enhanced denoising.  
- **Attention-Driven Design**: Employs multi-scale attention mechanisms to focus on noise-affected regions.  
- **State-of-the-Art Performance**: Outperforms leading denoisers on standard benchmark datasets.  
- **Ablation Studies**: Highlights the individual contributions of each network component.  

---

## 📰 Publication

This work has been **submitted to Pattern Recognition** for publication. Updates regarding the paper's status will be provided here.  
The paper offers a comprehensive explanation of the architecture, methodology, and performance evaluation.

---

## 📂 Code and Data

The **source code** and **datasets** will be released after the paper's acceptance. Stay tuned by starring this repository ⭐!  

---

## 📊 Results and Metrics

**WGSANet** demonstrates outstanding performance in terms of both quantitative metrics and visual quality. For specific results or further information, please contact the corresponding author.

---

## ✉️ Contact

For queries, collaborations, or additional information, feel free to reach out:

- **Debashis Das**  
  - Email 1: [ddebashisdas2108@gmail.com](mailto:ddebashisdas2108@gmail.com)  
  - Email 2: [debashis_2221cs31@iitp.ac.in](mailto:debashis_2221cs31@iitp.ac.in)  

---

## ✍️ Citation

If you find this work helpful, please consider citing it once the publication details are available.

---

## 🔑 Keywords

Computer Vision · Deep Learning · Image Denoising · Wavelet Domain · Gradient Information · Attention Mechanisms  

---

## 🤝 Acknowledgments

Thank you for your interest in **WGSANet**! Stay tuned for updates on the release of the code, datasets, and other resources.

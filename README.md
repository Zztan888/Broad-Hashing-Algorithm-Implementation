# 🗝️ Broad Hashing Algorithm Implementation (CIFAR-10)

This project implements a **Broad Hashing** framework for efficient feature representation and image retrieval using the **CIFAR-10** dataset. The implementation is guided by the methodology described in the associated research paper, adapted for optimized performance on image data.

---

## 📑 Project Components
* **Dataset:** [CIFAR-10 Python Version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
* **Core Algorithm:** Broad Learning System (BLS) inspired Hashing
* **Focus:** Feature mapping, enhancement nodes, and hash code generation
* **Language:** Python (Jupyter Notebook)

---

## 📂 Quick Links
* 📓 **Implementation (Code):** [Broad-Hashing_algorithm.ipynb](./Broad-Hashing_algorithm.ipynb)
* 📄 **Reference Guide:** [Research Paper (PDF)](./BH_algorithm.pdf)
* 📊 **Dataset Source:** [CIFAR-10 (University of Toronto)](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

---

## 🧠 Algorithm Overview
The project replicates a hashing algorithm based on the **Broad Learning System (BLS)**. Unlike deep architectures that go "deep" with many layers, this approach goes "broad" by expanding features horizontally.



### Key Steps in the Algorithm:
1. **Feature Mapping:** Input data from CIFAR-10 is transformed into mapped features using random weights.
2. **Feature Enhancement:** The mapped features are further expanded into "enhancement nodes" to capture non-linear relationships.
3. **Hashing Layer:** The broad features are compressed into compact binary hash codes. This allows for extremely fast similarity searches in large-scale databases.
4. **Optimization:** The system utilizes the Moore-Penrose pseudo-inverse for fast weight calculation instead of traditional backpropagation:
   $$W = (A^T A + \lambda I)^{-1} A^T Y$$

---

## 💻 Implementation Highlights
The notebook `Broad-Hashing_algorithm.ipynb` demonstrates:
* **Data Loading:** Extracting the 60,000 $32 \times 32$ color images from the CIFAR-10 python batches.
* **Broad Architecture Setup:** Constructing mapping nodes and enhancement nodes without the need for intensive deep-layer stacking.
* **Hash Generation:** Converting high-dimensional image features into bit-based signatures.



---

## 📈 Analysis & Results
The algorithm is evaluated on its ability to maintain high classification accuracy while significantly reducing the computational cost compared to traditional Deep CNNs.
* **Efficiency:** Focuses on the speed of training due to the broad structure.
* **Accuracy:** Performance metrics are logged for both the training and testing phases of the CIFAR-10 categories.

---

## 🚀 How to Run
1. Download the CIFAR-10 dataset from the link above and extract it into the project directory.
2. Install the required libraries:
   ```bash
   pip install numpy scipy scikit-learn matplotlib

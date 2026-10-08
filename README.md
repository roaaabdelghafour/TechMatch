# 💻 TechMatch

AI-powered laptop recommendation assistant.

## 🚀 Live Demo
🔗 [Open TechMatch](https://techmatch-5hpbgsbm5ynrb2rkgm4vae.streamlit.app/)

## 💡 Project Idea

TechMatch helps users find laptops that match their requirements such as brand, RAM, storage, GPU, and budget.

## 📥 Input

- User laptop requirements

Example:

> ASUS laptop with 16GB RAM and 1TB SSD under 60000

## 🔎 RAG

- Laptop Product Documents
- Product Specifications
- Prices
- Product Images & URLs
- Sentence Transformers for embeddings
- FAISS for vector retrieval

## 🔗 Pipeline

1. Understand User Requirements
2. Retrieve Relevant Laptops
3. Filter & Match Products
4. Rank Recommendations

## 📤 Output

- Recommended Laptops
- Price
- RAM
- Storage
- GPU
- Product Image
- Product Link

## 🛠️ Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Sentence Transformers
- FAISS

## 🚀 Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt

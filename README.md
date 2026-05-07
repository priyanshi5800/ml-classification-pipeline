# Text & Image Classification Pipeline
**IIT Bombay — Programming for Machine Learning (2025)**

## What It Does
An end-to-end unsupervised-to-supervised ML pipeline built on two 
unlabelled datasets — 1,500 text snippets and 60,000 MNIST images.
Covers text vectorization, dimensionality reduction, clustering, 
and multi-class classification with full model comparison.

## Tech Stack
- **Language:** Python
- **Libraries:** scikit-learn · NLTK · NumPy · Pandas · Matplotlib
- **Text methods:** TF-IDF · Bag-of-Words · Word2Vec · FastText
- **Image methods:** Convolution kernels (edge, blur, sharpen) via SciPy
- **Clustering:** K-Means (silhouette score: 0.46)
- **Classifiers:** MLP · Random Forest · Decision Tree

## Results

### Text Classification
| Vectorization | Classifier | Accuracy |
|---------------|------------|----------|
| TF-IDF | MLP | **89%** |
| Word2Vec | Random Forest | — |
| FastText | Decision Tree | — |

### Image Classification (MNIST subset — 15,000 images)
| Method | Classifier | Accuracy |
|--------|------------|----------|
| Raw pixel features | Random Forest | **94%** |
| Convolution features | MLP | — |

## Pipeline Overview
Raw unlabelled data → Vectorization/Feature extraction  
→ PCA / t-SNE dimensionality reduction  
→ K-Means clustering (label generation)  
→ Supervised classification → Model evaluation

## Key Takeaways
- TF-IDF outperformed Word2Vec and FastText on this dataset size
- MLP generalised better than Decision Tree on text features
- Convolution-based preprocessing improved image feature quality
- K-Means silhouette score of 0.46 indicated moderate cluster separation

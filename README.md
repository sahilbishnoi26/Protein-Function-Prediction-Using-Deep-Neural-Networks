# Protein function prediction using deep neural networks

A multi-label classification model to predict protein functions across three Gene Ontology (GO) categories: Molecular Function, Biological Process, and Cellular Component.

Key Aspects:

- Feature Engineering: Generated embeddings (e.g., ESM2, ProtT5, TAPE, ProtBert) from protein sequences and integrated InterPro annotations, adding biologically relevant features.

- Model Architectures: Developed various neural network architectures from scratch with dropout layers for better generalization.

- Logical Loss Integration: Created a custom loss function incorporating GO axioms, ensuring biologically consistent predictions.

- Per-Residue Protein Embeddings: Generated per-residue embeddings to capture local and contextual sequence features, enabling the use of sequence-based models, such as attention mechanisms and Transformers, for enhanced prediction accuracy.

- Hyperparameter Tuning and Model Selection: Conducted thorough hyperparameter optimization (e.g., learning rates, batch sizes, dropout rates) and selected the best-performing model based on F1-Score for final evaluation.

- Performance Metrics: Evaluated model with ROC AUC, Hamming Loss, Precision, Recall, and F1-Score, achieving high ROC AUC and low Hamming Loss, with improved recall through refined model configurations.

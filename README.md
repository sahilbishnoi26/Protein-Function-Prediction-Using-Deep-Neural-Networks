# Protein function prediction using deep neural networks

Introduction

Protein function annotation is a core challenge in bioinformatics, essential for understanding molecular mechanisms, disease pathways, and guiding drug discovery. Accurate and automated prediction of protein functions can significantly advance research in disease mechanism identification, drug discovery, and evolutionary studies.

Motivation and Project Purpose:
Our objective is to address the complexity of protein function annotation using multi-label classification methods enhanced by deep learning. We aim to develop neural networks that integrate multiple protein embeddings, incorporate additional biological data (e.g., InterPro annotations), and apply logical constraints to ensure biologically consistent predictions. By determining the optimal combination of embeddings, model architectures, and constraints, we strive to improve both accuracy and interpretability in predicting Gene Ontology (GO) terms.

As described by Ashburner et al. (2000), the Gene Ontology categorizes protein attributes into three main ontologies:

Molecular Function (MF): The elemental activities of a protein at the molecular level (e.g., sequence-specific DNA binding).
Biological Process (BP): The biological goals or pathways that the protein contributes to (e.g., cell death).
Cellular Component (CC): The subcellular location or environment in which the protein operates (e.g., nucleus).

Accurately annotating these GO terms is crucial for understanding protein roles in complex biological systems.



Related Work

Data-driven representation approaches leveraging large language models and deep learning have outperformed traditional, rule-based methods in protein function prediction. Pre-trained embeddings like ESM, ProtT5, and others capture meaningful sequence representations. Studies incorporating hierarchical GO structures, evolutionary data, and protein–protein interactions have demonstrated improved accuracy.

Advanced models, such as PFresGO, utilize attention mechanisms and hierarchical constraints. Transfer learning techniques leverage large unlabeled datasets to improve performance on downstream tasks. However, top-tier methods like DeepGO achieve F1-scores near 90%, indicating a gap between simpler approaches and state-of-the-art (SOTA) performance. Bridging this gap necessitates richer data integration, improved thresholding, and more computationally intensive strategies.



Dataset

The dataset integrates three primary components:

Protein Sequences: Textual amino acid sequences serve as the foundational input, transformed into embeddings that encode functional and structural information.
InterPro Annotations: Multi-hot encoded features indicating protein families, domains, and functional sites. Since each protein can belong to multiple InterPro categories, these annotations add complexity and biological richness.
GO Labels (MF, BP, CC): Proteins have multiple GO terms assigned, creating a challenging multi-label setting. The relationships between proteins and GO terms are many-to-many, complicating the prediction of the exact set of terms for each protein.

image

This multi-dimensional dataset, integrating sequences, structural/functional annotations (InterPro), and hierarchical ontologies (GO), offers a rich environment to study protein functionality. The complexity arises from the multi-label structure and the need for biologically coherent predictions.







Proposed Methodology / Detailed Description

1. Data Loading and EDA:
We began by loading datasets for Molecular Function, Biological Process, and Cellular Component ontologies. Exploratory Data Analysis (EDA) helped us understand term distributions, label imbalances, and hierarchical relationships in the data.



image

image

2. Feature Engineering and Representation Learning:

Protein Embeddings (ESM2, ProtT5, TAPE, ProtBert):
We employed four pre-trained embedding models from Hugging Face, each offering unique representation strengths.

Preprocessing: Rare amino acids replaced with 'X', spaces inserted as required by the tokenizer.
Tokenization: Used each model’s tokenizer (e.g., T5 tokenizer for ProtT5, BERT tokenizer for ProtBert).
Embedding Generation: Mean-pooled the last-layer hidden states to create fixed-size vector embeddings.
InterPro Integration:
InterPro annotations were multi-hot encoded, providing information about protein families, domains, and functional sites. Integrating these features alongside embeddings may help the model capture aspects of protein functionality not evident from sequence alone.

image

3. Model Architectures and Training:

Baseline Neural Network:Input: Fixed-size embeddings.
Hidden Layers: Two fully connected layers (1024 → 512 units) with ReLU activation and 30% dropout to reduce overfitting.
Output Layer: Sigmoid activation for multi-label probabilities.
Base Loss: Binary Cross-Entropy (BCE).
Model Variants:

Embedding-Only Models: Input only the embedding vectors.
Concatenated Models (Embedding + InterPro): Merge embedding vectors with InterPro annotations into a single input vector.
Separate Processing Models: Process embeddings and InterPro annotations independently in separate network branches, merging the features before final prediction layers. This allows specialized representation learning for each input type.

4. Logical Loss for Biological Consistency:To ensure predictions are biologically plausible and respect GO axioms, we introduced a logical loss function added to the BCE loss. This logical loss enforces constraints such as:

A Implies B (NF1): If GO term A is predicted, B should also be predicted. Penalize cases where P(A) > P(B).
Disjointness (NF2): Certain terms are mutually exclusive and cannot co-occur. Penalize if ∑P(disjoint terms) > 1.
A and B Imply C (NF3/NF4): If A and B are both predicted, C must also be predicted. Penalize when min(P(A), P(B)) > P(C).

These constraints guide the model to produce biologically consistent outputs. Earlier attempts to incorporate network data or related GO terms directly led to data leakage and inflated metrics. The logical loss avoids such issues by penalizing biologically inconsistent predictions at training time without artificially augmenting predictions.

5. Handling Complexities:

Computational Constraints:
While per-residue embeddings might provide richer context, they were infeasible (~50 hours/epoch on our GPU). We relied on mean-pooled embeddings.
Threshold Tuning:
Adjusting decision thresholds for each label can improve the Precision-Recall trade-off and potentially increase F1-scores.



Results

All planned objectives were completed, including generating multiple embeddings, integrating InterPro data, applying logical loss, and experimenting with different architectures. We tested embedding-only, concatenated, and separate-processing models.

Evaluation Metrics:

ROC AUC: Measures ranking ability. A high ROC AUC suggests good discrimination but can be misleading with class imbalance.
Hamming Loss: Fraction of incorrectly predicted labels. Lower values indicate fewer label-wise errors.
Subset Accuracy: Requires predicting all labels for an instance correctly. Often near zero for complex multi-label tasks, highlighting difficulty.
Precision & Recall: Precision checks correctness of predicted positives, Recall measures how many of the true positives are identified.
F1-Score: Harmonic mean of Precision and Recall, balancing both metrics.

Results (Validation Data):

image

Key Insights:

The model is conservative, often predicting fewer positive labels to minimize false positives. This improves Precision and lowers Hamming Loss but reduces Recall, limiting F1-scores.
ESM2 embedding-only models yielded strong F1-scores, indicating the importance of robust embeddings.
Logical loss improved biological plausibility, preventing the model from producing inconsistent GO term combinations.
High ROC AUC and low Hamming Loss do not guarantee high Recall or correct full-label sets (low subset accuracy). Predicting all relevant GO terms remains challenging.

image

image

image

image



Test Data Results

Evaluating our best-performing model (e.g., ESM2 embedding-only) on the test dataset produced similar results to validation. High ROC AUC, low Hamming Loss, and moderate F1-scores were observed, suggesting that the model’s strengths and weaknesses generalize beyond the training environment.



TEST DATA RESULTS



Comparison with SOTA

Our best F1-scores (~0.44) are significantly lower than those of DeepGO (~90%). DeepGO’s advantage likely comes from more extensive pre-training, richer biological data (e.g., protein–protein interactions), and possibly more sophisticated threshold optimization.

While logical constraints and InterPro data improved interpretability and certain metrics, bridging the gap to SOTA performance will require integrating additional biological signals, exploring per-residue embeddings if feasible, refining thresholds, and potentially leveraging more advanced architectures.



Conclusion

Our work demonstrates the complexity and potential of deep learning approaches for multi-label protein function annotation:

By integrating multiple embeddings and InterPro annotations, we provided the model with diverse feature sets.
Logical loss ensured that predictions adhered to GO axioms, enhancing biological consistency and interpretability.
Despite achieving high ROC AUC and low Hamming Loss, the model’s conservatism led to lower Recall and moderate F1-scores.
ESM2-based embedding-only models performed strongly, underscoring the importance of high-quality embeddings.
Predicting the exact combination of GO terms remains difficult, reflected in low subset accuracy and a significant gap from SOTA performance (DeepGO).

Future work may involve incorporating protein–protein interaction data, exploring more granular embeddings (if computationally feasible), refining thresholding strategies, and employing more complex architectures to improve Recall and approach SOTA performance levels.



Limitations

The main limitation was computational feasibility. While per-residue embeddings might offer more detailed insight, generating them was impractical (~50 hours/epoch). We also did not integrate protein–protein interaction data, which could provide essential contextual clues. Addressing these limitations in future work may significantly improve performance and narrow the gap to top-tier models.



GitHub Repository: For more details and code, please visit https://github.com/sahilbishnoi26/Protein-Function-Prediction-Using-Deep-Neural-Networks

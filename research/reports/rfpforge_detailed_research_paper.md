# RFPForge: A Local Retrieval-Augmented Generation Platform with Multi-Disease Prediction, Explainability, and Disease-Specific Safety Validation

## Abstract
RFPForge is a local artificial intelligence platform that combines two applied workflows within a single software system: Retrieval-Augmented Generation (RAG) for Request for Proposal (RFP) response automation, and multi-disease prediction with explainable machine learning and guarded medical-style guidance generation. The project uses FastAPI for backend orchestration, Streamlit for the user interface, SQLite for structured workflow persistence, ChromaDB for semantic retrieval, local embedding models for vector search, and local large language model (LLM) inference for grounded text generation. In the disease prediction module, the system supports diabetes, heart disease, breast cancer, and Parkinson's disease using tabular datasets and a model-selection pipeline that compares Logistic Regression, Random Forest, and XGBoost using cross-validated ROC-AUC. The best-performing model for each disease is calibrated, thresholded, and exposed through an API and interactive UI. To improve academic credibility and reduce unsafe output behavior, the project adds SHAP-based feature attribution, confidence labeling, calibrated probability display, and disease-specific retrieval validation in the RAG explanation layer. The retrieval safety layer constrains candidate passages by disease-specific query terms, lexical topic matching, forbidden-topic rejection, and a semantic similarity cutoff before the explanation text is assembled. The resulting system demonstrates how document intelligence, machine learning, explainability, and local LLM components can be integrated into a coherent engineering platform while remaining reproducible, private, and suitable for educational and research use.

## Keywords
RAG, FastAPI, Streamlit, disease prediction, XGBoost, Logistic Regression, Random Forest, SHAP, calibration, ChromaDB, semantic retrieval, explainable AI

## 1. Introduction
Applied AI systems are often presented as isolated demos: a single machine learning notebook, a standalone chatbot, or a single-purpose API. In practice, useful AI products require orchestration across data ingestion, persistence, retrieval, prompting, model inference, user interaction, and output safety. RFPForge was developed as a full-stack software system that addresses this gap.

The project began as a private RFP automation platform. Proposal teams often work across scattered internal documents and repeat a manual cycle of searching, drafting, revising, and exporting answers. A RAG-based workflow can reduce this cost by ingesting trusted organizational documents, retrieving relevant passages, and prompting a local LLM to produce grounded responses.

The project was then extended with a disease prediction module. This second workflow demonstrates how tabular clinical-style features can be passed through reproducible machine learning pipelines and then paired with human-readable explanation and safety framing. This creates a useful contrast with the RFP workflow: one side solves enterprise document intelligence, while the other explores responsible predictive assistance in a healthcare-adjacent domain.

The final system is therefore not just an AI prototype. It is a modular software platform with:

- local document ingestion and semantic retrieval
- workflow-aware API endpoints
- frontend interfaces for chat and structured prediction
- multi-model training and artifact management
- explainability outputs for prediction results
- retrieval guardrails for medical-style generated guidance

## 2. Problem Statement
The project addresses two related engineering problems.

### 2.1 RFP Automation Problem
Organizations responding to RFPs often maintain knowledge in PDFs, Word files, policy notes, and previous proposal documents. Manual searching is slow, repeated drafting is costly, and answer quality may vary across writers. A local RAG system can help retrieve internal evidence and draft grounded responses.

### 2.2 Predictive Output Interpretation Problem
Machine learning models can produce probability scores and class predictions from structured medical-style inputs, but raw predictions are difficult for users to interpret. If a language model is layered on top without controls, it may produce irrelevant or unsafe clinical content. The disease prediction workflow therefore requires both explainability and retrieval safety controls.

## 3. Objectives
The main objectives of the project are:

1. To build a local RAG-based platform for document-grounded RFP answer generation.
2. To extend the same platform with structured disease prediction workflows.
3. To compare multiple model families for each disease rather than assuming one model is always best.
4. To provide interpretable outputs using SHAP or feature-importance fallback logic.
5. To reduce misleading or irrelevant generated medical guidance through disease-specific retrieval validation.
6. To expose the complete workflow through a usable FastAPI and Streamlit application.

## 4. Related Technical Background

### 4.1 Retrieval-Augmented Generation
RAG combines parametric language modeling with non-parametric retrieval. Instead of relying only on LLM weights, the system retrieves context from a document store and injects those passages into the prompt. This improves grounding, traceability, and task relevance, especially when the knowledge base is private or domain-specific.

### 4.2 Tabular Disease Prediction
Structured disease prediction tasks are commonly approached with linear models, bagging methods, and boosted tree ensembles. Logistic Regression remains useful due to its calibrated probabilistic form and interpretability. Random Forest captures non-linear interactions using bootstrap aggregation and ensemble voting. XGBoost often performs strongly on tabular data by sequentially boosting weak learners under regularization.

### 4.3 Explainable AI
Explainability is especially important when model outputs are shown to users in sensitive domains. SHAP values attribute the contribution of individual features to a prediction and are a stronger research-facing output than generic explanatory text. When SHAP is unavailable, ranked feature importance can still provide partial interpretive support.

### 4.4 Safety in Medical-Style RAG
Medical retrieval systems face a specific failure mode: semantically similar but clinically unrelated passages can be injected into the final answer. This can happen when broad retrieval returns adjacent health topics such as procedures, symptom lists, or general wellness instructions. In this project, disease-specific retrieval validation was implemented to prevent unrelated clinical context generation in the RAG explanation module.

## 5. System Overview
RFPForge is implemented as a layered local system.

### 5.1 Backend Layer
The backend is built with FastAPI and exposes API routes for:

- chat and streaming response generation
- knowledge ingestion and retrieval
- disease registry inspection
- disease prediction and advice generation
- RFP session creation, question extraction, draft generation, finalization, and export

The application startup path initializes logging and database connectivity in [app/main.py](e:/vinayak/RFPForge/app/main.py).

### 5.2 Persistence Layer
Two persistence modes are used:

- SQLite via SQLAlchemy for structured workflow entities such as RFP sessions, questions, and drafts
- ChromaDB for semantic vector storage over ingested knowledge documents

### 5.3 Retrieval Layer
The retrieval engine uses query embeddings, vector similarity search, optional reranking, similarity thresholding, and structured metadata logging. Metadata filters are supported and forwarded to the vector store. The system also preserves both embedding similarity and reranker scores so that downstream validation logic can use the original semantic similarity rather than only reranker outputs.

### 5.4 Frontend Layer
The frontend is implemented with Streamlit in [frontend/app.py](e:/vinayak/RFPForge/frontend/app.py). It provides:

- a chat-oriented RFP workspace
- a disease prediction workspace
- registry-driven input forms for all supported diseases
- ordered-list input support for exact sample reproduction
- prediction summary cards, SHAP impact tables, and generated guidance

## 6. RFP Workflow
The RFP workflow is designed as a practical proposal drafting pipeline.

### 6.1 Knowledge Ingestion
Knowledge documents are uploaded or ingested from a configured folder. The ingestion path:

1. loads raw documents
2. extracts text
3. splits text into semantically manageable chunks
4. generates vector embeddings
5. writes chunk content, embeddings, and metadata into ChromaDB

### 6.2 Retrieval and Draft Generation
When a user asks a question or triggers draft generation for an RFP item:

1. the retrieval service embeds the query
2. candidate passages are fetched from ChromaDB
3. optional reranking refines ordering
4. the prompt builder injects the retrieved context
5. the local LLM generates a grounded response

### 6.3 RFP Session Management
The RFP API supports structured workflow operations in [app/api/rfp.py](e:/vinayak/RFPForge/app/api/rfp.py), including:

- creating RFP sessions
- parsing uploaded client files into questions
- generating first drafts
- saving manual draft edits
- regenerating answers
- finalizing responses
- exporting results to Word and Excel

## 7. Disease Prediction Workflow

### 7.1 Supported Diseases
The current disease registry is defined in [app/ml/registry.py](e:/vinayak/RFPForge/app/ml/registry.py) and includes:

- `diabetes`
- `heart_disease`
- `breast_cancer`
- `parkinsons`

Each disease configuration defines:

- dataset path
- target column
- positive class mapping
- feature schema
- artifact directory
- provenance metadata

### 7.2 Dataset Summary
The current registered disease datasets are summarized below.

| Disease | Dataset Source | Rows | Input Features | Target Mapping |
| --- | --- | ---: | ---: | --- |
| Diabetes | Local diabetes CSV | 768 | 8 | `Outcome = 1` means risk present |
| Heart Disease | UCI processed Cleveland dataset | 297 | 13 | target values `1-4` treated as disease present |
| Breast Cancer | Wisconsin Diagnostic Breast Cancer dataset from scikit-learn | 569 | 30 | target `0` treated as malignant / positive class |
| Parkinson's Disease | UCI Parkinson's dataset | 195 | 22 | `status = 1` means disease present |

### 7.3 Input Interface
The UI supports two input methods:

- manual entry through structured per-feature widgets
- ordered feature input using a pasted Python-style or JSON-style list

This second path is useful for reproducible test cases and research reporting because it allows exact sample arrays to be replayed in the same feature order expected by the trained artifact.

## 8. Machine Learning Methodology

### 8.1 Model Families
The training pipeline evaluates three model families:

- Logistic Regression
- Random Forest
- XGBoost

These are defined in `_model_grids()` in [app/ml/training.py](e:/vinayak/RFPForge/app/ml/training.py). The project does not force XGBoost for all diseases. Instead, each disease artifact stores the model family that achieved the best cross-validated ROC-AUC under the configured search space.

### 8.2 Pipeline Structure
The core training pipeline includes:

1. feature engineering
2. median imputation
3. standardization
4. SMOTE oversampling
5. model estimation

This sequencing is important because it avoids data leakage and keeps oversampling inside the train-time pipeline rather than applying it globally before validation.

### 8.3 Disease-Specific Feature Engineering
The training code currently uses `DiabetesFeatureEngineer` as a preprocessing step inside the pipeline. For diabetes this creates clinically motivated derived features such as:

- `BMI_Category`
- `Glucose_Age_Ratio`
- `Insulin_Glucose_Ratio`
- `BMI_Age_Product`

These engineered features improve the expressiveness of the tabular model beyond the raw eight original variables.

### 8.4 Cross-Validation and Hyperparameter Search
Each model family is trained using:

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- `GridSearchCV`
- multiple scoring metrics
- `refit="roc_auc"`

The best cross-validated ROC-AUC determines the selected model family for the disease artifact.

### 8.5 Calibration and Thresholding
After the best estimator is found, the pipeline is wrapped in `CalibratedClassifierCV(method="sigmoid", cv=3)` to improve the quality of predicted probabilities. The system then derives a decision threshold from training probabilities instead of hardcoding `0.5`. This helps present more believable scores and reduces the credibility problem caused by overly absolute probabilities.

### 8.6 Prediction-Time Outputs
The prediction service returns:

- binary prediction
- risk label
- reported probability
- confidence level
- selected model family
- threshold
- top contributing features
- artifact version
- disclaimer

The reported probability is capped below a displayed absolute `100%` to avoid visually suspicious output while still preserving threshold behavior.

## 9. Mathematical Foundations

### 9.1 Logistic Regression
Logistic Regression estimates the positive-class probability as:

```text
P(y=1|x) = 1 / (1 + e^-(w^T x + b))
```

The model is linear in feature space but probabilistic in output space, which often makes it a strong baseline for structured medical-style datasets.

### 9.2 Random Forest
Random Forest combines multiple decision trees trained on bootstrap samples. The impurity criterion at a node may be expressed with Gini impurity:

```text
Gini = 1 - sum(p_i^2)
```

where `p_i` is the class proportion in the node.

### 9.3 XGBoost
XGBoost optimizes a regularized additive objective:

```text
Obj = sum l(y_i, y_hat_i) + sum Omega(f_k)
Omega(f) = gamma T + 1/2 lambda ||w||^2
```

Its sequential boosting behavior often performs well on tabular datasets with non-linear interactions.

### 9.4 Standardization
Feature standardization is applied as:

```text
z = (x - mu) / sigma
```

where `mu` and `sigma` are computed from training data only.

### 9.5 Evaluation Metrics
The paper uses the following standard binary metrics:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

ROC-AUC is used as the model-selection metric because it provides threshold-independent discrimination quality.

## 10. Explainability Design

### 10.1 SHAP-Based Attribution
The project uses SHAP contributions where possible so the interface can show which input features most influenced a specific prediction. This is significantly stronger from a research perspective than displaying only generic educational paragraphs.

### 10.2 Fallback Behavior
When SHAP values are unavailable or numerically degenerate, the service falls back to ranked feature importance from the saved model metrics. This ensures that the UI still exposes a structured explanation signal.

### 10.3 UI Presentation
The disease prediction UI shows:

- prediction summary cards
- calibrated probability
- confidence level
- threshold
- a table of top feature impacts labeled as `SHAP Impact`

## 11. RAG Safety Improvements for Disease Explanations

### 11.1 Original Weakness
Generic medical retrieval can surface clinically adjacent but irrelevant passages. For example, a breast cancer result may accidentally retrieve procedure instructions for unrelated imaging or generalized symptom text. If this content is passed straight into generation, the final explanation becomes less credible and potentially unsafe.

### 11.2 Implemented Guardrails
The disease advice route in [app/api/diseases.py](e:/vinayak/RFPForge/app/api/diseases.py) adds disease-specific retrieval validation with:

- disease-specific query terms
- required lexical topic terms
- forbidden topic terms
- a minimum semantic similarity cutoff (`0.75`)
- rejection of unrelated passages before generation

### 11.3 Generation Strategy
For the registered disease workflows, the system uses deterministic disease-aware templates rather than unconstrained free-form medical explanation. This reduces the chance that irrelevant retrieved content or model drift will distort the final output.

### 11.4 Safety Statement
The system explicitly frames outputs as:

- educational screening support
- decision-support only
- not a diagnosis
- requiring clinical confirmation for real-world decisions

## 12. Experimental Results

### 12.1 Current Artifact Metrics
The current deployed artifact metrics are shown below.

| Disease | Selected Model | Best CV ROC-AUC | Test Accuracy | Precision | Recall | F1 | Test ROC-AUC | Threshold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Diabetes | XGBoost | 0.8432 | 0.7403 | 0.6207 | 0.6667 | 0.6429 | 0.8176 | 0.57 |
| Heart Disease | Random Forest | 0.8937 | 0.8333 | 0.8462 | 0.7857 | 0.8148 | 0.9420 | 0.48 |
| Breast Cancer | Logistic Regression | 0.9956 | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9967 | 0.44 |
| Parkinson's Disease | XGBoost | 0.9632 | 0.9231 | 0.9333 | 0.9655 | 0.9492 | 0.9828 | 0.20 |

### 12.2 Interpretation
The results show that XGBoost is not uniformly best across all diseases. In the current configured search space:

- diabetes selected XGBoost
- heart disease selected Random Forest
- breast cancer selected Logistic Regression
- Parkinson's disease selected XGBoost

This is an important methodological point: model family should be selected empirically per dataset rather than by preference alone.

### 12.3 Feature Signals
The learned feature signals also align with domain expectations:

- diabetes is strongly influenced by glucose, BMI-derived features, and insulin-related interactions
- heart disease gives high weight to `thal`, chest pain type (`cp`), vessel count (`ca`), and exercise-related features
- breast cancer is highly sensitive to morphology descriptors such as `radius_error`, `worst_texture`, `worst_concavity`, `worst_area`, and related tissue-shape features
- Parkinson's disease depends strongly on `spread1`, `PPE`, shimmer, and vocal-frequency-related features

## 13. Software Engineering Contributions
The value of the project is not only in model metrics. It also demonstrates several engineering contributions:

### 13.1 Unified Platform Design
The same project integrates:

- RFP document intelligence
- disease prediction
- model explainability
- guarded language generation
- exportable workflows

### 13.2 Local-First Operation
The platform is designed to run locally using:

- local vector storage
- local embedding models
- local LLM inference
- local application persistence

This architecture improves privacy for both enterprise documents and structured prediction workflows.

### 13.3 Reproducibility
The system stores:

- trained artifacts
- metrics
- dataset profiles
- cross-validation results
- figure outputs
- registry metadata

This makes the project easier to evaluate, reproduce, and extend.

## 14. Limitations
Despite the system’s strengths, several limitations remain.

1. The disease prediction module is an educational research workflow and is not clinically validated.
2. Public benchmark datasets may not generalize to broader or local populations.
3. Retrieval validation is lexical-plus-semantic rather than full ontology-driven topic verification.
4. The current feature engineering step is centered on the diabetes workflow and could be made more disease-aware in future training pipelines.
5. The frontend and API expose useful local workflows, but production deployment would still require stronger authentication, access control, and infrastructure hardening.

## 15. Future Work
Future work can proceed in several directions.

### 15.1 Research Extensions
- external validation on independent cohorts
- calibration curves and Brier score reporting
- confidence intervals for evaluation metrics
- fairness analysis across subgroups
- more rigorous ablation studies

### 15.2 Safety Extensions
- ontology-assisted retrieval filtering
- source citation previews in medical explanations
- clinician-reviewed templates per disease
- stronger refusal logic for unsupported diagnostic interpretations

### 15.3 Product Extensions
- role-based access control
- Dockerized deployment
- richer proposal analytics for the RFP workflow
- more disease modules with separate provenance and guidance policies

## 16. Conclusion
RFPForge demonstrates a practical full-stack AI architecture that combines local RAG for enterprise document workflows with explainable, calibrated, and safety-aware disease prediction. The project shows that effective applied AI is not only a matter of training a model or calling an LLM. It requires data handling, retrieval design, workflow orchestration, user interface clarity, artifact management, and output safety controls.

The RFP component demonstrates how document-grounded local AI can support proposal drafting while preserving privacy. The disease prediction component demonstrates how tabular models, explainability, and constrained generation can be integrated into a research-grade educational system. The addition of disease-specific retrieval validation, SHAP-based presentation, confidence levels, and calibrated probability reporting substantially improves the credibility of the system and addresses a real weakness in medical-style RAG pipelines.

Overall, the project provides a strong foundation for both academic reporting and future software extension. It is best understood not as a single model demonstration, but as a complete applied AI platform.

## References
1. FastAPI Documentation. https://fastapi.tiangolo.com/
2. Streamlit Documentation. https://docs.streamlit.io/
3. ChromaDB Documentation. https://docs.trychroma.com/
4. Sentence-Transformers Documentation. https://www.sbert.net/
5. Ollama Documentation. https://ollama.com/
6. XGBoost Documentation. https://xgboost.readthedocs.io/
7. scikit-learn Documentation. https://scikit-learn.org/
8. UCI Machine Learning Repository, Heart Disease Dataset. https://archive.ics.uci.edu/
9. UCI Machine Learning Repository, Parkinson's Dataset. https://archive.ics.uci.edu/
10. scikit-learn Breast Cancer Dataset Documentation. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Appendix A: Current Project File Roles
- `app/api/`: API routes for chat, knowledge, RFP, and disease workflows
- `app/knowledge_engine/`: loaders, chunking, embeddings, vector store, retrieval, prompting
- `app/ml/`: registry, training, preprocessing, explainability, service layer
- `frontend/app.py`: Streamlit application for RFP and disease workflows
- `ML-MODEL/artifacts/`: saved model artifacts, metrics, backgrounds, and figures
- `research/reports/`: research notes, metrics tables, and this paper

## Appendix B: Example Commands
```powershell
venv\Scripts\python.exe scripts\prepare_disease_datasets.py --dataset all
venv\Scripts\python.exe scripts\train_disease_models.py --disease all
venv\Scripts\python.exe -m uvicorn app.main:app --reload
venv\Scripts\streamlit.exe run frontend/app.py
```

## Appendix C: Suggested Paper Title Variants
- `RFPForge: A Local RAG and Multi-Disease Prediction Platform with Explainability and Retrieval Safety`
- `Design and Evaluation of a Local Retrieval-Augmented Generation System with Explainable Disease Prediction`
- `A Full-Stack AI Platform for RFP Automation and Safety-Aware Disease Prediction`

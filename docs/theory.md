# Theory Notes — WSI Survival Prediction

## 1. Survival Analysis Fundamentals

### Right-Censored Data
A patient is **right-censored** if their event (death) has not occurred by the end of observation.
We observe $(T_i, \delta_i)$ where $T_i$ is the observed time and $\delta_i \in \{0,1\}$ is the event indicator.
For censored patients ($\delta_i = 0$), we know only that $T_i^{true} > T_i$.

### Cox Proportional Hazards Model
The Cox model specifies the hazard function:

$$h(t | x_i) = h_0(t) \cdot \exp(\theta^T x_i)$$

where $h_0(t)$ is an unspecified baseline hazard and $\exp(\theta^T x_i)$ is the patient-specific multiplicative factor.

The **partial likelihood** (Cox 1972) eliminates $h_0(t)$:

$$L(\theta) = \prod_{i:\delta_i=1} \frac{\exp(\theta^T x_i)}{\sum_{j: T_j \geq T_i} \exp(\theta^T x_j)}$$

We replace $\theta^T x_i$ with the neural network output $f_\theta(G_i)$ — the log-hazard for slide $i$.
The denominator runs over the **risk set** $\mathcal{R}(T_i)$ — all patients still under observation at time $T_i$.

**Breslow approximation for tied times:** When multiple events occur at the same time $t$, the denominator uses the full risk set at $t$ rather than computing per-event permutations (which would be computationally intractable).

**Training objective:** Minimise the negative mean log partial likelihood:

$$\mathcal{L} = -\frac{1}{|\mathcal{E}|} \sum_{i:\delta_i=1} \left[ f_\theta(G_i) - \log \sum_{j: T_j \geq T_i} \exp(f_\theta(G_j)) \right] + \lambda \|f_\theta\|_1$$

where $\mathcal{E}$ is the set of uncensored patients and the L1 term regularises the log-hazard outputs.

**Numerical stability:** We use `torch.logcumsumexp` after sorting by descending time, which computes the log-sum-exp in a single vectorised pass with numerical stability.

---

## 2. Concordance Index (C-index / Harrell's C)

The C-index measures discrimination: of all valid pairs $(i,j)$ where $i$ had an observed event earlier than $j$, what fraction did the model correctly rank ($\text{risk}_i > \text{risk}_j$)?

$$C = \frac{|\{(i,j): T_i < T_j, \delta_i=1, \hat{r}_i > \hat{r}_j\}|}{|\{(i,j): T_i < T_j, \delta_i=1\}|}$$

- $C = 0.5$: random predictor
- $C = 1.0$: perfect risk stratification
- Clinical SOTA for LUAD: ~0.65–0.70

**Bootstrap confidence intervals:** We use the percentile bootstrap ($n=1000$ resamples) to estimate 95% CIs, which handles the non-normality of the C-index distribution.

---

## 3. Multiple Instance Learning (MIL)

A WSI cannot be processed as a single image (too large for GPU memory). MIL formulates it as a **bag** of instances (patches) with a single bag-level label (survival time).

Formally: a slide $S$ is a bag $\{x_1, \ldots, x_N\}$ of patch features. We seek a function $f: \{x_i\} \to \hat{r}$ that predicts the risk score without patch-level supervision.

Standard ABMIL (Ilse et al. 2018):

$$z = \sum_{i=1}^N a_i h_i, \quad a_i = \frac{\exp(W^T \tanh(Vh_i))}{\sum_j \exp(W^T \tanh(Vh_j))}$$

Our heterogeneity-aware pooling augments the attention gating with the node's heterogeneity features:

$$a_i = \text{softmax}_i\left( \text{MLP}([h_i ; \text{het}_i]) \right)$$

This is a principled inductive bias: we hypothesise that morphologically divergent patches carry more prognostic information, consistent with the ITH literature.

---

## 4. Graph Attention Networks (GATv2)

Standard GAT (Veličković et al. 2018) uses static attention: the attention coefficient between nodes $i$ and $j$ is computed from their features *before* propagation:

$$e_{ij} = \text{LeakyReLU}(a^T [Wh_i \| Wh_j])$$

GATv2 (Brody et al. 2022) uses dynamic attention — the attention can depend on the *combination* of both nodes' features:

$$e_{ij} = a^T \text{LeakyReLU}(W[h_i \| h_j])$$

This seemingly small change makes GATv2 strictly more expressive than GAT: it is a universal approximator over graphs, while standard GAT is not.

In our model, we use GATv2 with:
- 4 attention heads (concatenated)
- Residual connections from layer $l$ to $l+1$
- LayerNorm after each block

**Spatial graph construction:** We connect each patch to its K=8 nearest neighbours in (x,y) coordinate space using a KD-tree. This encodes tissue spatial structure — a tumour-stroma interface is represented as cross-region edges.

---

## 5. HIPT — Hierarchical Image Pyramid Transformer

HIPT (Chen et al. 2022) is a vision transformer pretrained on TCGA pathology slides using DINO (self-supervised knowledge distillation). Key properties:

1. **Domain specificity:** Trained on >10,000 pathology slides, not ImageNet. Features capture staining patterns, nuclear morphology, gland architecture — not textures relevant to natural images.

2. **Hierarchical design:** HIPT uses two levels of ViT — a patch-level ViT-S/16 (256×256 → 192-dim token) and a region-level ViT that aggregates 4096×4096 regions. We use the patch-level encoder for memory efficiency.

3. **No labels required:** DINO training is fully self-supervised, meaning HIPT does not use any diagnostic labels. This makes it suitable as a universal feature extractor.

**Fallback:** When HIPT weights are unavailable, we fall back to ResNet-50 with ImageNet weights. Performance degrades (~0.03 C-index drop) because ImageNet features do not align with pathology morphology.

---

## 6. Intra-Tumour Heterogeneity (ITH)

ITH refers to the co-existence of morphologically and genomically distinct subclonal populations within a single tumour. It has been associated with:

- Chemotherapy resistance (subclones that survive treatment)
- Higher metastatic potential
- Poor prognosis in lung, breast, colorectal, and other cancers

**Computational proxies for ITH (our features):**

| Feature | Definition | Clinical rationale |
|---------|-----------|-------------------|
| Local entropy $H_i$ | Shannon entropy of L2-norm histogram in patch $i$'s K-neighbourhood | High entropy = diverse morphological phenotypes nearby |
| Cosine dissimilarity $d_i$ | $1 - \cos(h_i, \bar{h}_{N(i)})$ where $\bar{h}_{N(i)}$ is the neighbourhood centroid | Measures how "outlier" patch $i$ is in feature space |
| Feature spread $\sigma_i$ | Std of pairwise distances among neighbours | Spread of feature distributions = phenotypic diversity |

These features are computed without any labels and added to every node's feature vector before graph construction.

---

## 7. Integrated Brier Score (IBS)

The IBS measures calibration across the full time horizon. At each time $t$, the Brier score is:

$$\text{BS}(t) = \frac{1}{N} \sum_{i=1}^N \left[ \hat{S}(t|x_i) - \mathbf{1}(T_i > t) \right]^2 \cdot w_i(t)$$

where $w_i(t)$ is the inverse probability of censoring weight (IPCW) that corrects for the bias introduced by censored patients.

The IBS integrates over time:

$$\text{IBS} = \frac{1}{t_{max} - t_{min}} \int_{t_{min}}^{t_{max}} \text{BS}(t) \, dt$$

- Random model: IBS ≈ 0.25
- Good model: IBS < 0.20
- Perfect model: IBS = 0

---

## 8. Kaplan-Meier Estimator

The KM estimator is the non-parametric maximum likelihood estimate of the survival function:

$$\hat{S}(t) = \prod_{t_j \leq t} \left(1 - \frac{d_j}{n_j}\right)$$

where $d_j$ = number of events at time $t_j$, $n_j$ = number at risk just before $t_j$.

We use KM curves to visualise risk stratification: split patients at the median predicted risk score, estimate separate KM curves for high/low risk groups, and assess separation with the log-rank test.

**Log-rank test:** Tests $H_0$: the two groups have the same survival function. A $p < 0.05$ confirms statistically significant risk stratification.

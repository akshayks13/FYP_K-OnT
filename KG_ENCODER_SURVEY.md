# KG encoders for CrystaLLM — what each method is, and why it is or is not for us

**Project:** Battery knowledge graph + CrystaLLM (CIF language model)  
**Updated:** 2 September 2026  
**Point of the search:** OnT uses a **language model as an ontology encoder**. We wanted the same idea for a **knowledge graph**, instead of HGT.

This note is only about **encoders** (KG → vectors). Those vectors could later be keys/values in CrystaLLM cross-attention. We do **not** discuss a full pipeline here.

**The actual job (strip the papers):** we have a KG; we need embeddings to connect it to CrystaLLM. For *that*, the best encoder is **HGT**. KG-BiLM, KEPLER, SimKGC, RotatE, RulE, and OnT are either the wrong object, the wrong graph type, or extra machinery that does not make a better lookup table for this KG. Detail below.

**Our constraints (used in every “for us” verdict):**

- The graph is a **battery materials KG**: ~5,000 materials, typed nodes (Material, Element, Crystal, UnitCell, …), numeric literals (band gap, lattice), ~8.4M RDF triples. Almost **no Wikipedia-style descriptions**.
- CrystaLLM has a **closed 371-token CIF vocabulary**. KG text in the prompt becomes `<unk>`. So we need **vectors**, not prompt text.
- Hardware is a **laptop** (MPS), not a 64-GPU cluster.
- We already have **HGT** vectors and a **cross-attention** injector. The question is whether another encoder is a better *source of vectors*.

---

## 0. OnT (so the analogy is explicit)

**Paper:** Yang, Chen, Horrocks. *Language Models as Ontology Encoders* (OnT). ISWC 2025.  
**Code:** https://github.com/HuiYang1997/OnT

**What it does.**  
OnT is **not** a KG method. It embeds **OWL ontologies** (Description Logic EL): classes, subclass axioms, conjunction, existential restrictions `∃r.C`.

Recipe:

1. **Verbalize** a concept as English (e.g. “something that teaches some class”).
2. Encode that sentence with a **pretrained language model** (BERT / sentence transformer), mean-pool.
3. Map the vector into **hyperbolic space** (Poincaré ball) so **hierarchy** (`C ⊑ D`) is geometric.
4. Encode **roles** as rotations/transitions so `∃r.C` is not just another walk.

It exists because older ontology embeddings split badly: **boxes/balls** keep logic but ignore labels; **OWL2Vec*** uses labels/walks but drops logic. OnT tries to keep **both**.

**Why it is good (in its domain).**  
SNOMED, Gene Ontology, Uberon have rich **English class names**. OnT uses those names *and* still trains for subsumption and `∃r.C`. The paper reports large gains on axiom prediction/inference vs geometric and walk baselines.

**Why it is not good for us.**  
Our problem is not “is *CathodeMaterial* a subclass of *Material*?”. It is “what vector should represent **LiFePO4** vs **NMC811**?”. That is the **ABox** (instances and facts), not the **TBox** (class axioms). OnT on our tiny schema (~10 types, ~11 relations) would embed *Material* and *Crystal*, not the 5,000 compounds. The OnT authors **exclude** KG-BERT and KEPLER for this reason: KGs are relational facts, ontologies are DL axioms.

**Verdict:** Cite as the *ontology* analogue. Do not run OnT as the CrystaLLM encoder.

---

## 1. KEPLER

**Paper:** Wang, Gao, Zhu, Zhang, Liu, Li, Tang. *KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation*. TACL 2021.  
**Code:** https://github.com/THU-KEG/KEPLER (cloned in this repo)

**What it does.**  
KEPLER is one **RoBERTa**, trained with two losses:

- **MLM:** normal masked language modelling on Wikipedia + BookCorpus (so it stays a language model).
- **KE:** for each triple `(h, r, t)`, encode the **Wikipedia first section** of `h` and of `t` with the same RoBERTa. The `<s>` hidden state **is** the entity embedding. Score with **TransE**: `h + r ≈ t`.

They do **not** store a learned lookup table of IDs. The **description is the embedding**. That is why they can embed **unseen** entities if a new paragraph exists (inductive KE). They built **Wikidata5M** (~4.6M entities, aligned Wikipedia text) because older KG benchmarks had no descriptions.

They **do not** add cross-attention or entity linkers (unlike ERNIE / KnowBert). At inference, KEPLER *is* RoBERTa. For link prediction they dump `E_<s>(description)` to a file.

**Why it looks attractive.**  
This *is* “language model as KG encoder.” The recipe matches OnT: text in, LM encodes, **geometry loss** (TransE instead of hyperbolic EL). OnT cites KEPLER as the KG line they are not doing. If we had a paragraph per material, we could freeze those vectors and use them as K/V in CrystaLLM CA. The method is famous and citable.

**Why it is not good for us.**

1. **Input mismatch.** KEPLER encodes descriptions ≥5 words from Wikipedia. Our nodes are `mp-6668`, `LiFePO4`, lattice floats. There is no encyclopedic text. Fake one-liners (“LiFePO4 is a cathode”) are not the data the method was trained for.
2. **Compute.** The paper’s pretrain assumes large distributed GPU training (their example is on the order of **64×16GB V100s**, 125k updates, fairseq, RoBERTa-base). Not a laptop FYP.
3. **Wrong downstream.** KEPLER is evaluated on TACRED, FewRel, GLUE, LAMA — English NLP — and Wikidata link prediction. It is not a CIF model and does not produce CIF-space vectors. We would still need `kg_proj` + CA.
4. **Not an injector.** Using KEPLER does not “add a CA layer.” It only produces embeddings.

**Verdict:** Right *family*, wrong *data and budget*. Cite in related work. Do not train official KEPLER on Wikidata or on our TTL as-is.

---

## 2. SimKGC

**Paper:** Wang, Huang, Liu, Tu. *SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models*. ACL 2022.  
**Code:** https://github.com/intfloat/SimKGC

**What it does.**  
Same family as KEPLER/KG-BERT: a **pretrained LM encodes the text of entities** (name, and description if any). The difference is the **loss**.

KG-BERT feeds the whole triple `[CLS] head [SEP] relation [SEP] tail` into BERT and classifies true/false. That is slow (every candidate tail is a full forward pass) and, the SimKGC authors argue, weak contrastive learning.

SimKGC instead:

- Encodes **head+relation** and **tail** as two vectors.
- Trains with **InfoNCE**: true tails close, negatives far.
- Uses **in-batch**, **pre-batch**, and **self-negatives** so one batch gives many negatives cheaply.

They report large link-prediction gains vs TransE/RotatE *and* vs earlier text methods on WN18RR and Wikidata5M (including inductive). Code is a normal PyTorch project, not a fairseq fork.

**Why it is good for us (conditionally).**

1. It is the **practical OnT-for-KG**: LM encoder + structure-aware training, without Wikipedia-scale pretraining.
2. **Names are enough** for their model to run; descriptions help but are not mandatory the way KEPLER’s Wikipedia dump is.
3. **Public code**, contrastive training, feasible on one GPU / a determined laptop run on a **filtered** triple set (materials + elements, not every site–site bond).
4. Output is exactly what CA needs: **one vector per entity**.
5. If we **verbalize** (`LiFePO4 is a material. It has element Fe. Band gap is 0.66.`), the LM actually sees chemistry-ish text. That is the OnT move (verbalize, then encode).

**Why it is not automatically good.**

1. **Without verbalization** our “text” is `mp-6668` / formula tokens. BERT was not pretrained on CIF or Materials Project IDs. The LM advantage collapses; RotatE/HGT become more honest.
2. **Numeric literals** are not first-class. Band gap is only in the vector if we put it in the sentence. HGT reads `.x` as numbers.
3. **Heterogeneous types** (Material vs AtomicSite) are not HGT-style meta-relations; they are just different strings.
4. Link-prediction SOTA on WN18RR does **not** mean better CIF. CA still has to mix the vectors; syntax damage from CA is a separate issue.
5. Still a **second model** to train, debug, and export, plus a new `kg_dim`.

**Verdict:** Best *LM-as-KG-encoder* we can actually consider — **if** we write entity sentences. Not magic; not a replacement for CA. Stronger thesis story than “we downloaded KEPLER.”

---

## 3. KG-BERT and LMKE (same drawer as SimKGC)

**KG-BERT** (Yao et al., 2019): BERT as a triple classifier on concatenated descriptions. Historically first “BERT for KGC.” Slow, weaker contrastive setup than SimKGC. **For us:** same text requirement, worse engineering. Prefer SimKGC if we pick this family.

**LMKE** (Wang et al., 2022), *Language Models as Knowledge Embeddings*: title is the OnT analogue. Contrastive description-based KE; handles missing descriptions by using names/relations. **For us:** conceptually the same as SimKGC; SimKGC has clearer code and ACL 2022 results. Cite LMKE for the name; implement SimKGC if we implement any of this family.

**Verdict:** Do not implement KG-BERT. LMKE = citation twin of SimKGC.

---

## 4. KG-BiLM

**Paper:** Chen, Wang, Li, Guo, He. *KG-BiLM: Knowledge Graph Embedding via Bidirectional Language Models*. 2025 (arXiv:2506.03576).

**What it does.**  
Treats the KG as a **token sequence of triples** (and descriptions when they exist). Takes a **decoder-only** transformer and:

1. **Bidirectional Knowledge Attention:** removes the causal mask; allows attention along graph adjacency / multi-hop so tokens that share a relation path can see each other.
2. **Knowledge-Masked Prediction:** masks entity or word tokens and predicts them (they use a position-shifted loss).
3. **Contrastive Graph Semantic Aggregation:** two dropout views of a subgraph, InfoNCE on pooled `H^(L)`.

Goal stated in the paper: every **entity** and every **token** lives in one `R^d`. Evaluated as **link prediction** (WN18RR, FB15k-237, Wikidata5M, FB15k-237N). On **structure-only** graphs (no text) it still works — unlike KEPLER. On Wikidata5M with text they report beating KEPLER by a large MRR margin (they quote KEPLER MRR 21.0 vs their 40.3).

**Why it looks good.**  
Newest “LM as KG encoder.” Can train on **triples only**, which matches our TTL better than KEPLER. Explicitly produces embeddings. Graph-aware attention is a story for “preserving structure,” closer to HGT’s neighbourhood than TransE-only KEPLER.

**Why it is not good for us.**

1. **It is a whole second language model** trained on the graph. That is HGT-level (or worse) engineering, plus LM training, for a **lookup table** we will project into 512-d CA.
2. **No mature, small “fit on MPS in a week” stack** comparable to SimKGC’s repo (their code pointer is anonymous.4open.science).
3. Gains are **link-prediction MRR**, not CIF quality. NBFNet still beats them on FB15k-237 structure-only; “newer” ≠ “better for CrystaLLM.”
4. Still does not ingest **float literals** as HGT `.x` does, unless verbalized into the token sequence.
5. **Not a CA module.** Same plug-in as KEPLER: dump vectors, then our injector.

**Why not KG-BiLM instead of HGT (direct question).**  
KG-BiLM *can* dump entity vectors. That is not enough. Those vectors still go through the same `kg_proj` + CA. The encoder has to match **this** graph:

- We have **typed nodes** and **numeric `.x`**. HGT uses both. KG-BiLM treats the KG as a **token sequence**; types are not meta-relations, floats are not features unless written as words.
- We do **not** have Wikipedia. KG-BiLM can train on triples only, but then it is a **second full language model** on the graph — heavier than HGT — for a lookup table. Papers report **link-prediction MRR**, not CIF quality.
- “It is an LM” does not make CA better. CrystaLLM never sees KG-BiLM’s tokens; it only sees a vector.

**Verdict:** Valid research cousin of OnT/KEPLER. **Do not use it as the CrystaLLM encoder.** If we ever wanted an LM encoder, SimKGC is the smaller instance of the same idea. For “KG → vectors → CrystaLLM,” use HGT.

---

## 5. RotatE (and what came after)

**Paper:** Sun, Deng, Nie, Tang. *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space*. ICLR 2019.

**What it does.**  
No language model. Each entity is a complex vector; each relation is a **rotation** of modulus 1: `t ≈ h ∘ r` (Hadamard product). Trained with self-adversarial negative sampling. The paper’s claim is that this geometry can **model and infer**:

- symmetry / antisymmetry  
- inversion  
- composition  

TransE cannot do symmetry well; DistMult cannot do inversion; ComplEx cannot do composition (in their table). RotatE is the 2019 answer to “preserve relation *patterns*.”

**Why it is good.**

1. Input is **exactly triples** — we have those.
2. **No text**, so KEPLER’s Wikipedia hole does not apply.
3. In **PyKEEN** and **PyG**; laptop-scale on a filtered graph.
4. Clear “rules” story for a thesis chapter (the three patterns).
5. **HousE** (Li et al., ICML 2022) is the honest 2019→2022 upgrade: high-dimensional Householder rotations **and** 1-to-N / N-to-N mappings (RotatE’s known hole). QuatE, Rotate3D, DualE, GoldE (2024), DaBR (2025) sit on the same line.

**Why it is not good as our main encoder (vs HGT / vs OnT-family).**

1. **Wrong kind of rules for this KG.** Our graph is heterogeneous and numeric, not WordNet-style `father ∘ father = grandfather`. Inverse edges (`belongsToElectrode` / `usesMaterial`) exist, but the distinctive content is **types + floats + 1-to-many sites**.
2. RotatE **flattens types**: Material and Element are just IDs. HGT’s meta-relation `(src_type, edge, dst_type)` is discarded.
3. **No `.x`.** Band gap never enters unless we concatenate features by hand.
4. **1-to-many:** one material → many properties / sites. RotatE’s distance-preserving rotations are weak here (HousE’s motivation).
5. **Not an LM encoder.** It does not answer the OnT question. It answers “GNN vs classical KGE.”

**Verdict:** Best **non-LM** alternative to HGT if we insist on no GNN. Concatenate `Material.x` / `Element.x` if we use it. **Not** the OnT analogue. HousE over RotatE only if we commit to this family and want a 2022 citation.

### 5.1 Does RotatE handle heterogeneous graphs?

**No — not in the HGT sense.** It handles **many relation types**. It does not handle **many node types with type-specific features**.

A heterogeneous graph (ours) has two kinds of variety:

| | What it means | RotatE | HGT |
|--|----------------|--------|-----|
| **Edge types** | `hasStructure`, `hasElement`, `hasBondTo`, … | **Yes.** Each relation is its own rotation. FB15k-style KGs are multi-relational; that is RotatE’s setting. | Yes. Meta-relation in attention. |
| **Node types** | Material vs Element vs Crystal vs UnitCell, each with different `.x` | **No.** Every node is the same kind of ID in one table. LiFePO4 and Fe are just two points. | **Yes.** Separate parameters and feature matrices per type. |

RotatE’s world is a flat triple `(entity_17, relation_3, entity_88)`. Types exist only *indirectly* (materials tend to use `hasStructure`). What that drops:

- **Type-specific numbers** — Material is 11-d, Element is 7-d. RotatE has no `.x` unless concatenated by hand.
- **Type-specific behaviour** — HGT uses different weights for `Material—hasStructure→Crystal` vs `AtomicSite—hasBondTo→AtomicSite`. RotatE only has two rotation vectors; nodes still share one space.
- **1-to-many by type** — one material → many sites. RotatE is already weak at 1-to-N.

You *can* train RotatE on a hetero KG by dumping every `(head, rel, tail)` regardless of type. That is **multi-relational**, not heterogeneous in the GNN sense. The graph is flattened first.

---

## 5b. RulE (RotatE + first-order rules)

**Paper:** Tang, Zhu, Liang, Zhang. *RulE: Knowledge Graph Reasoning with Rule Embedding*. ACL Findings 2024. https://aclanthology.org/2024.findings-acl.256.pdf  
**Code:** https://github.com/XiaojuanTang/RulE

**What it does.**  
RulE is **not** an LM encoder and **not** a hetero-GNN. It is **RotatE plus first-order logical rules**.

RotatE only sees facts: `(Alice, wife_of, Bob)`. RulE also sees rules that apply to everyone, e.g.:

```
∀x,y    wife_of(x,y) → husband_of(y,x)
∀x,y,z  r1(x,y) ∧ r2(y,z) → r3(x,z)
```

It embeds **entities, relations, and rules** in the same complex space. For a chain rule `r1 ∧ … ∧ rl → r_head` it asks `r_head ≈ (r1 ∘ … ∘ rl) ∘ R_rule` (the rule is an extra rotation). Training uses triple loss **and** rule loss so embeddings are regularized by the rules.

At prediction it combines (1) the RotatE-style triple score with (2) **soft rule grounding**: find paths that fire a rule, give each rule a **confidence** (not hard true/false — “works in NYC but lives in NJ”), mix with an MLP. Biggest empirical gains are on **Family, Kinship, UMLS** (rule-heavy graphs). On FB15k-237 the lift over RotatE is small (MRR 0.337 → 0.362).

Rules are **chain** rules (symmetry, inversion, composition, hierarchy), mined or expert-defined.

**Why it looks useful.**  
This is the 2024 paper that actually **injects first-order rules** into RotatE — closer to “preserve rules” than vanilla RotatE’s algebraic patterns alone. Soft confidence avoids brittle logic.

**Why it is not useful for us.**

1. Still the **flat entity table**. Same hetero limitation as RotatE. No `.x`.
2. Needs **real first-order rules**. Our RDF is mostly a **fixed schema** (material → crystal → cell → site), true by construction, not uncertain implications like Family/Kinship.
3. The MLP / BFS grounding scores **missing triples** (link prediction). For CrystaLLM we only need **entity vectors** from stage 1; the rule-reasoning head is not a CA context.
4. Mining rules and running BFS on ~8M triples / 156k sites is extra cost for little CIF-relevant signal.

**Verdict:** Cite if we discuss RotatE + logic. **Do not implement as the encoder** for CrystaLLM.

---

## 6. HGT (current)

**Paper:** Hu, Dong, Wang, Sun. *Heterogeneous Graph Transformer*. WWW 2020.

**What it does.**  
A Transformer-style GNN for **heterogeneous** graphs. Attention and message weights depend on the **meta-relation** `⟨node type of source, edge type, node type of target⟩`. Multi-layer stacking mixes **multi-hop neighbours** (soft meta-paths). Node features `.x` are the input.

Our pipeline already: TTL → `HeteroData` → HGT link prediction → `node_embeddings.pt` → composition lookup.

**Why it is good for us.**

1. **Matches the graph we actually built:** 10 types, typed edges, literals in `.x`, 1-to-many.
2. **Preserves more of “this KG”** than RotatE (types, neighbourhood, numbers) and more than KEPLER (no need for Wikipedia).
3. **Already trained and wired** to CrystaLLM CA. August CA showed **formula-element bias** vs vanilla (e.g. stayed on Li/Fe/P/O when vanilla wrote Ba) — so the vectors are not useless.
4. No verbalization step.

**Why it is not good (or not enough).**

1. **Not an LM encoder.** It does not fulfil “OnT but for KG.”
2. **Our HGT is a weak ranker** (val MRR ≈ 0.07, Hits@1 ≈ 0.03). Neighbourhood is blurry; it does not contain `fract_x` / occupancy.
3. **No OWL logic** (disjointness, negation). Missing edge ≠ forbidden. Same as RotatE/KEPLER.
4. CA **syntax damage** was from the injector, not proof that HGT is the wrong encoder — but HGT also cannot “fix CIF” by itself.

**Verdict:** For the job “KG → embeddings → CrystaLLM,” **this is the one to use.** It preserves this battery graph better than RotatE/RulE (types + `.x` + neighbours) and does not need Wikipedia like KEPLER/KG-BiLM/SimKGC. It is the wrong answer only if the *research question* is specifically “LM as KG encoder” (OnT analogue) — that is a different question from connecting this KG to CrystaLLM.

---

## 7. KG-Adapter

**Paper:** Tian et al. *KG-Adapter: Enabling Knowledge Graph Integration in Large Language Models through Parameter-Efficient Fine-Tuning*. ACL Findings 2024.

**What it does.**  
**Not an embedding model.** It **injects** a subgraph into **Llama/Mistral** (~7B) with ~28M trainable params:

1. **SEHI:** fuse pretrained KGE vectors with the **LLM’s subword embeddings of the entity name**.
2. **RGAT** (node-centric) + **MLP_trip** (one vector per triple).
3. **Bi-directional cross-attention** between LLM hidden states and those triples.
4. LLM weights frozen; trained on QA (OBQA, CSQA, WebQSP, CWQ).

Ablation: dropping SEHI barely hurts on 7B models (the LM already “knows” English entity names). Dropping GNN or `MLP_trip` hurts more.

**Why it looks relevant.**  
It is the paper that *is* “add KG as cross-attention.” Decoder-only LM, PEFT, freeze the base — same injector pattern as our CrystaLLM CA.

**Why it is not good as an encoder (and only partly as an injector).**

1. It **requires embeddings as input**. It does not replace HGT/KEPLER/SimKGC.
2. **SEHI fails on CrystaLLM:** Llama can embed `"lithium iron phosphate"`; our tokenizer maps that to `<unk>`. Their “word embeddings carry the KG” trick is for large **text** LMs, not a 371-token CIF model.
3. Tasks are **English multiple-choice / KGQA**, not CIF.
4. Internal **GNN** is bringing graph nets back into the LM — the thing we wanted to leave if the goal is OnT-style LM encoding.

**Verdict:** Related work for **injection**. Do not treat it as an alternative to HGT/KEPLER/SimKGC.

---

## 8. Side-by-side: good for us or not

| Method | What it actually is | Good for us? | Why |
|--------|--------------------|--------------|-----|
| **OnT** | LM + hyperbolic geometry on **OWL classes** | **No** | TBox, not materials |
| **KEPLER** | RoBERTa on **Wikipedia** + TransE | **No (to train)** | No descriptions; GPU cluster. **Yes to cite** as OnT’s KG cousin |
| **SimKGC** | LM on **entity text** + contrastive triples | **Yes, if we verbalize** | Runnable OnT-for-KG. **No**, if we refuse to write sentences |
| **KG-BERT / LMKE** | Same family as SimKGC | **No extra implement** | Cite; SimKGC is the one to run |
| **KG-BiLM** | Bidirectional LM over triples | **No** | Can dump vectors, but ignores types/`.x`; second LM; LP metrics ≠ CIF. Not instead of HGT |
| **RotatE / HousE** | Rotations on triples, no LM | **Only as non-LM fallback** | Multi-relational yes; **hetero node types no**. Concat `.x` if used |
| **RulE** | RotatE + first-order rule embeddings | **No** | Needs Family-style logical rules we do not have; still not hetero |
| **HGT** | Typed GNN + `.x` | **Yes — use this** | Best match for this KG → CrystaLLM vectors |
| **KG-Adapter** | CA inside Llama | **No as encoder** | Injector for a different LM |

**If the question is only “I have a KG, I want embeddings for CrystaLLM”:** HGT.  
**If the question is “OnT but for KG”:** SimKGC (verbalize), cite KEPLER; still worse than HGT for *this* graph.  
**If the question is “preserve first-order rules in RotatE”:** RulE — not our job.

None of them stores OWL negation. None of them writes CIF by itself. A better encoder does not by itself repair CA occupancy syntax.

---

## 9. Citations

- Yang, Chen, Horrocks. Language Models as Ontology Encoders (OnT). ISWC 2025.  
- Wang et al. KEPLER. TACL 2021.  
- Wang et al. SimKGC. ACL 2022.  
- Yao et al. KG-BERT. 2019.  
- Wang et al. LMKE. 2022.  
- Chen et al. KG-BiLM. arXiv:2506.03576, 2025.  
- Sun et al. RotatE. ICLR 2019.  
- Li et al. HousE. ICML 2022.  
- Hu et al. Heterogeneous Graph Transformer. WWW 2020.  
- Tian et al. KG-Adapter. ACL Findings 2024.  
- Tang et al. RulE: Knowledge Graph Reasoning with Rule Embedding. ACL Findings 2024.

Slides (same content, no pipeline): `KG_ENCODER_SURVEY.pptx`.

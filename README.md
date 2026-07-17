# AOC-IFCN Workshop: Generative AI for Sleep Signals Through Vibe Coding

**From PSG Signals to Deep Learning: An AI-Assisted Hands-on Approach**  
IFCN Asia-Oceania Chapter Congress · Learning Series, Taipei — Topic 3  
Po-Yu Lin, MD — National Cheng Kung University Hospital

Automated detection of **REM Sleep Behavior Disorder (RBD)** — one of the strongest
early warning signs of Parkinson's disease — from polysomnography, by measuring **REM
sleep without atonia (RSWA)**. In one browser session you go from raw sleep signals →
a transparent rule → a small deep-learning model → model interpretation.

---

## 🚀 Open the notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Po-yuLin/AOC-IFCN_workshop/blob/main/workshop_notebook.ipynb)

No installation needed — the notebook installs its own packages and downloads the data
automatically. Open it in Google Colab, enable a GPU, and run the cells top to bottom
with `Shift+Enter`.

This is a **vibe-coding** notebook: many code cells are left blank, each with an example
prompt above it — you generate the code with Colab's built-in AI (Gemini), run it, and
fix errors by pasting them back. Stuck? A fully worked
[answers notebook](https://colab.research.google.com/github/Po-yuLin/AOC-IFCN_workshop/blob/main/workshop_notebook_answers.ipynb)
is also provided.

---

## Before the workshop

Please prepare your environment in advance:

👉 **[setup_instructions.md](setup_instructions.md)** — Google Colab (recommended, nothing
to install) or an optional local setup with Miniforge.

The key step is to **enable a T4 GPU** in Colab and run the one-cell test in the guide,
then **disconnect the runtime** to preserve your free GPU quota for the session.

---

## What you'll do

1. **Module 1 — Signals → RSWA → a rule.** Read the three PSG channels (EEG, EOG,
   chin EMG), score RSWA per **3-second mini-epoch**, and build a transparent
   rule-based RBD test (sensitivity / specificity, ROC).
2. **Module 2 — Train a 1D-CNN.** A small multi-scale (Inception-style) network learns
   RSWA from the raw signal; watch it **overfit** on too little data.
3. **Module 3 — Fight overfitting.** Data augmentation and more data; evaluate on a
   **held-out test set** (confusion matrix, sensitivity, PPV) and compare with the rule.
4. **Module 4 — Open the black box.** Grad-CAM and channel ablation ask: *does the
   model look where a clinician looks?*

Several steps are **"vibe-coding" challenges** — you prompt an AI assistant to write the
code, then run it.

---

## What's in this repository

Everything a participant needs is at the **top level**:

- **`workshop_notebook.ipynb`** — the vibe-coding notebook (blank cells + example prompts).
- **`workshop_notebook_answers.ipynb`** — the same notebook with every blank filled in.
- **`setup_instructions.md`** — environment setup (Colab or optional local).
- **`README.md`** — this file.

**`data preprocessing scripts/`** — a slim, self-contained pipeline that regenerates the
workshop dataset from the public PhysioNet CAP recordings (download → 3-second REM
mini-epochs + labels → float16 shrink → Module 1 demo clips). Provided for transparency
and reproducibility only; **participants don't need it** — the notebook downloads the
prepared data automatically. See that folder's own `README.md` for the steps.

---

## Data

Public, de-identified polysomnography from the **PhysioNet CAP Sleep Database**. The
notebook automatically downloads a prepared copy (3-second REM mini-epochs for every
subject, plus two short raw recordings for the live demo) from this repository's
[Releases](https://github.com/Po-yuLin/AOC-IFCN_workshop/releases) — no login, no manual
download. The data is redistributed under the Open Data Commons Attribution License
(ODC-By); please cite PhysioNet and the CAP Sleep Database (references 1–2 below).

---

## Questions

Open an [issue](https://github.com/Po-yuLin/AOC-IFCN_workshop/issues) or email
p88124019@gs.ncku.edu.tw.

---

## References

**Data source**

1. Terzano MG, Parrino L, Sherieri A, et al. Atlas, rules, and recording techniques for
   the scoring of cyclic alternating pattern (CAP) in human sleep. *Sleep Med.*
   2001;2(6):537–553.
2. Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet:
   components of a new research resource for complex physiologic signals. *Circulation.*
   2000;101(23):e215–e220.

**Clinical background and RSWA scoring**

3. American Academy of Sleep Medicine. *International Classification of Sleep Disorders.*
   3rd ed. Darien, IL: American Academy of Sleep Medicine; 2014.
4. Schenck CH, Boeve BF, Mahowald MW. Delayed emergence of a parkinsonian disorder or
   dementia in 81% of older men initially diagnosed with idiopathic rapid eye movement
   sleep behavior disorder: a 16-year update on a previously reported series. *Sleep Med.*
   2013;14(8):744–748.
5. Lapierre O, Montplaisir J. Polysomnographic features of REM sleep behavior disorder:
   development of a scoring method. *Neurology.* 1992;42(7):1371–1374.
6. Ferri R, Manconi M, Plazzi G, et al. A quantitative statistical analysis of the
   submentalis muscle EMG amplitude during sleep in normal controls and patients with
   REM sleep behavior disorder. *J Sleep Res.* 2008;17(1):89–100.
7. Ferri R, Rundo F, Manconi M, et al. Improved computation of the atonia index in normal
   controls and patients with REM sleep behavior disorder. *Sleep Med.* 2010;11(9):947–949.
8. Berry RB, Brooks R, Gamaldo CE, et al; for the American Academy of Sleep Medicine.
   *The AASM Manual for the Scoring of Sleep and Associated Events: Rules, Terminology and
   Technical Specifications.* Version 2.1. Darien, IL: American Academy of Sleep Medicine;
   2012.
9. Frauscher B, Iranzo A, Gaig C, et al; SINBAR (Sleep Innsbruck Barcelona) Group.
   Normative EMG values during REM sleep for the diagnosis of REM sleep behavior disorder.
   *Sleep.* 2012;35(6):835–847.
10. McCarter SJ, St Louis EK, Duwell EJ, et al. Diagnostic thresholds for quantitative REM
    sleep phasic burst duration, phasic and tonic muscle activity, and REM atonia index in
    REM sleep behavior disorder with and without comorbid obstructive sleep apnea. *Sleep.*
    2014;37(10):1649–1662.
11. Leclair-Visonneau L, Feemster JC, Bibi N, et al. Contemporary diagnostic visual and
    automated polysomnographic REM sleep without atonia thresholds in isolated REM sleep
    behavior disorder. *J Clin Sleep Med.* 2024;20(2):279–291. doi:10.5664/jcsm.10862.

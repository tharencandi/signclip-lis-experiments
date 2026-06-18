// #import "template/cvpr2025.typ": cvpr2025, conf-name, conf-year, eg, etal, indent
#import "template/cvpr-custom.typ": cvpr_report, heading3, dtcite, distill-outset, distill-page, distill-screen
#import "/logo.typ": LaTeX, TeX
#import "@preview/primaviz:0.7.0"
// For Australian English (en-AU):
#set text(lang: "en", region: "au")
// --------------------------------------
#import "@preview/marginalia:0.3.1": *



#let affls = (
  uni: (
    institution: "Università degli studi di Padova",),
)

#let authors = (
  (name: "Tharen Emmanuel Candi", affl: ( ), email: "tharenemmanuel.candi@studenti.unipd.it"),
  (name: "Giuseppe Castellana", affl: (), email: "giuseppe.castellana@studenti.unipd.it"),
)



#show: cvpr_report.with(
  renderer: "distill",
  title: [The Limits of Cross-Lingual Transfer: Evaluating SignCLIP on LIS],
  authors: (authors, affls),
  keywords: (),
  abstract: [ 
  We present the first study of the multilingual transfer ability of SignCLIP for Italian Sign Language (LIS) across zero-shot, few-shot, and fine-tuning. We find that its pretraining induces negative zero-shot transfer. In contrast, few-shot results confirm robust sign embeddings. We find monolingual fine-tuning highly effective on small datasets, achieving top results with Global Noise-Contrastive Estimation (GlobalNCE) and parameter-efficient ProLIP, compared to InfoNCE.
],
  //bibliography: bibliography("main.bib"),
  accepted: none,
  id: none,
)



= Introduction <sec:intro>
Sign Languages are the primary means of communication for millions of deaf individuals worldwide @Jiang24 @micieli25. Isolated Sign Language Recognition (ISLR) remains an open research area at the intersection of computer vision (CV) and natural language processing (NLP) @Jiang24 @bohacek2023fewshot. Similar to spoken language research, there is a large discrepancy in the efficacy of state-of-the-art solutions between high-resource and low-resource languages. Unlike the widely studied American (ASL), British (BSL) and Chinese Sign Language (CSL), Italian Sign Language (LIS) remains under-resourced, lacking a large-scale, annotated corpora required for the training of a deep neural network that can recognise it effectively @micieli25.

To overcome this limitation, recent research has pivoted toward transfer learning and few-shot recognition, leveraging models pre-trained on large multilingual datasets @bilge2024crosslingual. One of these models is SignCLIP, which utilises contrastive learning to project spoken language text and sign language videos into a shared embedding space. It is pre-trained on Spreadthesign, a dataset containing approximately 500,000 video clips in up to 44 different sign languages @Jiang24, including LIS. However, downstream evaluations for LIS recognition and text-video retrieval tasks were entirely omitted in its original benchmarks @Jiang24. 

Consequently, the applicability of these multi-lingual priors to low-resource LIS datasets remains unexamined. To address this research gap, we present the first investigation for LIS that explores the performance of zero-shot, few-shot, and fine-tuning paradigms using SignCLIP as a foundation model, evaluating both cross-modal Video-Text retrieval and ISLR. For this evaluation, we used two datasets: A3LIS-147, introduced in @fagiani2012a3lis, and SignIT @micieli25. These datasets enable complementary evaluations by contrasting a controlled, balanced multi-signer environment with domain-specific vocabulary against naturalistic, unbalanced, core-vocabulary signs, respectively.


Our main findings are as follows:

- *Empirical benchmarking of SignCLIP for LIS*: Our results suggest that aligning diverse multilingual signs to a monolingual text space induces negative transfer. While cross-lingual iconicity aids specific vocabulary subsets, performance stratification indicates that cross-lingual semantic divergence in sign motion primitives acts as a bottleneck during training. Nonetheless, highly competitive few-shot results confirm that the backbone successfully preserves discriminative spatial embeddings.
- *Ablation of fine-tuning strategies:* Through extensive optimisation ablations, we demonstrate that a Global Noise-Contrastive Estimation (GlobalNCE) objective yields superior adaptation for low-resource sign languages. By maximising hard-negative density within the global contrastive space, GlobalNCE most effectively organises the embedding space.


= Related works

== Italian sign language recognition

LIS ISLR research has largely targeted small-scale, controlled settings, utilising the A3LIS-147 dataset as the primary benchmark @marin2024listudio @marchisio2023tglis. Early approaches with Hidden Markov Models (HMMs) have been improved upon by more recent work reaching an accuracy of 80.4% with fully-supervised CNN models (Inception3D and SlowFast) @marin2024listudio.

The above work suffers from several structural limitations in the context of scalable SLR. The architectures employed are incapable of adapting to out-of-dictionary vocabulary without retraining. Furthermore, they are optimised for clean artifact-free datasets, potentially suffering in performance with out-of-distribution noisy data seen during real-world deployment @vandendriessche2024oneshot.

To address the latter, the SignIT dataset was recently introduced to benchmark LIS ISLR on real-world data. Baseline evaluations of the SignIT dataset demonstrate that current state-of-the-art approaches struggle to effectively classify LIS signs at the gloss level, as opposed to the categorical level @micieli25 #footnote[SignIT is not the only new dataset, for example, MultiMedaLIS, which explores multimodal inputs, and TGLIS-227, which collects continuous data from RAI television newscasts. @caligiore2024multisource @marchisio2023tglis. LIS is also present in multi-lingual datasets,i.e Spreadthesign.].


== Zero-shot, few-shot and cross-lingual recognition

Early Zero-Shot SLR attempts struggled due to cross-lingual complexities between signs and natural language, as well as high variation in sign execution @bilge2019zeroshot @rastgoo2021multimodal, resulting in a pivot towards few-shot, visual retrieval paradigms @bilge2024crosslingual.

Bilge et al. introduced Few-Shot Sign Language Recognition (FSSLR) via a meta-learning framework across sign languages, proving sparse source examples can generalise to unseen target languages. They discovered "synonym" subsets between languages failed to yield higher performance, suggesting signs are heavily diversified rather than net-iconic @bilge2024crosslingual. 

Similarly, Vandendriessche et al. (2025) embedded pose key points for distance-based visual retrieval, enabling one-shot ISLR that generalises to out-of-domain vocabularies without any retraining. Both frameworks operate entirely within a visual domain; achieving high cross-lingual transferability, but lack any inherent coupling to natural language text or semantic meaning @vandendriessche2024oneshot.

In contrast, Cheng et al. utilise contrastive learning in CiCo to model retrieval as a cross-lingual problem, successfully aligning a single sign language video modality directly to a spoken language text space (e.g., ASL to English). It trains a domain-agnostic sign encoder before the domain-aware retrieval. @cheng2023cico. 


== SignCLIP, multilingual corpora, and multilingual sign language

SignCLIP aligns multilingual signs to a single text space in English (as a matter of efficiency). Their work relies on the 'Iconicity Hypothesis' - that universal motion primitives are semantically shared across sign languages, and adapts the distributional hypothesis to sign language. The model captures the core meaning of a sign as a 'cluster centre' in the embedding space, preserving the individual variance of different signers. However, mapping these clusters is made difficult by the Spreadthesign corpus, which is skewed to only one video per sign per language @Jiang24.

SignCLIP uses cross-lingual contrastive learning with prefixed language identifying tokens, e.g `‘<en> <ase> {word}’` for ASL. Ultimately, the authors note that the model's zero-shot performance on out-of-domain data is deficient, and they posit that few-shot learning or fine-tuning is necessary to achieve noticeable performance @Jiang24.

The authors do not investigate the underlying architectural or semantic mechanisms that cause this failure, leaving the specific limitations of their cross-modal alignment unexamined. #footnote[SignCLIP has also been used for text-alignment for continuous sign language processing, showing notable improvements with language-specific ISLR fine-tuning @jiang2024sea. Hence, our work is illuminating for future work concerning the alignment of continuous LIS corpora. i.e., TGLIS-227.]



= Datasets
==  A3LIS-147 characteristics
- *Scale and composition*: The dataset comprises exactly 1,470 (1490) video sequences representing 147 (+2) distinct, isolated LIS signs.
- *Semantic categories*: The vocabulary is distributed across six daily-life domains where automatic tools for social inclusion of deaf people could be effectively applied: Public Institute (39), Railway Station (35), Hospital (19), Highway (8), Common Life (16), and Education (30). There are 2 extra signs corresponding to the silence pose and the sign for “lingua dei segni”. We did not get access to the exact category matching for the vocabulary.
- *Participants*: Recordings were performed by 10 different signers (7 males, 3 females), each executing every sign in the vocabulary exactly once.
- *Recording environment*: Data collection was conducted in a highly controlled laboratory setting featuring a fixed frontal camera, a green-screen background, and uniform diffuse lighting provided by two 400W spotlights @fagiani2012a3lis.


== SignIT characteristics

- *Scale and composition*: The dataset comprises 644 video sequences representing 94 distinct, isolated LIS signs. The total collection spans approximately 3 and a half hours and contains around 99,000 annotated frames.
- *Semantic categories*: The vocabulary focuses on fundamental concepts that are typically taught first to sign language learners. It is divided into five macro-categories: Animals (32), Food (20), Family (20), Emotions (5), and Colours (17).
- *Distribution*: The dataset is long-tailed, with mean examples per class of 3.34 (train), 2.00 (test), and 1.50 (validation). The mode is 2 (train), 1 (test), and 1 (validation). 45 signers contribute unevenly.
- *Recording environment*: In contrast to the highly controlled green-screen setting of A3LIS-147, SignIT was collected from publicly available online sources, including YouTube user profiles, providing a wide variety of backgrounds and lighting conditions. Video resolutions range from 426x240 to 1024x1024 pixels at 24 to 30 frames per second.
- *Preprocessing details*: Some videos contained background text of gloss labels; to ensure models learn from the signer’s movements rather than background information, the authors applied a blur to those areas #footnote[Our own visual analysis of the dataset reveals that some background text and images corresponding to the signs' meanings remain unblurred (e.g., a sad face drawing for an emotion sign). We believe this could favour models that use raw-video or frame data, including the LLaVA-OneVision model employed by the authors] @micieli25.


== Preprocessing
We follow the same pipeline used for the training of the frozen SignCLIP backbone @Jiang24, including:

- *3D keypoint extraction*: Trimmed RGB videos are processed through the MediaPipe Holistic framework to extract 3D spatial coordinates (x, y, z) of the signers per frame.
- *Pose preprocessing*: MediaPipe’s output is filtered to align with the SignCLIP video encoder. Lower-body keypoints are discarded, and the face mesh is simplified to core contours, reducing the feature space to 203 relevant keypoints per frame.
- *Sequence normalisation*: For signer-invariant representations, we spatially normalise the skeleton by shifting the shoulder midpoint to the origin and scaling all coordinates by the shoulder width. 
- *Truncation and padding*: The resulting sequence is truncated or zero-padded to a fixed context length of 256 frames to match the required input dimensions of the SignCLIP architecture. For A3LIS-147, 10 videos were truncated, with the longest one being 269 frames. For SignIT, 39 videos were truncated, the longest being 739 frames. Whilst this is not ideal, we notice that the longer videos of SignIT are typically due to a repetition of the sign motion.



= Methodology
We investigate whether SignCLIP's multilingual pretraining generalises to LIS, a language present in the pretraining corpus but excluded from the original evaluation. Our approach tests this through three phases: Zero-shot evaluation, to assess the frozen multilingual prior's native LIS structure; few-shot adaptation, to evaluate whether this structure supports recognition from a minimal number of examples using a frozen backbone; and a  fine-tuning ablation, to determine the performance ceiling of lightweight fine-tuning whilst preserving cross-modal alignment.

== Dual-dataset evaluation
A3LIS-147 and SignIT together evaluate model transfer across three axes:
- *Vocabulary coverage*: A3LIS-147 tests institutional language, which likely does not benefit from sign iconicity, and tests out-of-vocabulary (OOV) transfer with 31% of its signs. SignIT assesses in-vocabulary transfer with 95.6% coverage on foundational vocabulary, most likely to benefit from sign-iconicity. 
- *Acquisition conditions*: A3LIS-147 conditions are likely to provide clean keypoint extraction matching the pretrained conditions. SignIT is likely to have more variation, thereby assessing robustness to pose estimation noise, scaling, and signer variance encountered in real-world deployments.
- *Sample distribution*: A3LIS-147 evaluates optimisation under balanced conditions, while SignIT approximates more difficult one-to-three-shot scenarios.

== Zero-shot evaluation
The zero-shot evaluation applies the frozen SignCLIP checkpoint directly to both datasets. Predictions are generated by computing the cosine similarity between the video embedding and the text embedding of the English gloss (prompted as `<en> <lis> [gloss]`).

We report Recall\@1, 5, 10, and Median Rank. To better investigate the "Iconicity Hypothesis" and transfer ability, we perform a per-class analysis stratified by Category, Median Rank, qualitative ASL/BSL similarity (iconicity proxy), and Spreadthesign presence.

#heading3([Translation], [We manually translated A3LIS-147 using Spreadthesign. Remaining out-of-vocabulary (OOV) terms were translated as accurately as possible. We also recreated the unavailable categories. Both are listed in @app:dataset.])



== Few-shot evaluation

We evaluate few-shot ISLR to determine whether the solid results reported in the SignCLIP paper generalise to LIS.

- *Linear probing:* Following the original SignCLIP protocol @Jiang24, we train a scikit-learn linear classifier with 100 iterations on embeddings from the frozen backbone to evaluate the linear separability of the embedding space.
- *Prototypical retrieval:* The original SignCLIP few-shot evaluation also uses a $k$-NN classifier ($K = floor(sqrt(N)) $), but yields poor performance across benchmarks @Jiang24. We instead adopt the Prototypical Network framework @Snell. By computing a mean prototype for each class from its support embeddings and performing retrieval via nearest distance, we mitigate the impact of outliers. Unlike linear probing, this non-parametric approach naturally supports open-vocabulary classification without requiring retraining.

== Fine-tuning and loss function ablation on A3LIS-147

We initialise from the baseline checkpoint and fine-tune on A3LIS-147 using a 70/10/20 signer-stratified split. This ensures that our evaluation measures generalisation to unseen signers (see @app:dataset for the exact partition). Each configuration is trained for 50 epochs and evaluated across zero-shot retrieval, linear probing, and prototypical retrieval. For all the details about the hyperparameters used, see @app:finetune-configs. 

The text Transformer $f_(theta_t)$ and CNN backbone $f_(theta_"CNN")$ are frozen to preserve pre-trained semantic anchors. We unfreeze the visual adaptation parameters $Theta_"adapt" = {theta_"MLP", theta_v, tau}$, denoting the video token MLP, video Transformer encoder, and logit-scale temperature, respectively.

Because contrastive models exhibit high sensitivity to objectives and batch scales on low-resource datasets, we conduct an ablation evaluating the following optimisation regimes:

- *InfoNCE:* Used during SignCLIP's pretraining. It treats all other cross-modal pairs within the current batch as negatives @oord_2018_infoNCE.
- *Supervised Contrastive (SupCon):* Pulls together all same-class samples in the batch, whilst pushing apart samples from different classes @supcon_khosla2021.
- *Cross-Entropy (CE):* Focuses strictly on class discriminability over cross-modal alignment.
- *Decoupled Hard Negative Noise Contrastive Estimation (DHN-NCE):* Alleviates the negative-positive coupling effect that suppresses gradients from hard negatives when positive pairs are already well-aligned @koleilat2024medclipsambridgingtextimage.
- *Global Noise-Contrastive Estimation (GlobalNCE):* Draws negative instances from a noise distribution over the full dataset, effectively decoupling negative sampling from the mini-batch size @globalNCE.
- *ProLIP:* A parameter-efficient regime that isolates adaptation to a minimal subspace by only unfreezing the final linear layer of the projection MLP $theta_"MLP"$ and the logit scale $tau$ @fahes2024prolip. Cross-entropy training is regularised by penalising the weight divergence between the fine-tuned and pre-trained projection matrices, adapting the model without destroying foundational cross-modal structures @fahes2024prolip.


== SignIT fine-tuning
The single best-performing fine-tuning regime identified on A3LIS-147 is applied to SignIT (details in @app:finetune-configs). To address the dataset's naturalistic acquisition and long-tailed distribution, we apply light spatial augmentation to preserve semantic meaning, and heavier temporal augmentation (aug_sigma_temporal: 0.25, aug_sigma_spatial: 0.15, aug_sigma_noise: 0.002, aug_p_flip: 0.0,aug_strength_max: 3.5). SignIT's richer macro-categories, and its previous literature motived additional experiments on category zero-shot and few-shot retrieval. For these experiments, we include recall, precision, and F1 alongside R\@1 for better comparison with the original authors.

= Experiments

#let wide-table(caption, body) = figure(
  kind: table,
  caption: caption,
  placement: top,
)[
  #set text(size: 9pt)
  #body
]

#let narrow-table(caption, body) = figure(
  kind: table,
  caption: caption,
)[
  #set text(size: 9pt) 
  #body
]

#set table(
  stroke: none,
  inset: 4pt,
)

#let top-rule(cols) = table.hline(start: 0, end: cols, stroke: 1.2pt)
#let mid-rule(cols) = table.hline(start: 0, end: cols, stroke: 0.6pt)
#let bottom-rule(cols) = table.hline(start: 0, end: cols, stroke: 1.2pt)

#let distill-figure-outset(content) = context {
  if target() == "html" {
    distill-outset(content)
  } else {
    content
  }
}

#let distill-figure-page(content) = context {
  if target() == "html" {
    distill-page(content)
  } else {
    content
  }
}

#let distill-figure-screen(content) = context {
  if target() == "html" {
    distill-screen(content)
  } else {
    content
  }
}

== Zero-shot complete-dataset evaluations

Baseline zero-shot evaluations in @tab:signit-zero-shot-category and @tab:a3lis-zero-shot-category show poor overall performance, in line with Jiang et al. findings for out-of-domain transfer @Jiang24. However, there is stratification between categories. In SignIT, the 'Food' domain achieves the highest exact retrieval (10.96% R\@1), while 'Emotions' demonstrates superior neighbourhood alignment (51.60% R\@10). Similarly, A3LIS-147 exhibits a split between early recall ('Common Life', 7.22% R\@1) and broader neighbourhood density ('Public Institute', MedR 42.5).  This variance indicates that while overall cross-lingual transfer is weak, the model successfully transfers universal, cross-lingual iconic primitives from the pretraining distribution for specific semantic clusters. For gloss-level and more category details, see @app:zero-shot-strat.

#narrow-table([SignIT zero-shot category results.])[
  #table(
    columns: (2fr, 1fr, 1fr, 1fr, 1fr),
    align: (col, row) => if col == 0 { left } else { center },
    top-rule(5),
    table.header(
      [Cat.],
      [R\@1],
      [R\@5],
      [R\@10],
      [MedR],
    ),
    mid-rule(5),

    [Animals], [0.041], [0.149], [0.3108], [23.6],
    [Colors], [0.0572], [0.2321], [0.4353], [18.2],
    [Emotions], [0.04], [0.3117], [*0.516*], [*13.2*],
    [Family], [0.0071], [0.0155], [0.0496], [42.1],
    [Food], [*0.1096*], [*0.3614*], [0.4947], [13.7],
    mid-rule(5),
    [Overall], [0.0506], [0.1876], [0.326], [24.0],

    bottom-rule(5),
  )
] <tab:signit-zero-shot-category>

#narrow-table([A3LIS zero-shot category results. ])[
  #table(
    columns: (2fr, 1fr, 1fr, 1fr, 1fr),
    align: (col, row) => if col == 0 { left } else { center },
    top-rule(5),
    table.header(
      [Cat.],
      [R\@1],
      [R\@5],
      [R\@10],
      [MedR],
    ),
    mid-rule(5),

    [Common Life], [*0.0722*], [*0.2167*], [*0.2722*], [48.9],
    [Education], [0.0433], [0.1233], [0.17], [61.0],
    [Highway], [0.025], [0.125], [0.25], [50.1],
    [Hospital], [0.0263], [0.0895], [0.1684], [46.0],
    [Public Institute], [0.0447], [0.1342], [0.2], [*42.5*],
    [Railway Station], [0.0083], [0.0333], [0.0833], [47.9],
    mid-rule(5),
    [Overall], [0.0356], [0.1114], [0.1732], [49.2],

    bottom-rule(5),
  )
] <tab:a3lis-zero-shot-category>


== Zero-shot medianK stratification, iconicity, and OOV analysis

#narrow-table([A3LIS-147 zero-shot tier stratification.])[
  #table(
    columns: (2fr, 2fr, 1.5fr, 1.5fr),
    align: (col, row) => if col == 0 { left } else { center },
    top-rule(4),
    table.header(
      [Tier],
      [MedR Range],
      [Portion],
      [Cum. MedR],
    ),
    mid-rule(4),

    [Great], [1–3], [0.537], [1.6],
    [Good], [3.1–15], [0.1678], [7.2],
    [Fair], [15.1–40], [0.2282], [18.4],
    [Neutral], [40.1–74], [0.3087], [33.0],
    [Adverse], [74.1–148], [0.2416], [49.2],

    bottom-rule(4),
  )
] <tab:a3lis-zero-shot-tiers>

#narrow-table([SignIT zero-shot tier stratification.])[
  #table(
    columns: (2fr, 2fr, 1.5fr, 1.5fr),
    align: (col, row) => if col == 0 { left } else { center },
    top-rule(5),
    table.header(
      [Tier],
      [MedR Range],
      [Portion],
      [Cum.  MedR],
    ),
    mid-rule(5),

    [Great], [1–3], [0.43], [1.9],
    [Good], [3.1–10], [0.226], [5.6],
    [Fair], [10.1–25], [0.366], [12.3],
    [Neutral], [25.1–47], [0.226], [18.5],
    [Adverse], [47.1–93], [0.140], [24.0],

    bottom-rule(5),
  )
] <tab:signit-zero-shot-tiers>

#narrow-table([A3LIS signs present in pretraining.])[
  #table(
    columns: (2fr, 1fr, 1fr, 1fr, 1fr),
    align: (col, row) => if col == 0 { left } else { center },
    top-rule(5),
    table.header(
      [LIS Sign in STS],
      [R\@1],
      [R\@5],
      [R\@10],
      [MedR],
    ),
    mid-rule(5),

    [No], [*0.0413*], [0.1109], [0.1543], [51.1],
    [Yes], [0.0337], [*0.1169*], [*0.1888*], [48.4],
    [Yes, but different], [0.0286], [0.0786], [0.1357], [*47.9*],

    bottom-rule(5),
  )
] <tab:a3lis-pretraining-presence>



#narrow-table([A3LIS iconicity proxy results.])[
  #table(
    columns: (3.3fr, 1fr, 1fr, 1fr, 1fr),
    align: (col, row) => if col == 0 { left } else { center },
    top-rule(5),
    table.header(
      [Iconicity Proxy (UK/US)],
      [R\@1],
      [R\@5],
      [R\@10],
      [MedR],
    ),
    mid-rule(5),

    [Kind of], [0.0143], [0.0571], [0.1214], [60.1],
    [No], [0.026], [0.1135], [0.1698], [50.5],
    [Yes], [*0.0667*], [*0.1256*], [*0.2*], [*42.0*],

    bottom-rule(5),
  )
] <tab:a3lis-iconicity-proxy>

SignCLIP's cross-lingual alignment induces a structurally bimodal transfer effect. We argue that since the pre-trained text encoder operates in an English-centric semantic space, language prefix identifiers provide insufficient separation. Consequently, the objective forces visually disparate sign videos toward a quasi-singular text anchor. This semantic asymmetry creates an optimisation conflict that marginalises low-resource languages, resulting in negative transfer, evidenced by the adverse tiers in A3LIS-147 (24.16%) and SignIT (14.0%) in  @tab:a3lis-zero-shot-tiers and @tab:signit-zero-shot-tiers.

For iconic signs, the shared anchor is beneficial (achieving a MedR of 42.0); for non-iconic signs, the anchor provides a weak or adversarial signal, collapsing retrieval accuracy (MedR 60.1) in @tab:a3lis-iconicity-proxy. Pre-training exposure does not overcome this issue, @tab:a3lis-pretraining-presence shows OOV LIS signs marginally outperform in-vocabulary signs at R\@1 (4.13% vs. 3.37%). 

We believe data scaling is unlikely to resolve these failures. Shared human articulatory constraints result in heavy overlap in the discriminative features between languages, a problem further complicated by high individual signer-variance (@fig:variance-profile in Appendix). Thus, diversification within synonym classes @bilge2024crosslingual and cross-lingual "false friends" lead to gradient conflicts. Our findings suggest these factors limit zero-shot performance for any architecture imposing a single joint embedding space without language-gated alignment. These issues can be resolved by monolingual fine-tuning (@tab:signit-finetuning-fewshot), likely at the  expense of multilingual understanding, but this remains unexamined.

Linear probing on the frozen backbone achieves 66.78% R\@1 on A3LIS-147 in @tab:a3lis-finetuning-ablation-condensed, confirming that the video encoder learns robust representations.  

== A3LIS fine-tuning ablation

@tab:a3lis-finetuning-ablation-condensed shows that GlobalNCE yields the strongest fine-tuning performance on A3LIS and the linear-probe matches previous SOTA @marin2024listudio. We attribute this to its global negative sampling across distributed batches, providing the critical density of hard negatives required to stabilise contrastive gradients. ProLIP achieves within 0.3% R\@1 of GlobalNCE at zero-shot (75.84% vs. 76.17%) while adapting only the final MLP layer and logit scale, making it the preferred regime when compute or overfitting risk is the primary concern.

#narrow-table([A3LIS fine-tuning ablation (Condensed).See @app:finetune for the complete ablation over all optimisers.])[
  
  #table(
    columns: (1.7fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (col, row) => if col < 2 { left } else { center },
    top-rule(6),
    table.header(
      table.cell(colspan: 2)[Method],
      [R\@1],
      [R\@5],
      [R\@10],
      [MedR],
    ),
    mid-rule(6),

    table.cell(rowspan: 3)[Baseline], [Zero], [0.0369], [0.1309], [0.1946], [40],
    [Proto], [0.6477], [0.9094], [0.9698], [*1*],
    [LP], [0.6678], [0.9329], [0.9698], [*1*],
    mid-rule(6),

    table.cell(rowspan: 3)[GlobalNCE16], [Zero], [*0.7617*], [*0.9262*], [*0.9698*], [*1*],
    [Proto], [*0.7886*],	[0.9430],[0.9732],	[*1*],
    [LP], [*0.8020*], [0.9430], [0.9732], [*1*],
    mid-rule(6),

    table.cell(rowspan: 3)[PLIP16], [Zero], [0.7584], [0.9161], [0.953], [*1*],
    [Proto], [0.7718], [0.9396], [0.9597], [*1*],
    [LP], [0.7785], [0.9364], [0.9564], [*1*],

    bottom-rule(6),
  )
] <tab:a3lis-finetuning-ablation-condensed>

== SignIT few-shot and fine-tuning ablation
@tab:signit-finetuning-fewshot shows that augmentation of SignIT improves generalisation. Our results trail the LLaVA-OneVision (Acc 0.238 video+pose) of the SignIT authors @micieli25.  We outperform all non-video baselines they evaluated, including pose-only LLaVA (Acc 0.121), establishing a competitive key point-only result.

#narrow-table([SignIT fine-tuning and few-shot ablation.])[
  #table(
    columns: (1.7fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (col, row) => if col < 2 { left } else { center },
    top-rule(6),
    table.header(
      [Model],
      [Mode],
      [R\@1],
      [R\@5],
      [R\@10],
      [MedR],
    ),
    mid-rule(6),

    table.cell(rowspan: 3)[Baseline], [Zero], [0.0359], [0.1692], [0.2769], [22.0],
    [Proto], [0.0974], [0.3077], [0.4462], [13.0],
    [LP], [0.0923], [0.3641], [0.5333], [10.0],
    mid-rule(6),

    table.cell(rowspan: 3)[Fine-tune], [Zero], [0.1385], [*0.4308*], [0.5538], [*7.0*],
    [Proto], [0.1487], [*0.4256*], [0.5692], [*8.0*],
    [LP], [0.1538], [0.4308], [0.5744], [*7.0*],
    mid-rule(6),

    table.cell(rowspan: 3)[Fine-tune + Aug ], [Zero], [*0.1436*], [*0.4308*], [*0.6154*], [8.0],
    [Proto], [*0.1744*], [0.4103], [*0.6103*], [*8.0*],
    [LP], [*0.1744*], [*0.4462*], [*0.5897*], [8.0],

    bottom-rule(6),
  )
] <tab:signit-finetuning-fewshot>



== SignIT macro-category retrieval

Zero-shot on categories achieves an F1-score (0.48) that is competitive with some fully supervised video baselines, such as I3D (0.34 F1)@micieli25. Because this relies on measuring the distance between visual embeddings and the textual embeddings of broad macro-categories, these results highlight an advantage of contrastive pretraining: the latent space is semantically organised, allowing the model to generalise to categorical distributions it never explicitly encountered during pretraining. Our strongest few-shot linear-probe configuration reaches 64.62% R\@1, approaching the performance of SignIT's best fully supervised MLP (0.726 Accuracy) @micieli25.

#narrow-table([SignIT macro-category retrieval, metrics to match original authors] )[
  #table(
    columns: (1.7fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (col, row) => if col < 2 { left } else { center },
    top-rule(6),
    table.header(
      [Model],
      [Mode],
      [R\@1],
      [Pr],
      [Re],
      [F1],
    ),
    mid-rule(6),

    table.cell(rowspan: 3)[Baseline], [Zero], [0.3744], [*0.54*], [0.34], [0.30],
    [Proto], [0.4103], [0.3909], [0.3921], [0.3844],
    [LP], [0.5846], [0.61], [0.52], [0.55],
    mid-rule(6),

    table.cell(rowspan: 3)[Fine-tune], [Zero], [0.4872], [0.48], [*0.55*], [*0.48*],
    [Proto], [0.5641], [0.5219], [0.5371], [0.5251],
    [LP], [*0.6462*], [0.64], [*0.59*], [*0.61*],
    mid-rule(6),

    table.cell(rowspan: 3)[Fine-tune + Aug], [Zero], [*0.4974*], [0.49], [0.52], [*0.48*],
    [Proto], [*0.5949*], [*0.5561*], [*0.5708*], [*0.5503*], 
    [LP], [0.6103], [*0.68*], [0.57], [0.59],
    

    bottom-rule(6),
  )
] <tab:signit-macro-category>

== Sign language identification
#narrow-table([Sign language identification results.])[
  #table(
    columns: (1fr, 1fr, 1fr, 1fr),
    align: center,
    top-rule(4),
    table.header(
      [Random Chance],
      [R\@1],
      [R\@2],
      [MedR],
    ),
    mid-rule(4),

    [0.1250], [0.3510], [0.6523], [2.0 / 8],

    table.hline(start: 0, end: 4, stroke: 1.2pt),
    table.cell(colspan: 4, align: left)[
      #v(2pt)
      #text(size: 8pt)[
        *False positives:* lsf - 20, bsl - 688, ngt - 227, and lse - 32.
      ]
    ]
  )
] <tab:sign-language-identification>


The Sign language identification of @tab:sign-language-identification complicates our earlier finding that in-vocabulary LIS signs do not outperform OOV. This simplified retrieval task suggests that SignCLIP does learn some language separation, as shown by the R\@2 (65.23%). However, performance drops sharply at R\@1 (35.10%), with substantial confusion between LIS, BSL, and NGT (@app:sil). It may be worth investigating if this is due to higher inter-language iconicity.

= Conclusion
This work demonstrates that SignCLIP's contrastive alignment induces a structurally bimodal transfer effect on LIS, beneficial for iconic vocabulary, adverse for non-iconic signs, indicating a geometric limitation of the shared embedding space paradigm rather than a data-scaling problem. Few-shot and fine-tuning strategies mitigate these limitations, confirming that the video encoder learns discriminative representations that zero-shot retrieval cannot exploit without fine-tuning in a monolingual context.

We see two promising directions for future research. Since pretraining exposure to LIS signs does not guarantee positive transfer, fine-tuning on the LIS-specific Spreadthesign subset could be adequate for OOD LIS. A more effective multilingual embedding space requires language-conditioned projections that both allow for iconicity transfer and decouple text anchors for non-iconic glosses across sign languages.





// 1. Insert your bibliography first
#bibliography("main.bib", title: "References")
// 1. Reset the heading counter to 0 (so the next heading becomes 1 -> A)
#counter(heading).update(0)

// 2. Tell Typst to use letters for headings and label references as "Appendix"
#set heading(numbering: "A.1", supplement: [Appendix])

// 3. (Optional) Make the document headings explicitly read "Appendix A" instead of just "A"
#show heading.where(level: 1): it => {
  block(width: 100%, below: 1em)[
    Appendix #counter(heading).display(it.numbering) #h(0.5em) #it.body
  ]
}
// #show: appendix

= Extended evaluation <app:eval-cont>


== Leave-one-signer-out baseline linear-probe

#let a3lis-data = (
  (name: "R@1", mean: 0.7148, sd: 0.0631, c: rgb("E67E22")), // Orange
  (name: "R@5",         mean: 0.9403, sd: 0.0276, c: rgb("27AE60")), // Green
  (name: "R@10",        mean: 0.9725, sd: 0.0176, c: rgb("8E44AD")), // Purple
)
#figure(
  caption: [Visualisation for Leave-one-signer-out evaluation on A3LIS on frozen baseline with linear-probe. Note: Median Rank (MedR) is excluded from the visualised profile as it achieved a stable 1.00 ± 0.00.],
  context {
    if target() == "html" {
      distill-figure-outset([
      #table(
        columns: (1fr, 1fr, 1fr),
        align: (left, center, center),
        table.header([Metric], [Mean], [Std. Dev.]),
        [R\@1], [0.7148], [0.0631],
        [R\@5], [0.9403], [0.0276],
        [R\@10], [0.9725], [0.0176],
      )
    ])
  } else {
    distill-figure-outset([
      #layout(bounds => {
      let left-margin = 90pt
      let right-margin = 20pt
      
      let available-width = bounds.width - left-margin - right-margin
      
      let x-min = 0.6
      let x-max = 1.05
      let range = x-max - x-min
      
      let scale(x) = (x - x-min) / range * available-width

      block(width: bounds.width, height: 140pt, inset: 10pt)[
        
        // 1. Draw X-axis and tick marks
        #place(top + left, dx: left-margin, dy: 110pt)[
          #line(length: available-width, stroke: 1pt + luma(150))
          #for tick in (0.6, 0.7, 0.8, 0.9, 1.0) {
            place(top + left, dx: scale(tick), dy: 0pt)[
              #line(length: 4pt, angle: 90deg, stroke: 1pt + luma(150))
              #place(dx: -8pt, dy: 6pt)[
                #text(size: 8pt, fill: luma(100))[#tick]
              ]
            ]
          }
        ]

        // 2. Draw Data Tracks
        #for (i, row) in a3lis-data.enumerate() {
          let y-pos = i * 35pt + 20pt
          
          // Y-axis Expanded Label
          place(top + left, dx: 0pt, dy: y-pos - 4pt)[
            #text(weight: "bold", size: 9pt, fill: row.c.darken(10%))[#row.name]
          ]
          
          // Faint background track line
          place(top + left, dx: left-margin, dy: y-pos)[
            #line(length: available-width, stroke: (paint: luma(230), thickness: 0.5pt, dash: "dashed"))
          ]
          
          let cx = scale(row.mean)
          
          // 3. Draw Variance Halo
          if row.sd > 0 {
            let sd-w = (row.sd / range) * available-width
            let halo-width = sd-w * 2
            
            let halo-grad = gradient.linear(
              row.c.lighten(85%), 
              row.c.lighten(30%), 
              row.c.lighten(85%)
            )
            
            // Shaded expansion box
            place(top + left, dx: left-margin + cx - sd-w, dy: y-pos - 6pt)[
              #box(
                width: halo-width, 
                height: 12pt, 
                fill: halo-grad, 
                radius: 6pt
              )
            ]
            
            // SD terminal bound markers
            place(top + left, dx: left-margin + cx - sd-w, dy: y-pos - 4pt)[
              #line(length: 8pt, angle: 90deg, stroke: 1.5pt + row.c)
            ]
            place(top + left, dx: left-margin + cx + sd-w, dy: y-pos - 4pt)[
              #line(length: 8pt, angle: 90deg, stroke: 1.5pt + row.c)
            ]
          }
          
          // 4. Mean Anchor (Central point)
          place(top + left, dx: left-margin + cx - 3pt, dy: y-pos - 3pt)[
            #circle(radius: 3pt, fill: row.c.darken(30%))
          ]
          
          // 5. Mean ± SD Text Label (Centered dynamically)
          let label-width = 60pt
          place(top + left, dx: left-margin + cx - (label-width / 2), dy: y-pos + 6pt)[
            #box(width: label-width, )[
              #text(size: 7pt, fill: row.c.darken(30%), weight: "semibold")[
                #row.mean ± #row.sd
              ]
            ]
          ]
        }
      ]
      })
    ])
    }
  }
) <fig:variance-profile>

Signer variability presented in @fig:variance-profile primarily degrades R\@1, seen by its $plus.minus$6.3% standard deviation. Broader retrieval remains robust. This variance underscores cross-signer generalisation as a persistent difficulty.

== Complete A3LIS fine-tuning ablation <app:finetune>

In Section 4.5, we presented a condensed view of our A3LIS-147 fine-tuning ablation, highlighting the performance of the default SignCLIP objective (NCE) against our best-performing GlobalNCE regime. @tab:a3lis-finetuning-ablation-full presents the comprehensive results across all evaluated loss functions, batch sizes, and sampling strategies. 

#narrow-table([Complete A3LIS fine-tuning ablation. Baseline = frozen SignCLIP checkpoint; Zero = zero-shot retrieval; Proto = prototype retrieval; LP = linear-probe.])[
  #table(
    columns: (2fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (col, row) => if col < 2 { left } else { center },
    top-rule(6),
    table.header(
      table.cell(colspan: 2)[Method],
      [R\@1],
      [R\@5],
      [R\@10],
      [MedR],
    ),
    mid-rule(6),

    table.cell(rowspan: 3)[Baseline], [Zero], [0.0369], [0.1309], [0.1946], [40],
    [Proto], [0.6477], [0.9094], [0.9698], [*1*],
    [LP], [0.6678], [0.9329], [0.9698], [*1*],
    mid-rule(6),

    table.cell(rowspan: 3)[InfoNCE128], [Zero], [0.7248], [0.906], [0.9396], [*1*],
    [Proto], [0.7584], [0.9430], [0.9765], [*1*],
    [LP], [0.7617], [*0.9597*], [*0.9799*], [*1*],
    mid-rule(6),

    table.cell(rowspan: 3)[SupCon32x4], [Zero], [0.5912], [0.8591], [0.9128], [*1*],
    [Proto], [0.7013], [0.9128], [0.9664], [*1*],
    [LP], [0.7785], [0.9396], [0.9765], [*1*],
    mid-rule(6),

    table.cell(rowspan: 3)[Cross-Entropy 16], [Zero], [0.0503], [0.1611], [0.245], [33],
    [Proto], [0.772], [0.946], [*0.987*], [*1*],
    [LP], [0.7651], [0.9463], [*0.9799*], [*1*],
    mid-rule(6),

    table.cell(rowspan: 3)[GlobalNCE 16], [Zero], [*0.7617*], [*0.9262*], [*0.9698*], [*1*],
    [Proto], [*0.7886*], [0.9430], [0.9732], [*1*],
    [LP], [*0.802*], [0.943], [0.9732], [*1*],
    mid-rule(6),

    table.cell(rowspan: 3)[ProLIP 16], [Zero], [0.7584], [0.9161], [0.953], [*1*],
    [Proto], [0.7718], [0.9396], [0.9597], [*1*],
    [LP], [0.7785], [0.9364], [0.9564], [*1*],
    mid-rule(6),

    table.cell(rowspan: 3)[DHN-NCE 64], [Zero], [0.7081], [0.8926], [0.9295], [*1*],
    [Proto], [0.7651], [*0.9497*], [0.9732], [*1*],
    [LP], [0.7617], [0.9564], [0.9765], [*1*],

    bottom-rule(6),
  )
] <tab:a3lis-finetuning-ablation-full>


== Sign language identification scores <app:sil>

#narrow-table([Sign language identification languages and guesses for A3LIS])[
  #table(
    columns: (1fr, 1fr, 1fr),
    align: center,
    top-rule(4),
    table.header(
      [Target language],
      [Count],
      [Proportion],
    ),
    mid-rule(4),

    [`<en> <lis>`], [523], [0.351],
    [`<en> <ase>`], [0], [0],
    [`<en> <dgs>`], [0], [0],
    [`<en> <lsf>`], [20], [0.0134],
    [`<en> <bsl>`], [688], [0.4618],
    [`<en> <ngt>`], [227], [0.1523],
    [`<en> <lse>`], [32], [0.0215],
    [`<en> <csl>`], [0], [0],

    bottom-rule(6),

  )
]


=  Fine-tuning configurations <app:finetune-configs>

== SignIT with augmentation fine-tuning hyperparameters


#v(8pt)

#table(
  columns: (1.5fr, 2fr),
  align: (left, left),
  stroke: none,
  fill: (col, row) => if calc.even(row) { luma(245) } else { none },
  inset: 7pt,
  
  table.hline(stroke: 1.5pt),
  table.header([*Parameter*], [*Value*]),
  table.hline(stroke: 1pt),
  [Base Checkpoint],[signclip_v1_1],
  [Model Architecture], [`MMFusionSeparate`],
  [Video Encoder], [`MMBertForEncoder` (12 layers, dim: 609)],
  [Text Encoder], [`BertModel` (`bert-base-cased`)],
  [Loss Function], [`GlobalNCE`],
  [Optimiser], [Adam ($beta_1=0.9, beta_2=0.98$)],
  [Base Learning Rate], [5.0e-05],
  [LR Scheduler], [Polynomial Decay (122 warmup updates)],
  [Weight Decay], [0.02],
  [Gradient Clipping], [2.0 (Max Norm)],
  [Max Epochs], [50],
  [Batch Size], [16],
  [Precision], [FP16 Mixed Precision],
  [Max Sequence Length], [Video: 256 frames / Text: 64 tokens],
  [Pose Components], [`reduced_face`],
  [Data Augmentation], [Temporal ($sigma=0.25$), Spatial ($sigma=0.15$), Noise ($sigma=0.002$)],
  
  table.hline(stroke: 1.5pt)
)


== A3LIS and no augmentation fine-tuning hyperparameters
Note for ProLIP, there are two additional hyperparamters set: prolip_lambda: 0.5, and  prolip_lambda_mode: inv_n
  #v(8pt)

  #table(
    columns: (1.5fr, 2fr),
    align: (left, left),
    stroke: none,
    fill: (col, row) => if calc.even(row) { luma(245) } else { none },
    inset: 7pt,
    
    table.hline(stroke: 1.5pt),
    table.header([*Parameter*], [*Value*]),
    table.hline(stroke: 1pt),
    [Base Checkpoint],[signclip_v1_1],
    [Model Architecture], [`MMFusionSeparate`],
    [Video Encoder], [`MMBertForEncoder` (12 layers, dim: 609)],
    [Text Encoder], [`BertModel` (`bert-base-cased`)],
    [Loss Function], [(depends on experiment)],
    [Video SupCon Weight], [`0.5`],
    [Optimiser], [Adam ($beta_1=0.9, beta_2=0.98$)],
    [Base Learning Rate], [5.0e-05],
    [LR Scheduler], [Polynomial Decay (122 warmup updates)],
    [Weight Decay], [0.01],
    [Gradient Clipping], [2.0 (Max Norm)],
    [Max Epochs], [50],
    [Batch Size], [16],
    [Precision], [FP16 Mixed Precision],
    [Max Sequence Length], [Video: 256 frames / Text: 64 tokens],
    [Pose Components], [`reduced_face`],
    [Data Augmentation], [Temporal Augmentation Enabled],
    
    table.hline(stroke: 1.5pt)
  )


= Zero-shot stratification <app:zero-shot-strat>

== SignIT glosses by median rank <app:signit-gloss-strat>

#v(12pt)

 #context if target() == "html" [
  *1. Great (1-3):* bear, bread, color, watermelon.

  *2. Good (3.1-10):* anger, brown, cake, chocolate, cow, fear, fuchsia, giraffe, grey, joy, light colors, orange, pizza, relatives, rooster, salt, sheep, snail, tiger, vegetable, wine.

  *3. Fair (10.1-25):* apple, banana, bird, blue, butterfly, candy, cat, dark colors, disgust, donkey, family, fish, frog, fruit, grandfather, green, horse, light blue, lion, meat, monkey, parents, pasta, pear, pig, pineapple, pink, purple, rabbit, rice, spider, turtle, yellow, zebra.

  *4. Neutral / Random (25.1-47):* aunt, black, brother-in-law, bull, cousin, crocodile, dad, daughter-in-law, dog, elephant, goat, goose, grandmother, milk, parrot, red, sadness, sky blue, uncle, water, wolf.

  *5. Perverse (47.1-93):* boyfriend, brother, hen, husband, mom, mouse, nephew, sister, snake, son, son-in-law, white, wife.
] else [
 #stack(
  spacing: 14pt,
  
  // 1. Great
  block(fill: rgb("e8f5e9"), inset: 12pt, radius: 4pt, width: 100%, stroke: 1pt + rgb("2e7d32"))[
    *🟢 1. Great (1-3)* --- _Top performing examples (Count: 4)_ \
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#1b5e20"))[bear, bread, color, watermelon]
  ],
  
  // 2. Good
  block(fill: rgb("f1f8e9"), inset: 12pt, radius: 4pt, width: 100%, stroke: 1pt + rgb("4caf50"))[
    *🟢 2. Good (3.1-10)* --- _High performing examples (Count: 21)_ \
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#33691e"))[anger, brown, cake, chocolate, cow, fear, fuchsia, giraffe, grey, joy, light colors, orange, pizza, relatives, rooster, salt, sheep, snail, tiger, vegetable, wine]
  ],
  
  // 3. Fair
  block(fill: rgb("fff8e1"), inset: 12pt, radius: 4pt, width: 100%, stroke: 1pt + rgb("ffb300"))[
    *🟡 3. Fair (10.1-25)* --- _Average performing examples (Count: 34)_ \
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#ff6f00"))[apple, banana, bird, blue, butterfly, candy, cat, dark colors, disgust, donkey, family, fish, frog, fruit, grandfather, green, horse, light blue, lion, meat, monkey, parents, pasta, pear, pig, pineapple, pink, purple, rabbit, rice, spider, turtle, yellow, zebra]
  ],
  
  // 4. Neutral / Random
  block(fill: rgb("f5f5f5"), inset: 12pt, radius: 4pt, width: 100%, stroke: 1pt + rgb("757575"))[
    *⚪ 4. Neutral / Random (25.1-47)* --- _Random or neutral examples (Count: 21)_ \
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#424242"))[aunt, black, brother-in-law, bull, cousin, crocodile, dad, daughter-in-law, dog, elephant, goat, goose, grandmother, milk, parrot, red, sadness, sky blue, uncle, water, wolf]
  ],
  
  // 5. Perverse
  block(fill: rgb("ffebee"), inset: 12pt, radius: 4pt, width: 100%, stroke: 1pt + rgb("c62828"))[
    *🔴 5. Perverse (47.1-93)* --- _Poor performing examples (Count: 13)_ \
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#b71c1c"))[boyfriend, brother, hen, husband, mom, mouse, nephew, sister, snake, son, son-in-law, white, wife]
  ]
)
]


== A3LIS-147 glosses by median rank <app:a3lis-gloss-strat>

 
#v(12pt)

 #context if target() == "html" [
  *1. Great (1-3):* caldo, data, falconara, freddo, giudizio, iniezione, scadenza, senigallia.

  *2. Good (3.1-15):* abitare, affitto, ancona, aperto, avviso, consegnare, dirigente, dolore, emergenza, jesi, macerata, modello, modulo, multa, notte, pomeriggio, presente, pubblica, ritirare_il_numero, sciopero, sostegno, traffico, tratta, vacanze, verde.

  *3. Fair (15.1-40):* acqua, allegare, ambulanza, annullato, arrivo, ascoli, banca, binario, cambio, commissione, compilare, costo, cura, domenica, esame, fermo, giallo, giovedì, giorno, infermiere, infezione, istituto, marche, mattina, medico, operazione, partenza, promosso, provincia, ritardo, s.benedetto, tassa, torino, università.

  *4. Neutral / Random (40.1-74):* abbonamento, allergia, amministrazione, andata, andata_e_ritorno, assente, assistente_alla_comunicazione, bidello, biglietto, bocciato, casa, casello, chiuso, cibo, civitanova, comune, diploma, disinfettare, fano, venerdì, giorni, ieri, laurea, litro, lunedì, martedì, mercoledì, mesi, obliterare, ospedale, pesaro-urbino, posta, rallentamenti, regione, ricevuta, ritorno, roma, rosso, segretario, sera, sindaco, stazione, strada, treno.

  *5. Perverse (74.1-148):* asilo_nido, assessore, assistente, autostrada, domani, elementari, ente_pubblico, entro, flebo, impiegato, interprete, lingua_dei_segni, malattia, mangiare, marca_da_bollo, medie, nota, oggi, obliteratrice, orari, preside, professore, pronto_soccorso, registro, sabato, sala_d'attesa, scuola, scuola_materna, sil, superiori, sportello, studente, tecnico, telefono, ufficio_informazioni, voto.
] else [
 #stack(
  spacing: 14pt,
  
  // 1. Great
  block(fill: rgb("e8f5e9"), inset: 12pt, radius: 4pt, width: 100%, stroke: 1pt + rgb("2e7d32"))[
    *🟢 1. Great (1-3)* --- _Top performing examples (Count: 8)_ \
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#1b5e20"))[caldo, data, falconara, freddo, giudizio, iniezione, scadenza, senigallia]
  ],
  
  // 2. Good
  block(fill: rgb("f1f8e9"), inset: 12pt, radius: 4pt, width: 100%, stroke: 1pt + rgb("4caf50"))[
    *🟢 2. Good (3.1-15)* --- _High performing examples (Count: 25)_ \
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#33691e"))[abitare, affitto, ancona, aperto, avviso, consegnare, dirigente, dolore, emergenza, jesi, macerata, modello, modulo, multa, notte, pomeriggio, presente, pubblica, ritirare_il_numero, sciopero, sostegno, traffico, tratta, vacanze, verde]
  ],
  
  // 3. Fair
  block(fill: rgb("fff8e1"), inset: 12pt, radius: 4pt, width: 100%, stroke: 1pt + rgb("ffb300"))[
    *🟡 3. Fair (15.1-40)* --- _Average performing examples (Count: 34)_ \
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#ff6f00"))[acqua, allegare, ambulanza, annullato, arrivo, ascoli, banca, binario, cambio, commissione, compilare, costo, cura, domenica, esame, fermo, giallo, giovedì, giorno, infermiere, infezione, istituto, marche, mattina, medico, operazione, partenza, promosso, provincia, ritardo, s.benedetto, tassa, torino, università]
  ],
  
  // 4. Neutral / Random
  block(fill: rgb("f5f5f5"), inset: 12pt, radius: 4pt, width: 100%, stroke: 1pt + rgb("757575"))[
    *⚪ 4. Neutral / Random (40.1-74)* --- _Random or neutral examples (Count: 45)_ \
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#424242"))[abbonamento, allergia, amministrazione, andata, andata_e_ritorno, assente, assistente_alla_comunicazione, bidello, biglietto, bocciato, casa, casello, chiuso, cibo, civitanova, comune, diploma, disinfettare, fano, venerdì, giorni, ieri, laurea, litro, lunedì, martedì, mercoledì, mesi, obliterare, ospedale, pesaro-urbino, posta, rallentamenti, regione, ricevuta, ritorno, roma, rosso, segretario, sera, sindaco, stazione, strada, treno]
  ],
  
  // 5. Perverse
  block(fill: rgb("ffebee"), inset: 12pt, radius: 4pt, width: 100%, stroke: 1pt + rgb("c62828"))[
    *🔴 5. Perverse (74.1-148)* --- _Poor performing examples (Count: 36)_ \
    #v(4pt)
    #text(size: 9.5pt, fill: rgb("#b71c1c"))[asilo_nido, assessore, assistente, autostrada, domani, elementari, ente_pubblico, entro, flebo, impiegato, interprete, lingua_dei_segni, malattia, mangiare, marca_da_bollo, medie, nota, oggi, obliteratrice, orari, preside, professore, pronto_soccorso, registro, sabato, sala_d'attesa, scuola, scuola_materna, sil, superiori, sportello, studente, tecnico, telefono, ufficio_informazioni, voto]
  ]
)
]



== SignIT median-rank category proportions <app:signit-cat-proportions>

#figure(
  caption: [SignIT category distribution across median-rank buckets.],
)[
  #context if target() == "html" [
    #distill-figure-page([
      #table(
        columns: (2fr, 1fr, 1fr, 1fr, 1fr, 1fr),
        align: (left, center, center, center, center, center),
        table.header([Category], [Great], [Good], [Fair], [Neutral], [Adverse]),
        [animals], [1], [6], [14], [8], [3],
        [colors], [1], [5], [7], [3], [1],
        [emotions], [0], [3], [1], [1], [0],
        [family], [0], [1], [3], [7], [9],
        [food], [0], [2], [6], [9], [2],
      )
    ])
  ] else [
  #distill-figure-page([
    #block(width: 86%)[
      #let colors = (
        great: rgb("3b82f6"), good: rgb("ef4444"), fair: rgb("f59e0b"),
        neutral: rgb("10b981"), perverse: rgb("f97316")
      )

      #let horizontal-stacked-bar(category, data) = {
        grid(
          columns: (22%, 1fr),
          gutter: 6pt,
          align: (right + horizon, left),
          text(size: 8pt, weight: "medium", category),
          layout(size => {
            let total = data.sum()
            if total == 0 { total = 1 }
            stack(
              dir: ltr,
              spacing: 0pt,
              ..data.enumerate().map(((index, value)) => {
                let percentage = (value / total) * 100
                let bar_width = percentage * 1%
                let label = if percentage >= 12 {
                  text(size: 6.5pt, fill: white, weight: "bold", str(int(percentage)) + "%")
                } else { "" }

                rect(
                  width: bar_width,
                  height: 13pt,
                  fill: colors.values().at(index),
                  stroke: none,
                  align(center + horizon, label)
                )
              })
            )
          })
        )
      }

      #text(size: 10.5pt, weight: "semibold", fill: rgb("444444"))[
        SignIT - Distribution of Median-Rank Buckets by Category
      ]
      #v(4pt)

      #grid(
        columns: (auto, auto, auto, auto, auto),
        gutter: 6pt,
        ..colors.pairs().zip((
          "1. Great (1-3)", "2. Good (3.1-10)", "3. Fair (10.1-25)",
          "4. Neutral/Random (25.1-47)", "5. Adverse (47.1-93)"
        )).map((((key, color), label)) => {
          stack(dir: ltr, spacing: 3pt, square(size: 6pt, fill: color), text(size: 7pt, label))
        })
      )
      #v(8pt)

      #stack(
        spacing: 6pt,
        horizontal-stacked-bar("animals",  (1, 6, 14, 8, 3)),
        horizontal-stacked-bar("colors",   (1, 5, 7, 3, 1)),
        horizontal-stacked-bar("emotions", (0, 3, 1, 1, 0)),
        horizontal-stacked-bar("family",   (0, 1, 3, 7, 9)),
        horizontal-stacked-bar("food",     (0, 2, 6, 9, 2)),
      )

      #pad(left: 22% + 6pt)[
        #v(2pt)
        #grid(
          columns: (1fr, 1fr, 1fr, 1fr),
          align: (left, center, center, right),
          text(size: 7pt, fill: rgb("888888"), "0%"),
          text(size: 7pt, fill: rgb("888888"), "25%"),
          text(size: 7pt, fill: rgb("888888"), "50%"),
          text(size: 7pt, fill: rgb("888888"), "100%"),
        )
      ]
    ]
  ])
  ]
] <fig:app-signit-medianr-cat>

== A3LIS-147 median-rank category proportions <app:a3lis-cat-proportions>

#figure(
  caption: [A3LIS-147 category distribution across median-rank buckets.],
)[
  #context if target() == "html" [
    #distill-figure-page([
      #table(
        columns: (2fr, 1fr, 1fr, 1fr, 1fr, 1fr),
        align: (left, center, center, center, center, center),
        table.header([Category], [Great], [Good], [Fair], [Neutral], [Adverse]),
        [common life], [2], [5], [2], [5], [4],
        [education], [2], [3], [6], [6], [13],
        [highway], [0], [3], [0], [3], [2],
        [hospital], [1], [3], [7], [4], [4],
        [public inst.], [3], [8], [8], [11], [8],
        [railway station], [0], [3], [11], [17], [5],
      )
    ])
  ] else [
  #distill-figure-page([
    #block(width: 86%)[
      #let colors = (
        great: rgb("3b82f6"), good: rgb("ef4444"), fair: rgb("f59e0b"),
        neutral: rgb("10b981"), perverse: rgb("f97316")
      )

      #let horizontal-stacked-bar(category, data) = {
        grid(
          columns: (22%, 1fr),
          gutter: 6pt,
          align: (right + horizon, left),
          text(size: 8pt, weight: "medium", category),
          layout(size => {
            let total = data.sum()
            if total == 0 { total = 1 }
            stack(
              dir: ltr,
              spacing: 0pt,
              ..data.enumerate().map(((index, value)) => {
                let percentage = (value / total) * 100
                let bar_width = percentage * 1%
                let label = if percentage >= 12 {
                  text(size: 6.5pt, fill: white, weight: "bold", str(int(percentage)) + "%")
                } else { "" }

                rect(
                  width: bar_width,
                  height: 13pt,
                  fill: colors.values().at(index),
                  stroke: none,
                  align(center + horizon, label)
                )
              })
            )
          })
        )
      }

      #text(size: 10.5pt, weight: "semibold", fill: rgb("444444"))[
        A3LIS-147 - Distribution of Median-Rank Buckets by Category
      ]
      #v(4pt)

      #grid(
        columns: (auto, auto, auto, auto, auto),
        gutter: 6pt,
        ..colors.pairs().zip((
          "Great (1-3)", "Good (3.1-15)", "Fair (15.1-40)",
          "Nuetral (40.1-74)", "Adverse (74.1-148)"
        )).map((((key, color), label)) => {
          stack(dir: ltr, spacing: 3pt, square(size: 6pt, fill: color), text(size: 7pt, label))
        })
      )
      #v(8pt)

      #stack(
        spacing: 6pt,
        horizontal-stacked-bar("common life",      (2, 5, 2, 5, 4)),
        horizontal-stacked-bar("education",        (2, 3, 6, 6, 13)),
        horizontal-stacked-bar("highway",          (0, 3, 0, 3, 2)),
        horizontal-stacked-bar("hospital",         (1, 3, 7, 4, 4)),
        horizontal-stacked-bar("public inst.",     (3, 8, 8, 11, 8)),
        horizontal-stacked-bar("railway station",  (0, 3, 11, 17, 5)),
      )

      #pad(left: 22% + 6pt)[
        #v(2pt)
        #grid(
          columns: (1fr, 1fr, 1fr, 1fr),
          align: (left, center, center, right),
          text(size: 7pt, fill: rgb("888888"), "0%"),
          text(size: 7pt, fill: rgb("888888"), "25%"),
          text(size: 7pt, fill: rgb("888888"), "50%"),
          text(size: 7pt, fill: rgb("888888"), "100%"),
        )
      ]
    ]
  ])
  ]
] <fig:app-a3lis-medianr-cat>



= A3LIS-147 details and splits  <app:dataset>

#align(center)[
  #block(width: 90%)[
    #set text(size: 9pt)
    #table(
      columns: (1.5fr, 3fr),
      align: (left, left),
      stroke: (x, y) => if y == 0 { (top: 1pt, bottom: 0.5pt) } else if y == 9 { (bottom: 1pt) } else { none },
      fill: (x, y) => if y == 0 { gray.lighten(80%) } else { none },
      
      [*Parameter*], [*Configuration Details*],
      
      [Strategy], [_Signer-independent_ (Tests generalization to unseen signers)],
      [Train Signers], [`fal`, `fef`, `fsf`, `mdp`, `mdq`, `mic`, `mmr` (7 signers)],
      [Validation Signers], [`mrla` (1 signer)],
      [Test Signers], [`mrlb`, `msf` (2 signers)],
      
      [Split Ratio], [70% Train / 10% Val / 20% Test],
      [Total Signers], [10 total signers mapped],
      
      [Sample Estimates], [
        - ~147 signs per individual signer
        - *Train:* ~1,043 samples
        - *Val:* ~147 samples
        - *Test:* ~280 samples
      ],
      [Notes], [Alternative strategies can be configured by editing the raw config file.]
    )
    #v(-4pt)
    #text(size: 8pt, )[A3LIS-147 dataset split.]
  ]
]


The following table provides the full mapping used for our A3LIS-147 analysis, including category classification, presence in the SpreadTheSign (STS) corpus, and our qualitative iconicity proxy (visual similarity to English-speaking sign languages).

#let full-trans-table(body) = {
  // This rule allows the figure to break across pages
  show figure: set block(breakable: true)
  show "_": "_" + sym.zws
  
  figure(
    kind: table,
    caption: [A3LIS-147 dataset vocabulary, STS presence, and iconicity proxy.],
  )[
    #set text(size: 8pt)
    #table(
      columns: (2.5fr, 2fr, 2fr, 1.5fr, 2fr),
      fill: (col, row) => if row == 0 { luma(230) },
      stroke: 0.5pt + luma(150),
      table.header([*Italian*], [*English*], [*Category*], [*In STS?*], [*Iconicity Proxy*]),
      ..body
    )
  ]
}

#full-trans-table((
  [abbonamento], [subscription], [railway station], [yes but different], [no],
  [abitare], [live], [common life], [yes], [no],
  [acqua], [water], [common life], [yes], [no],
  [affitto], [rent], [common life], [yes], [no],
  [allegare], [attach], [education], [no], [yes],
  [allergia], [allergy], [hospital], [yes], [no],
  [ambulanza], [ambulance], [hospital], [yes], [no],
  [amministrazione], [administration], [public institute], [yes], [yes],
  [ancona], [ancona], [public institute], [no], [no],
  [andata], [one way], [railway station], [no], [no],
  [andata\_e\_ritorno], [round trip], [railway station], [no], [no],
  [annullato], [cancelled], [railway station], [yes], [yes],
  [aperto], [open], [common life], [yes], [yes],
  [arrivo], [arrival], [railway station], [yes], [no],
  [ascoli], [ascoli], [public institute], [no], [no],
  [asilo\_nido], [day nursery], [education], [yes but different], [no],
  [assente], [absent], [education], [yes], [no],
  [assessore], [assessor], [public institute], [no], [no],
  [assistente], [assistant], [public institute], [yes], [no],
  [assistente\_alla\_comunicazione], [communication assistant], [public institute], [no], [no],
  [autostrada], [motorway], [highway], [yes], [kind of],
  [avviso], [notice], [education], [yes], [yes],
  [banca], [bank], [public institute], [yes], [no],
  [bidello], [janitor], [education], [no], [no],
  [biglietto], [ticket], [railway station], [yes], [yes],
  [binario], [platform], [railway station], [yes but different], [no],
  [bocciato], [failed], [education], [yes but different], [no],
  [caldo], [hot], [common life], [yes but different], [no],
  [cambio], [change], [railway station], [no], [no],
  [casa], [home], [common life], [yes], [no],
  [casello], [toll gate], [highway], [yes], [yes],
  [chiuso], [closed], [common life], [yes but different], [yes],
  [cibo], [food], [common life], [yes], [yes],
  [civitanova], [civitanova], [public institute], [no], [no],
  [commissione], [commission], [education], [yes but different], [no],
  [compilare], [compile], [public institute], [yes], [no],
  [comune], [municipality], [public institute], [yes], [no],
  [consegnare], [deliver], [common life], [yes], [yes],
  [costo], [cost], [common life], [yes], [kind of],
  [cura], [care], [hospital], [yes], [yes],
  [data], [date], [public institute], [yes], [no],
  [diploma], [diploma], [education], [yes], [yes],
  [dirigente], [executive], [public institute], [yes], [yes],
  [disinfettare], [disinfect], [hospital], [no], [no],
  [dolore], [pain], [hospital], [yes but different], [no],
  [domani], [tomorrow], [railway station], [yes but different], [kind of],
  [domenica], [sunday], [railway station], [yes], [no],
  [elementari], [elementary school], [education], [no], [no],
  [emergenza], [emergency], [hospital], [yes], [no],
  [ente\_pubblico], [public body], [public institute], [no], [no],
  [entro], [within], [education], [no], [no],
  [esame], [exam], [education], [yes], [no],
  [falconara], [falconara], [public institute], [no], [no],
  [fano], [fano], [public institute], [no], [no],
  [fermo], [still], [railway station], [no], [no],
  [flebo], [intravenous drip], [hospital], [no], [no],
  [freddo], [cold], [common life], [yes], [yes],
  [giallo], [yellow], [hospital], [yes], [no],
  [giorni], [days], [railway station], [yes], [no],
  [giorno], [day], [railway station], [yes], [no],
  [giovedì], [thursday], [railway station], [yes but different], [no],
  [giudizio], [judgement], [education], [no], [yes],
  [ieri], [yesterday], [railway station], [yes], [yes],
  [impiegato], [employee], [public institute], [yes but different], [no],
  [infermiere], [nurse], [hospital], [yes but different], [kind of],
  [infezione], [infection], [hospital], [no], [no],
  [iniezione], [injection], [hospital], [no], [yes],
  [interprete], [interpreter], [public institute], [yes], [no],
  [inviare\_sms], [messaging], [common life], [no], [no],
  [istituto], [institute], [education], [yes], [no],
  [jesi], [jesi], [public institute], [no], [no],
  [laurea], [graduation], [education], [yes], [no],
  [lingua\_dei\_segni], [sign language], [common life], [yes but different], [no],
  [litro], [litre], [common life], [yes], [yes],
  [lunedì], [monday], [railway station], [yes], [no],
  [macerata], [macerata], [public institute], [no], [no],
  [malattia], [illness], [hospital], [yes], [no],
  [mangiare], [eat], [common life], [yes], [yes],
  [marca\_da\_bollo], [revenue stamp], [public institute], [no], [no],
  [marche], [marche], [public institute], [no], [no],
  [martedì], [tuesday], [railway station], [yes], [no],
  [mattina], [morning], [railway station], [yes], [kind of],
  [medico], [doctor], [hospital], [yes], [yes],
  [medie], [middle school], [education], [no], [no],
  [mercoledì], [wednesday], [railway station], [yes], [no],
  [mesi], [months], [railway station], [yes], [no],
  [modello], [model], [public institute], [yes], [no],
  [modulo], [form], [public institute], [yes], [yes],
  [multa], [fine], [highway], [yes], [yes],
  [nota], [note], [education], [yes], [kind of],
  [notte], [night], [railway station], [yes], [yes],
  [obliterare], [stamp], [railway station], [no], [no],
  [obliteratrice], [stamping machine], [railway station], [no], [no],
  [oggi], [today], [railway station], [yes], [no],
  [operazione], [operation], [hospital], [no], [no],
  [orari], [times], [railway station], [no], [no],
  [ospedale], [hospital], [hospital], [yes], [no],
  [partenza], [departure], [railway station], [yes], [no],
  [pesaro-urbino], [pesaro-urbino], [public institute], [no], [no],
  [pomeriggio], [afternoon], [railway station], [yes], [yes],
  [posta], [mail], [public institute], [yes], [kind of],
  [presente], [present], [education], [yes], [no],
  [preside], [headmaster], [education], [yes], [no],
  [professore], [professor], [education], [yes], [no],
  [promosso], [promoted], [education], [no], [yes],
  [pronto\_soccorso], [first aid], [hospital], [yes], [yes],
  [provincia], [province], [public institute], [yes but different], [no],
  [pubblica], [public], [public institute], [yes], [yes],
  [rallentamenti], [slowdowns], [highway], [no], [yes],
  [regione], [region], [public institute], [yes], [kind of],
  [registro], [log book], [education], [yes], [yes],
  [ricevuta], [receipt], [public institute], [no], [no],
  [ritardo], [delay], [railway station], [no], [no],
  [ritirare\_il\_numero], [take the number], [public institute], [no], [no],
  [ritorno], [return], [railway station], [no], [no],
  [roma], [rome], [public institute], [yes], [no],
  [rosso], [red], [hospital], [yes], [kind of],
  [s.benedetto], [s.benedetto], [public institute], [no], [no],
  [sabato], [saturday], [railway station], [yes], [no],
  [sala\_d'attesa], [waiting room], [hospital], [yes], [no],
  [scadenza], [expiration], [education], [yes], [no],
  [sciopero], [strike], [railway station], [yes], [yes],
  [scontrino], [receipt], [public institute], [yes], [kind of],
  [scuola], [school], [education], [yes], [no],
  [scuola\_materna], [nursery school], [education], [yes], [no],
  [segretario], [secretary], [education], [yes], [no],
  [senigallia], [senigallia], [public institute], [no], [no],
  [sera], [evening], [railway station], [yes], [kind of],
  [sil], [silence sign], [common life], [no], [no],
  [sindaco], [mayor], [public institute], [yes], [no],
  [sostegno], [aid], [education], [yes], [kind of],
  [sportello], [reception window], [public institute], [yes], [yes],
  [stazione], [station], [railway station], [yes], [no],
  [strada], [street], [highway], [yes], [yes],
  [studente], [student], [education], [yes], [no],
  [superiori], [high school], [education], [yes], [yes],
  [tassa], [fee], [public institute], [yes], [kind of],
  [tecnico], [technician], [highway], [yes], [yes],
  [telefono], [telephone], [common life], [yes], [yes],
  [torino], [turin], [public institute], [no], [no],
  [traffico], [traffic], [highway], [yes], [no],
  [tratta], [section], [highway], [no], [yes],
  [treno], [train], [railway station], [yes], [kind of],
  [ufficio\_informazioni], [information office], [public institute], [no], [no],
  [università], [university], [education], [yes], [no],
  [vacanze], [vacation], [common life], [yes], [yes],
  [venerdì], [friday], [railway station], [yes], [no],
  [verde], [green], [hospital], [yes], [no],
  [voto], [voting], [education], [yes], [yes]
))

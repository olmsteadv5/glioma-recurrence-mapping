# Sources

The reference behind each `# why: ... [tag]` comment in the code.

```
[recur21]   [secondary]
            Limited recurrence distance of glioblastoma under modern
            radiotherapy era. BMC Cancer. 2021.
            https://bmccancer.biomedcentral.com/articles/10.1186/s12885-021-08467-3
            -> why the standard margin is 2 cm
            -> 90-95% of recurrences within 2 cm of the GTV margin
            -> 80% within 10 mm
            -> ~69% local, ~18% distant, ~13% both
            -> roughly a third of patients recur multifocally (31 of 97 in one
               series), some with 3-5 separate lesions

[lee99]     [verified]
            Lee SW, Fraass BA, Marsh LH, Herbort K, Gebarski SS, Martel MK,
            Radany EH, Lichter AS, Sandler HM. Patterns of failure following
            high-dose 3-D conformal radiotherapy for high-grade astrocytomas: a
            quantitative dosimetric study. Int J Radiat Oncol Biol Phys.
            1999;43(1):79-88. doi:10.1016/s0360-3016(98)00266-1
            https://pubmed.ncbi.nlm.nih.gov/9989517/
            -> the abstract defines four classes by the share of the recurrence
               volume inside D95, the region treated to 95% of the prescription:
               central 95% or more, in-field 80% or more, marginal 20 to 80%,
               outside less than 20%
            -> the paper calls the last class "outside"; later papers call it
               "distant", which is the word our overlap CSV uses
            -> its 71 patients had "high-grade supratentorial astrocytomas",
               which in 1999 meant grade III and grade IV, glioblastoma being
               the grade IV astrocytoma. The abstract gives no split by grade.
               We borrow only the classes, which are defined by geometry against
               the high-dose region and are applied to glioblastoma routinely,
               e.g. https://pmc.ncbi.nlm.nih.gov/articles/PMC9872393/
            -> we have no dose, so measure.overlap substitutes 2 cm from the
               tumor core for the 60 Gy volume, which is smaller than a real 95%
               isodose (no PTV margin) and so undercounts central and in-field

[billot23]  [verified]
            Billot B, Greve DN, Puonti O, Thielscher A, Van Leemput K,
            Fischl B, Dalca AV, Iglesias JE. SynthSeg: Segmentation of brain
            MRI scans of any contrast and resolution without retraining.
            Med Image Anal. 2023;86:102789. doi:10.1016/j.media.2023.102789
            PMID 36857946. PMCID PMC10154424.
            -> trained on synthetic data with a random subset of the usual
               preprocessing applied per example, "such that SynthSeg can
               segment scans with or without preprocessing", which is why
               scan/tissue.py feeds it the raw T1 rather than the stripped one
            -> white matter lesions are simulated during training, pasted with
               50% probability, and the ablation in section 5.2 is offered as
               "a first evidence of the robustness of SynthSeg to the presence
               of lesions"
            WARNING the paper does not license this use on glioma. Its own
            limitation reads: "future work will seek to precisely quantify the
            performance of SynthSeg when exposed to various types of lesions,
            tumours, and pathologies." Nothing published supports the labels it
            returns inside a tumor, and this collection's tumors are 35-170 mL.
            Treat tumor-region labels as unvalidated; see scan/tissue.py.
```

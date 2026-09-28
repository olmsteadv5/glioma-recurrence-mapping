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

[lee99]     [secondary]
            Lee SW, Fraass BA, Marsh LH, et al. Patterns of failure following
            high-dose 3-D conformal radiotherapy for high-grade astrocytomas: a
            quantitative dosimetric study. Int J Radiat Oncol Biol Phys. 1999.
            https://pubmed.ncbi.nlm.nih.gov/9989517/
            -> the central / in-field / marginal / distant classification
            -> cutoffs >95%, 80-95%, 20-80%, <20% of the recurrence volume
               inside the high-dose volume, as later papers apply them:
               https://pmc.ncbi.nlm.nih.gov/articles/PMC9872393/
            -> secondary because the cutoffs were read from citing papers, not
               the original; confirm against the 1999 text before publishing
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

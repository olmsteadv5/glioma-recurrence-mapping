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

[rtog0825]  [verified]
            RTOG 0825 protocol (bevacizumab with chemoradiation for newly
            diagnosed glioblastoma), section 6.4, target volumes.
            https://irochouston.mdanderson.org/rpc/credentialing/files/0825.pdf
            -> drawn on the post-operative MRI, the pre-op used for correlation
            -> GTV1 = T2 or FLAIR abnormality plus all enhancement and the
               cavity; CTV1 = GTV1 + 2 cm; 46 Gy in 23 fractions
            -> no edema: enhancement and cavity + 2.5 cm instead
            -> GTV2 = enhancement and cavity; CTV2 = GTV2 + 2 cm; boost of
               14 Gy in 7 fractions to 60 Gy
            -> either CTV "may be reduced to 0.5 cm around natural barriers to
               tumor growth such as the skull, ventricles, falx", which is
               where analysis/walkdown.py takes its 5 mm floor
            -> PTV = CTV + 3 to 5 mm for setup; not modelled here

[nrg19]     [verified]
            Kruser TJ, Bosch WR, Badiyan SN, et al. NRG brain tumor
            specialists consensus guidelines for glioblastoma contouring.
            J Neurooncol. 2019;143(1):157-166. PMID 30888558, PMC6483830.
            -> ten NRG brain tumor radiation oncologists; volumes as RTOG:
               FLAIR + 2 cm for 46 Gy, cavity and enhancement + 2 cm for 60 Gy
            -> falx: trim "directly to the falx (not 5 mm); assume the falx
               is a hard barrier to tumor spread, as long as there is no FLAIR
               or T2 abnormality crossing the corpus callosum"
            -> careful trimming at barriers cut the planned brain by 13.6% on
               average (66.8 cc) without losing coverage
            -> closes by calling for target volumes "beyond isometric
               expansion techniques for individual patients"
            -> analysis/walkdown.py reads "crossing" as at least 1 mL of the
               original whole tumor past the midline; that cutoff is ours
```

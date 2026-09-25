# Sources

Every `# why:` comment in the code carries a bracketed tag that resolves here.
`grep -rn "why:" plan/ scan/ yale/ show/` lists every clinical decision with its
tag; this file says what each tag actually licenses.

**Rules.** One entry per source, not per use site. The `->` line names what the
paper supports, so a citation cannot quietly drift from the claim it backs. A
tag with no entry here, or an entry that was guessed rather than read, is worse
than no citation at all.

`[verified]` means the paper itself was opened and the figures below were read
out of it. `[secondary]` means the figures come from a review, abstract or
search result and the primary source has not been checked. **Do not promote a
`[secondary]` entry to `[verified]` without opening the paper.**

---

## Target margins and guidelines

```
[estro23]   [secondary]
            Niyazi et al. ESTRO-EANO guideline on target delineation and
            radiotherapy details for glioblastoma. Radiother Oncol 2023.
            https://www.thegreenjournal.com/article/S0167-8140(23)00201-3/fulltext
            -> 15 mm CTV measured from enhancing tumor + cavity, edema excluded
            -> 0 mm at skull, falx, tentorium, brainstem, optic pathways
            -> 5 mm at the ventricles
            -> all reductions apply "provided the tumour is distant from the
               white matter tracts extending to these regions"
            -> 82% Delphi consensus against routine SVZ inclusion
            NOTE author list unconfirmed. Confirm before citing formally.

[rct25]     [verified]
            Zhong L, Zhou P, Chen L, et al. Modified Target Delineation and
            Moderately Hypofractionated Radiotherapy for High-Grade Glioma:
            A Randomized Clinical Trial. JAMA Netw Open. 2025;8(7):e2523053.
            doi:10.1001/jamanetworkopen.2025.23053
            -> 154 patients randomized, 76 experimental / 78 standard
            -> CTV2 = CTV1 + 1 cm along white matter tracts, excluding adjacent
               normal-signal gyri, using a tract ATLAS not patient DTI
            -> median CTV 225.0 -> 174.4 cm3, a 22.5% reduction. This is the
               benchmark to beat.
            WARNING the experimental arm also changed fractionation (64-66/60/54
            Gy in 27 fx with a simultaneous boost, vs 50 Gy/25 then 10 Gy/5).
            Survival and recurrence outcomes are therefore CONFOUNDED and must
            not be attributed to delineation. The 22.5% is purely geometric and
            is unaffected.

[geo14]     [secondary]
            Postoperative Radiotherapy for Glioma: Improved Delineation of the
            Clinical Target Volume Using the Geodesic Distance Calculation.
            https://pmc.ncbi.nlm.nih.gov/articles/PMC4045722/
            -> barrier-aware geodesic expansion with skull and falx, n = 1
            -> validated on inter-observer conformity and runtime only,
               NEVER against recurrence
            -> listed "incorporate white matter fiber orientation" as future work
            NOTE author list unconfirmed.
```

## Speed multipliers

```
[swanson00] [secondary]
            Swanson KR, Alvord EC, Murray JD. A quantitative model for
            differential motility of gliomas in grey and white matter.
            Cell Prolif. 2000.
            https://pmc.ncbi.nlm.nih.gov/articles/PMC6621920/
            -> sets wm_gain = 5.0
            -> white matter diffusion coefficient roughly 5x grey, from CT-based
               model fits; the underlying cell-migration observation is Giese et al.
            NOTE the exact Giese citation is not yet run down. A first guess of
            2.0 was low by more than a factor of two, so this value matters.

[wei23]     [secondary]
            Wei Y, Li C, Cui Z, et al. Structural connectome quantifies tumour
            invasion and predicts survival in glioblastoma patients.
            Brain. 2023;146(4):1714-1727.
            https://academic.oup.com/brain/article/146/4/1714/6746935
            -> justifies risk_gain existing at all
            -> 117 + 42 patients; disruption defined as connection strength
               >2 SD below healthy controls
            -> distant-region disruption vs furthest recurrence distance,
               r = 0.60, p < 0.001
            -> two cases where disrupted regions matched recurrence invisible
               on the preoperative scan
            -> proposed RT integration but gave NO margins. That is our gap.
            NOTE they measured disruption from patient DTI against controls.
            We PREDICT it from atlas connectivity and parcel overlap. Not the
            same quantity; say so wherever this is cited.
```

## Distances and caps

```
[recur21]   [secondary]
            Limited recurrence distance of glioblastoma under modern
            radiotherapy era. BMC Cancer. 2021.
            https://bmccancer.biomedcentral.com/articles/10.1186/s12885-021-08467-3
            -> sets ceiling = 20 mm and motivates satellite_ml
            -> 90-95% of recurrences within 2 cm of the GTV margin
            -> 80% within 10 mm
            -> ~69% local, ~18% distant, ~13% both
            -> roughly a third of patients recur multifocally (31 of 97 in one
               series), some with 3-5 separate lesions. This is why disconnected
               components are KEPT rather than discarded.

[yamahara10][secondary]
            Yamahara et al. 2010, seven autopsy brains.
            -> corroborates ceiling, and explains where 15 mm comes from
            -> tumor cells detectable to 14 mm beyond the contrast-enhancing
               margin
            NOTE full citation not yet run down. Reached via a 2017 review.

[infiltrate17] [secondary]
            Tumor Infiltration in Enhancing and Non-Enhancing Parts of
            Glioblastoma: A Correlation with Histopathology. PLOS One. 2017.
            https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0169292
            -> supports growing from FLAIR rather than enhancement alone
            -> >80% of non-enhancing samples contain tumor on histopathology
            -> non-enhancing tissue had the highest viable-tumor content

[butterfly22] [verified]
            Hazaymeh M, Lober-Handwerker R, Doring K, et al. Prognostic
            differences and implications on treatment strategies between
            butterfly glioblastoma and glioblastoma with unilateral corpus
            callosum infiltration. Sci Rep. 2022;12:19208.
            https://www.nature.com/articles/s41598-022-23794-6
            -> sets callosum = 15 mm
            -> 363 consecutive newly diagnosed GBM
            -> 96 (26.4%) had corpus callosum involvement
            -> of those 96, only 27 (28%) crossed into both hemispheres
               = 7.4% of the whole cohort
            -> this conditional rate is the real justification: tumor reaching
               the callosum is COMMON, crossing it is NOT. So the corridor is
               allowed through the callosum and capped past midline.

[butterfly12] [verified]
            Dziurzynski K, Blas-Boria D, Suki D, Cahill DP, Prabhu SS,
            Puduvalli V, Levine N. Butterfly glioblastomas: a retrospective
            review and qualitative assessment of outcomes.
            J Neurooncol. 2012;109(3):555-563.
            https://pmc.ncbi.nlm.nih.gov/articles/PMC3992290/
            -> 23 of 792 newly diagnosed GBM = 2.9%
            -> second, independent estimate of the butterfly rate

            CORRECTION a figure of "3.8-6.2%" appeared in an early draft of the
            plan. It could not be traced to any paper. The verified range across
            the two entries above is 2.9-7.4%. Do not reuse 3.8-6.2%.
```

## Subventricular zone

```
[svz25]     [secondary]
            Therapeutic and prognostic impact of target volume delineation in
            postoperative radiotherapy for high-grade glioma patients with
            subventricular zone involvement. Radiat Oncol. 2025.
            https://pmc.ncbi.nlm.nih.gov/articles/PMC11841192/
            -> justifies svz: auto firing on contact
            -> V60Gy >= 50% of the ipsilateral SVZ improves local control and PFS
               in patients whose tumor contacts the SVZ
            -> no benefit in non-contacting tumors, with real neurotoxicity cost

[svz13]     [secondary]
            Evaluation of high ipsilateral subventricular zone radiation therapy
            dose in glioblastoma: a pooled analysis.
            https://pubmed.ncbi.nlm.nih.gov/23462418/
            -> second source for the same conditional effect
            NOTE only the PubMed ID is confirmed; figures not read out.

[svzdef20]  [secondary]
            Valiyaveettil D, Malik M, Akram KS, Ahmed SF, Joseph DM.
            Prospective study to assess the survival outcomes of planned
            irradiation of ipsilateral subventricular and periventricular zones
            in glioblastoma. ecancermedicalscience. 2020;14:1021.
            https://doi.org/10.3332/ecancer.2020.1021
            -> sets SVZ_MM = 5.0 in plan/barriers.py
            -> "The SVZ was contoured as a 5 mm lateral expansion of the
               lateral ventricles as per published guidelines"
            NOTE this entry fixes the GEOMETRY of the rind only, not the dose.
            The 5 mm figure is the standard contouring convention across the SVZ
            literature; the guidelines the sentence defers to were not chased
            down, and the quote comes from the PMC HTML rather than the PDF.
```

## Functional sparing

```
[ntms22]    [secondary]
            nTMS-derived DTI-based motor fiber tracking in radiotherapy
            treatment planning of high-grade gliomas.
            https://www.sciencedirect.com/science/article/pii/S0167814022001955
            -> supports soft-only sparing
            -> nTMS lands within ~5 mm of direct cortical stimulation

[fmri15]    [secondary]
            Integration of BOLD-fMRI and DTI into radiation treatment planning
            for high-grade gliomas. Radiat Oncol. 2015.
            https://link.springer.com/article/10.1186/s13014-015-0364-1
            -> fMRI/DTI localization error up to 1 cm
            -> together with [ntms22]: an ATLAS prior is worse than either,
               because it describes the average brain rather than this patient,
               and these brains have reorganized around a tumor. Hence soft
               penalty only, never hard exclusion.
```

## Deliverability

```
[penumbra]  [secondary]
            -> sets sigma_mm = 5 and the tube_mm >= 1.2 * sigma_mm floor
            -> 6 MV photon 80-20% lateral penumbra approximately 3-5 mm
            -> standard MLC leaf width 5 mm (Varian Millennium 120) or 10 mm
               (Millennium 80); 3 mm micro-MLC for small targets
            NOTE assembled from dosimetry comparisons found via search, not from
            a primary reference or an AAPM task group report. Find one.
```

## Tools

```
[brats25]   [secondary]
            BraTS package: https://arxiv.org/abs/2506.13807
            Adult Glioma Segmentation (Pre & Post Treatment) 2025 challenge:
            https://arxiv.org/abs/2405.18368
            -> BraTS25_1 won the challenge and requires CUDA
            -> BraTS25_2 is the best CPU algorithm, and is what every result
               computed on this machine so far has used

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

[billot23b] [secondary]
            Billot B, Magdamo C, Cheng Y, Arnold SE, Das S, Iglesias JE.
            Robust machine learning segmentation for large-scale analysis of
            heterogeneous clinical brain MRI datasets. PNAS 2023;120(9).
            doi:10.1073/pnas.2216399120
            -> SynthSeg+, which is what mri_synthseg --robust runs
            -> cited here only because the --robust flag exists; the pipeline
               does not use it
            NOTE author initials taken from the SynthSeg README, not the paper.

TODO        SynthStrip (Hoopes et al.)
TODO        greedy registration (Yushkevich et al.)
TODO        Yale Brain Atlas
TODO        hippocampal avoidance, NRG-CC001
```

## Worth reading

```
From Fiber Tracts to Tumor Spread: Biophysical Modeling of Butterfly Glioma
Growth Using Diffusion Tensor Imaging. https://arxiv.org/pdf/2507.17707
-> recent, and the closest published work to the corridor mechanism.
   Read before stage 2 of the parameter fitting.
```

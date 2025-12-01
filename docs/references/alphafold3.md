AlphaFold 3: Unified Deep Learning for Biomolecular Complexes

AlphaFold 3 (AF3) extends the AlphaFold family from mainly protein-focused structure prediction (AF2, AlphaFold-Multimer) to a general model for biomolecular complexes that can jointly predict proteins, DNA/RNA, ligands, ions, glycans and modified residues in one framework. The motivation is that accurate complex structures are critical for mechanistic biology and drug design, but existing methods are usually specialized (only ligands, only nucleic acids, etc.) and often less accurate than physics-based tools in their niche. AF3 shows that a single deep network can match or outperform many of these specialized systems across tasks.

Architectural changes: from Evoformer to Pairformer + Diffusion

AF3 keeps the overall “trunk + structure module” idea of AF2 but changes both parts. The trunk reduces the emphasis on MSAs and replaces the AF2 Evoformer with a Pairformer: 48 attention blocks operating on pair and single representations, while the MSA stream is much smaller and discarded early. This makes the model more data-efficient and easier to extend beyond proteins.

The structure module is completely redesigned as a diffusion model over raw atom coordinates. Instead of predicting residue frames and torsion angles, AF3 repeatedly denoises noisy coordinates to obtain a final 3D structure. Low noise steps force the network to learn local stereochemistry, while high noise steps force it to learn global arrangements. Because the model is generative, multiple samples per seed give a distribution of possible conformations, and local geometry is usually sharp even when global placement is uncertain (e.g., side chains). Notably, the architecture does not enforce SE(3) equivariance; the authors find good performance without explicit rotation/translation invariance, simplifying the design.

To reduce hallucinated ordered structure in disordered regions—a risk with generative diffusion—the training set is cross-distilled from AlphaFold-Multimer predictions, which typically show extended loops where no stable structure exists. Training AF3 to mimic that behaviour significantly reduces hallucinations on the CAID2 disorder benchmark. Confidence metrics (pLDDT, PAE and a distance-error matrix) are trained using a special “mini-rollout” diffusion during training so that the confidence head sees realistic full-structure predictions.

Accuracy across proteins, nucleic acids, and ligands

On a large “recent PDB” benchmark, AF3 improves over AlphaFold-Multimer v2.3 for protein monomers, protein–protein interfaces, and especially antibody–antigen complexes. The bar plots in Fig. 1c show higher DockQ success rates for protein–protein and antibody interfaces and higher LDDT for monomers.

For protein–ligand docking, AF3 is evaluated on the PoseBusters benchmark, using only sequence and ligand SMILES as input (no bound protein structure). Even though traditional docking tools like AutoDock Vina have access to the experimental protein pocket, AF3 achieves a much higher fraction of ligands with pocket-aligned RMSD < 2 Å and better stereochemical validity. Extended Data Figs. 3–4 illustrate cases where AF3 recovers accurate ligand poses while Vina and Gold fail, and show that this performance holds even when training data are carefully time-filtered.

For nucleic acids, AF3 predicts protein–DNA/RNA complexes and RNA-only structures with higher accuracy than RoseTTAFold2NA on several benchmarks, including a filtered subset of recent PDB complexes (<1000 residues) and CASP15 RNA targets. Extended Data Fig. 5 shows that AF3 beats RF2NA and earlier AIchemy_RNA variants on most CASP15 RNA targets, although the very best human-aided CASP submission (AIchemy_RNA2) still has a slight edge. AF3 also models covalent modifications (phosphorylation, base methylation, glycans) with good local accuracy; modelling these explicitly often improves backbone accuracy relative to simply replacing them with standard residues.

Confidence outputs remain well calibrated: higher pLDDT and interface ipTM correlate strongly with higher LDDT, DockQ, and ligand success rates across proteins, nucleic acids and ligands, as shown in Fig. 4 and Extended Data Fig. 8.

Limitations and open challenges

The paper is very explicit about remaining limitations. AF3 can violate chirality (wrong stereocenters on ligands) and sometimes produces severe atomic clashes, especially in large protein–nucleic complexes; the authors mitigate this partly by penalizing clashes and chirality when ranking samples, but cannot eliminate them. Like AF2, AF3 predicts static structures rather than ensembles, and frequently prefers a single conformational state even when biology demands multiple states (e.g., open vs closed E3 ligases). Antibody–antigen modelling improves with many seeds, but good predictions can require hundreds–thousands of samples, which is computationally expensive.

For RNA specifically, AF3 still struggles with some single-stranded and flexible regions, and it is evaluated mainly at the whole-chain or whole-complex level (LDDT, iLDDT, DockQ), not at the motif level (internal loops, junctions, non-canonical pairs).

Relevance to motif-level RNA work

In the context of your project, AF3 represents the state of the art in end-to-end biomolecular structure prediction from sequence, but:

It does not take cryo-EM density as an input; it works from sequence, MSAs, templates and ligand SMILES.

Its outputs and metrics are global (whole complex / interface accuracy), not designed to explicitly classify or label RNA motifs.

The model is very large and general-purpose; it is not tuned for tasks like “given a local density cube + PDB-derived features, what RNA motif is this?”.

So AF3 is highly relevant background as proof that diffusion-based, unified architectures can learn complex biomolecular geometry, including RNA. But there is still a clear gap for lightweight, motif-level models that integrate cryo-EM density and nucleic-acid-specific geometric features—exactly the space your RNA motif U-Net is targeting.
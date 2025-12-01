2. Literature Review
2.1 Cryo-EM and the Model-Building Gap

The “resolution revolution” in cryogenic electron microscopy (cryo-EM) has led to an explosion of high-resolution density maps for large macromolecular complexes, including protein–DNA/RNA assemblies. As of August 2022 there were 21,807 deposited cryo-EM maps, but only 12,166 associated atomic models, highlighting a large gap between data collection and full model building. The workflow illustrated in Figure 1 on page 2 of the DeepTracer-2.0 paper shows how raw micrographs are reconstructed into a 3D density map that typically contains both protein and nucleic-acid density; the bottleneck is turning that map into an accurate, complete macromolecular model.

Classical pipelines such as Phenix’s map_to_model and Rosetta-based approaches can build de novo models from cryo-EM density, but they tend to be computationally intensive, require parameter tuning, and often struggle with large complexes or mixed macromolecules (proteins plus DNA/RNA). These methods also suffer when different macromolecules are not clearly separated in the density, which leads to misassignment of voxels and false positives in the final structure.

2.2 Deep Learning for Density Segmentation and Annotation

Convolutional neural networks (CNNs) have increasingly been used to interpret cryo-EM maps at the voxel level. Haruspex, for example, uses a 3D U-Net to annotate 40³-voxel segments into four labels: α-helix, β-strand, nucleotide, or unassigned. The network’s output probability maps can be combined so that α/β/unassigned voxels represent protein density, while the nucleotide channel captures DNA/RNA density.

DeepTracer-2.0 adopts Haruspex-style segmentation as a front-end: the full macromolecular map is first passed through a CNN to separate protein and nucleotide density maps. The segmentation result in Figure 4 on page 7 shows an example where cyan voxels correspond to protein and dark-blue voxels to RNA density, forming the input to separate downstream protein and nucleotide pipelines. Accurate segmentation is crucial; mis-separated density leads to amino acids being “built” in nucleic-acid regions and vice versa.

2.3 From DeepTracer-1.0 to DeepTracer-2.0

DeepTracer-1.0 introduced a fully automated, deep learning–based pipeline for de novo modeling of multi-chain protein complexes from cryo-EM maps. It uses 3D U-Nets to predict voxel-wise atom types and backbone vs non-backbone labels, followed by geometric and sequence-based postprocessing to trace polypeptide chains and assign amino-acid types. Compared with state-of-the-art methods such as Phenix, Rosetta, and MAINMAST, DeepTracer-1.0 achieved higher Cα coverage and lower RMSD while being significantly faster.

However, DeepTracer-1.0 treated all non-protein density as background. In maps that contained nucleic acids, carbohydrates or lipids, those regions could be misinterpreted as protein, leading to false positive amino-acid predictions. The introduction and discussion sections of the DeepTracer-2.0 paper explicitly highlight this limitation and motivate an extension that can recognize and model nucleic acids alongside proteins.

2.4 DeepTracer-2.0 Pipeline for Protein–DNA/RNA Complexes

DeepTracer-2.0 extends the original pipeline to handle protein–DNA/RNA complexes by adding (i) a segmentation step that separates protein and nucleotide densities, and (ii) a nucleotide-specific U-Net and postprocessing pipeline. Figure 3 on page 6 summarizes the architecture: a shared segmentation and preprocessing stage followed by parallel protein and nucleotide branches, which are later merged into a single macromolecular model.

Preprocessing and data selection. Only high-resolution maps (≤ 4 Å) are used. For nucleotide network training, 293 EMDB/PDB pairs are selected where cryo-EM maps and models are consistent, contain at least one protein chain with α-helix/β-sheet and at least one nucleotide chain of ≥20 base pairs, and meet resolution criteria. Density values are normalized to [0, 1] and resampled into 64³-voxel cubes, similar to DeepTracer-1.0, so both protein and nucleotide networks operate on standardized input volumes.

Nucleotide U-Nets. Because nucleotide geometry differs markedly from proteins, DeepTracer-2.0 trains separate U-Nets for nucleic acids:

An atom-type U-Net predicts, for each voxel, whether it contains phosphate P, sugar carbons C1′ or C4′, or no atom (four output channels).

A backbone U-Net predicts whether a voxel belongs to the sugar-phosphate backbone, to the nitrogenous base, or to neither (three output channels).

Both networks focus on correctly identifying the phosphate backbone, which is the key scaffold for constructing DNA/RNA chains.

Nucleotide postprocessing. The raw voxel predictions are refined by enforcing nucleic-acid geometry. Postprocessing reduces the number of predicted phosphates and connects them into chemically plausible backbones:

Distances between neighboring P atoms are constrained, typically within ~5.9–8 Å, consistent with A-form and B-form DNA/RNA conformations.

Pseudotorsion angles based on P and C1′ atoms are used to simplify and regularize backbone geometry and to help distinguish different conformational states.

The Brickworx tool is then applied to fit helical motifs and recurrent RNA/DNA fragments into the refined P positions and density, extending double-stranded helices and recurrent motifs.

In parallel, the amino-acid branch uses the original DeepTracer-1.0 U-Nets for protein atoms and backbone prediction; both branches share the same segmentation and preprocessing steps. Finally, the predicted amino-acid and nucleotide models are combined to produce the full macromolecular complex.

2.5 Evaluation Against Phenix

The DeepTracer-2.0 paper evaluates performance on 20 experimental protein–DNA/RNA cryo-EM maps (2.0–4.0 Å) with both amino-acid and nucleotide chains. For amino acids, metrics include Cα RMSD, percent matching residues, sequence identity, and false-positive and false-connection rates; for nucleotides, phosphate precision and nucleotide precision (joint correctness of P and C1′ positions) are used.

Across these maps, DeepTracer-2.0 substantially outperforms Phenix’s map_to_model pipeline. Table 1 on page 13 reports average Cα RMSD of 0.638 Å versus 1.017 Å, matching residues of 85.3% versus 65.4%, and sequence identity of 81.5% versus 49.7%, with far fewer false positives and false connections for amino acids. Nucleotide phosphate and nucleotide precision are also slightly higher for DeepTracer-2.0 (0.872 and 0.793) than for Phenix (0.866 and 0.752).

The runtime comparison in Figure 7 on page 12 shows that DeepTracer-2.0 is orders of magnitude faster: small complexes that take Phenix ~6 hours can be processed by DeepTracer-2.0 in ~5 minutes, and the largest complexes in the benchmark are modeled in just over six minutes, whereas Phenix may require more than a day.

Visual comparisons in Figure 6 on page 12 highlight that DeepTracer-2.0 usually produces more complete protein and DNA models, especially for nucleosome-like complexes where DNA wraps around a protein core. In some more challenging RNA maps, both methods struggle with single-stranded regions, and in certain double-helix cases Phenix can locally outperform DeepTracer on RNA while still being slower overall.

2.6 Remaining Challenges and Research Gap

Despite strong quantitative and qualitative performance, several limitations remain. Accuracy and segmentation quality deteriorate on lower-resolution maps or very large complexes, and single-stranded RNA is specifically noted as a difficult case for future work. The current nucleotide pipeline focuses on backbone reconstruction (P and sugar atoms) and generic helical motifs; it does not explicitly detect or classify finer-grained RNA motifs such as hairpins, internal loops, junctions, or non-canonical base pairs.

This leaves an open space for methods that:

Use cryo-EM density together with richer PDB-derived geometric features (e.g., base-pairing patterns, torsion angles, pseudotorsions, local distances)

Operate explicitly at the motif level rather than just whole-chain backbones

Are tailored to single-stranded or partially ordered RNA regions, which are currently the most challenging cases for automated pipelines

Our project is positioned in this gap: we build on DeepTracer-style 3D U-Net architectures and preprocessing, but specialize the learning objective to RNA motif recognition and classification, using additional nucleic-acid–specific channels beyond the backbone alone to better capture local structural context in cryo-EM maps.
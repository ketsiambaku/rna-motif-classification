# RNA Motif Classification - Project Roadmap

## Project Timeline Overview

**Project Duration**: 12-14 weeks (Full Semester)  
**Deliverables**: Proposal → Progress Report (3 pages) → Final Paper (15 pages) + Code + Results

---

## Semester Milestones

### Week 1-2: Project Proposal & Setup ✓
**Status**: COMPLETED

**Deliverables**:
- [x] Project proposal presentation
- [x] Problem definition: 14-class RNA motif classification
- [x] Dataset acquisition and organization
- [x] Git repository initialization
- [x] Environment setup (Python, PyTorch, dependencies)

**Key Decisions**:
- Two models: U-Net (primary) and 3D CNN (baseline)
- Three features: sequence, torsion angles (if time allows), base pairing
- Multi-channel enhancement: 4-channel U-Net with component masks

---

### Week 3-4: Baseline Understanding & Data Leakage Discovery
**Status**: ✅ COMPLETED (Phase 1.1)

**Tasks Completed**:
- ✅ Understood example code in `example/TrainingForClassification/`
- ✅ Ran baseline code on dataset1 and dataset2
- ✅ **Discovered severe data leakage**: PDB P-P distance sparsity encodes size
- ✅ Quantified leakage: 76.8pp accuracy inflation (98.8% → 21.99%)
- ✅ Established realistic baseline: **58.51%** (density-only, 3-class)
- ✅ Documented comprehensive leakage analysis (500+ lines)
- ✅ Trained multiple models: PDB-only, density-only (14-class & 3-class)
- ✅ Generated confusion matrices and statistical analysis

**Deliverables**:
- ✅ Baseline performance reports (dataset1 & dataset2)
- ✅ `docs/dataset2-leakage-analysis.md` - comprehensive analysis
- ✅ `docs/sequence-features-analysis.md` - Phase 2 specification
- ✅ Trained models: `best_density_3class_model.pth` (58.51% accuracy)
- ✅ Analysis scripts and training code for leakage detection

**Key Findings**:
- Current 100% accuracy is **INVALID** (size leakage via sparsity)
- Density-only achieves 58.51% on 3-class (proves topology learning works)
- Need size-invariant PDB features for Phase 2

**Academic Focus**:
- ✅ Identified critical flaw in current approach (major contribution!)
- ✅ Established methodology for detecting feature leakage
- ✅ Built foundation for "Problem Discovery" section in paper
- ✅ Realistic baseline for comparison (58.51% without leakage)

---

### Week 4 (End) - 5 (Start): Paper Writing - Introduction & Related Work
**Status**: ✅ COMPLETED (November 29-30, 2025)

**Tasks Completed**:
- ✅ Created IEEE format paper skeleton (`docs/paper.md`)
- ✅ Wrote Section 1.1: Background and Motivation
- ✅ Wrote Section 1.2: Problem Statement (formal definition, 4 key challenges)
- ✅ Created Section 1.3: Contributions (placeholder)
- ✅ Wrote Section 1.4: Paper Organization
- ✅ Wrote Section 2.1: RNA Secondary Structure Prediction (MXfold2)
- ✅ Wrote Section 2.2: Deep Learning for Cryo-EM (DeepTracer 1.0 & 2.0)
- ✅ Wrote Section 2.3: Unified Biomolecular Prediction (AlphaFold 3)
- ✅ Wrote Section 2.4: RNA-Specific Cryo-EM (DeepCryoRNA)
- ✅ Wrote Section 2.5: RNA Motif Similarity (RNAMotifComp)
- ✅ Wrote Section 2.6: Integrated RNA Folding (CaCoFold-R3D)
- ✅ Wrote Section 2.7: Research Gaps (5 comprehensive gaps)

**Papers Integrated**:
1. ✅ MXfold2 (Sato et al. 2021) - thermodynamic + DL integration
2. ✅ DeepTracer 1.0 (Pfab et al. 2020) - protein cryo-EM reconstruction
3. ✅ DeepTracer 2.0 (Pfab et al. 2022) - nucleic acid extension
4. ✅ AlphaFold 3 (Abramson et al. 2024) - diffusion models, multi-modal
5. ✅ DeepCryoRNA (Li & Chen 2025) - RNA-specific, 18 atom types
6. ✅ RNAMotifComp (Petrov et al. 2013) - overlapping families challenge
7. ✅ CaCoFold-R3D (Karan & Rivas 2025) - probabilistic grammar

**Deliverables**:
- ✅ `docs/paper.md` - 700+ line IEEE format paper
- ✅ Sections 1 (Introduction) and 2 (Related Work) complete
- ✅ All citations properly formatted with \cite{} commands
- ✅ Problem statement with mathematical notation
- ✅ Research gaps clearly identified and positioned
- ✅ 7 reference summary files in `docs/references/`

**Paper Progress**:
- ✅ Section 1: Introduction (1.1-1.4) - COMPLETE
- ✅ Section 2: Related Work (2.1-2.7) - COMPLETE
- ⏸️ Section 3: Dataset - Statistics ready, needs writing
- ⏸️ Section 4: Data Leakage Analysis - Phase 1.1 findings ready
- ⏸️ Section 5: Methodology - Awaiting Phase 2 implementation
- ⏸️ Section 6-9: Awaiting experimental results

**Academic Impact**:
- Strong literature coverage across RNA structure, cryo-EM, and deep learning
- Clear positioning of our work in existing research landscape
- Identified 5 research gaps that justify our approach
- Ready foundation for methodology and results sections

---

### Week 5: Phase 2.1 - Sequence Feature Implementation
**Status**: ✅ COMPLETED (December 1, 2025)

**Tasks Completed**:
- [x] Implement RNA sequence extractor from PDB SEQRES records
- [x] Compute 24 size-invariant sequence features:
  * Nucleotide composition (A%, U%, G%, C%)
  * GC content, purine/pyrimidine ratios
  * Di-nucleotide frequencies (16 features)
  * Sequence entropy (normalized)
- [x] Validate size-invariance implementation
- [x] Create hybrid dataset loader (density + sequence)
- [x] Implement hybrid U-Net architecture
- [x] Add 6-class consolidation support (small/large internal/bulge/hairpin)
- [x] Fix class weights bug (dimension mismatch for consolidation)
- [x] Create Colab training notebook with GPU support
- [x] Document sequence features in paper Methods section

**Deliverables**:
- ✅ `src/features/sequence_features.py` - 400+ line sequence extractor
- ✅ `src/data/hybrid_dataset.py` - 461+ line hybrid dataset loader
- ✅ `src/models/hybrid_unet.py` - 350+ line hybrid model architecture
- ✅ `src/train_hybrid.py` - 474+ line training script with CLI args
- ✅ `colab_training.ipynb` - Colab notebook for T4 GPU training
- ✅ `docs/colab-class-weights-fix.md` - Bug fix documentation
- ✅ Paper Methods section 5.4.1 - Detailed sequence feature documentation
- 🔄 Training on Colab (awaiting results)

**Success Metrics Achieved**:
- ✅ Sequence features extract correctly (24 features validated)
- ✅ All features use size-invariant formulas (percentages/ratios)
- ✅ Hybrid model architecture complete (3.7M parameters)
- ✅ Dataset supports 6-class and 15-class modes
- ✅ Class weights computed correctly for n_classes
- ✅ Colab setup complete with dataset2.zip extraction
- 🔄 Training accuracy (awaiting Colab results)

---

### Week 6: Phase 2.2 - Base Pairing Features
**Status**: PENDING (After Week 5)

**Tasks**:
- [ ] Implement base pairing detector (N1/N3 atom distances)
- [ ] Compute ~10 size-invariant pairing features:
  * Pairing ratio, pairing density
  * Average pairing distance (normalized)
  * Stem length distribution
  * Loop closure patterns
- [ ] Update hybrid model to include pairing branch
- [ ] Train updated model
- [ ] Validate size-invariance

**Deliverables**:
- `src/features/base_pairing.py` - pairing feature extractor
- Updated hybrid model with pairing branch
- Trained model with **70-75% validation accuracy**
- Feature ablation analysis

**Success Metrics**:
- Base pairing features extract correctly (~10 features)
- **70-75% validation accuracy** ← **SUFFICIENT FOR PAPER**
- Captures secondary structure topology differences

**Decision Point**:
- If accuracy ≥72%, proceed to Week 8 (scaling)
- If accuracy <72%, continue to Week 7 (torsion angles)

---

### Week 7: Phase 2.3 - Torsion Angles (OPTIONAL)
**Status**: CONDITIONAL (Only if Week 6 < 72%)

**Tasks**:
- [ ] Implement torsion angle calculation (7 angles × 30 residues)
- [ ] Extract backbone and glycosidic torsion angles
- [ ] Update hybrid model to include torsion branch
- [ ] Train final model
- [ ] Validate size-invariance

**Deliverables**:
- `src/features/torsion_angles.py` - torsion extractor
- Final hybrid model with all features
- Trained model with **75-80% validation accuracy**

**Note**: Complex implementation (5-7 days), only add if needed

---

### Week 8: Phase 3 - Scaling to Full Dataset
**Status**: PENDING (After Phase 2.1+2.2 or 2.3)

**Tasks**:
- [ ] Implement P-P distance extraction from PDB files
- [ ] Implement torsion angle calculation (7 angles × 30 residues)
- [ ] Implement base pairing detection (N1/N3 proximity)
- [ ] Validate feature extraction on known structures
- [ ] Create feature extraction pipeline
- [ ] Handle edge cases (missing atoms, variable residue counts)

**Deliverables**:
- `src/features/pdb_extractor.py` with all three features
- Unit tests for feature extraction
- Feature validation report
- Documentation of geometric calculations

**Success Metrics**:
- All features extract with correct dimensions
- No NaN or Inf values in extracted features
- Reasonable value ranges confirmed

---

### Week 7-8: Mask Generation & Multi-Channel U-Net
**Status**: PENDING

**Tasks**:
- [ ] Fix bugs in `example/label/` mask extraction code
- [ ] Implement corrected ribose mask generation
- [ ] Implement phosphate mask generation
- [ ] Implement base mask generation
- [ ] Verify mask alignment with density maps
- [ ] Implement 4-channel U-Net architecture
- [ ] Test multi-channel input processing

**Deliverables**:
- `src/preprocessing/mask_generator.py`
- `src/models/multi_channel_unet.py`
- Visual validation of masks (overlays on density maps)
- Multi-channel U-Net implementation

**Known Issues to Fix**:
- Ribose and sugar labels are identical in example code
- Need proper separation of backbone, ribose, and base components

---

### Week 9: Baseline 3D CNN Implementation
**Status**: PENDING

**Tasks**:
- [ ] Implement simple 3D CNN architecture
- [ ] Add late fusion with PDB features
- [ ] Match input/output interface with U-Net
- [ ] Preliminary training on small subset
- [ ] Document architectural differences

**Deliverables**:
- `src/models/cnn.py`
- Baseline CNN trained model
- Comparison with U-Net on validation set

**Academic Purpose**:
- Establishes baseline for comparison
- Required: "at least two algorithms" per guidelines
- Demonstrates U-Net advantage (if any)

---

### Week 10: Progress Report Preparation (DEADLINE)
**Status**: PENDING

**Tasks**:
- [ ] Write 3-page progress report in ACM format
- [ ] Include sections:
  - Abstract (150 words)
  - Introduction (problem + background)
  - Preliminary Methods
  - Initial Results (if any)
  - Plan toward final paper
- [ ] Create flowchart of system architecture
- [ ] Prepare 1-minute flash presentation
- [ ] Submit progress report

**Deliverables**:
- 3-page ACM format PDF
- Presentation slides (1 minute)
- Flowchart/architecture diagram

**Key Content**:
- Show completed work (feature extraction, models implemented)
- Show initial experimental results (even preliminary)
- Clear plan for remaining weeks

---

### Week 11-12: Comprehensive Training & Experimentation
**Status**: PENDING

**Tasks**:
- [ ] Implement data augmentation (rotations, flips, noise)
- [ ] Train U-Net with different feature combinations
- [ ] Train multi-channel U-Net
- [ ] Train 3D CNN baseline
- [ ] Perform ablation studies:
  - Density only vs density + features
  - Different feature combinations
  - Single-channel vs multi-channel
- [ ] Implement cross-validation
- [ ] Track all experiments with proper logging
- [ ] Generate training curves and metrics

**Deliverables**:
- Trained models for all configurations
- Training logs and TensorBoard visualizations
- Ablation study results
- Model checkpoints saved

**Best Practices** (per guidelines):
- Data augmentation for accuracy boost
- Feature engineering experiments
- Consider ensemble methods
- Document all hyperparameters

---

### Week 13: Evaluation & Results Analysis
**Status**: PENDING

**Tasks**:
- [ ] Comprehensive evaluation on test set
- [ ] Generate confusion matrices
- [ ] Compute per-class precision, recall, F1
- [ ] Statistical significance tests (t-test, McNemar)
- [ ] Create ROC curves
- [ ] Perform error analysis
- [ ] Visualize predictions vs ground truth
- [ ] Compare with related work (if available)

**Deliverables**:
- Complete results table
- All evaluation plots (high-res, publication quality)
- Statistical analysis report
- Error analysis with examples

**Academic Focus**:
- Results must be reproducible
- Include error bars / confidence intervals
- Compare both models statistically
- Interpret biological significance

---

### Week 14-15: Final Paper Writing
**Status**: PENDING

**Tasks**:
- [ ] Write complete 15-page paper in ACM format
- [ ] Structure per guidelines:
  - Abstract (200-250 words)
  - Introduction (2 pages): Background + Related Work
  - Methods (3-4 pages): Dataset, Features, Models, Training
  - Experiments (3-4 pages): Setup, Configurations, Ablations
  - Results (2-3 pages): Tables, Figures, Comparisons
  - Discussion (1-2 pages): Interpretation, Limitations
  - Conclusion (1 page): Summary + Future Work
  - References (properly formatted)
- [ ] Create all figures (300 DPI minimum)
- [ ] Use professional tools for formulas (LaTeX)
- [ ] Proofread multiple times
- [ ] Get peer feedback

**Deliverables**:
- Final 15-page ACM format paper
- All figures in high resolution
- BibTeX references properly formatted

**Quality Checklist** (per instructor advice):
- [ ] Abstract clearly summarizes problem, method, results
- [ ] Introduction has clear problem statement + picture
- [ ] Method shows novelty and creative ideas
- [ ] Flowchart/pseudocode included
- [ ] Results use clear figures and tables
- [ ] Comparison with related work (or explanation why novel)
- [ ] References are scientific papers (no Wikipedia)
- [ ] Grammar and spelling checked

---

### Week 15-16: Final Deliverables Preparation
**Status**: PENDING

**Tasks**:
- [ ] Create comprehensive README.md
- [ ] Write detailed reproduction instructions
- [ ] Add setup and installation guide
- [ ] Document all commands with examples
- [ ] Include screenshots of execution
- [ ] Prepare final presentation slides
- [ ] Package deliverables:
  - Final paper PDF
  - Source code
  - README
  - Dataset (or download link)
  - Trained model checkpoints (optional)
- [ ] Create .zip file for submission
- [ ] Final presentation practice

**Deliverables**:
- Complete .zip package with:
  1. Final paper (PDF)
  2. Source code (well-documented)
  3. README.md (reproduction instructions)
  4. Dataset or download link
  5. Supplementary materials
- Final presentation (10-15 minutes)
- Demo of trained model

**README Must Include**:
- System requirements (Python version, GPU specs)
- Installation steps (pip install commands)
- Dataset preparation steps
- Training commands (exact commands used)
- Evaluation commands
- Expected results
- Screenshots showing execution

---

## Risk Management

### High Priority Risks

**1. Small Dataset Size**
- **Risk**: Limited training samples may lead to overfitting
- **Mitigation**: 
  - Aggressive data augmentation
  - Cross-validation
  - Regularization (dropout, weight decay)
  - Transfer learning if applicable

**2. Class Imbalance**
- **Risk**: Unequal distribution across 3 classes
- **Mitigation**:
  - Weighted loss function
  - Oversampling minority classes
  - Stratified train/val split
  - Report per-class metrics

**3. Mask Generation Bugs**
- **Risk**: Example code has known bugs
- **Mitigation**:
  - Fix bugs early (Week 7-8)
  - Visual validation of masks
  - Test on known structures first
  - Have single-channel U-Net as fallback

**4. Missing PDB Atoms**
- **Risk**: Some structures may have incomplete atom records
- **Mitigation**:
  - Implement robust error handling
  - Padding/imputation strategies
  - Data cleaning pipeline
  - Document problematic samples

### Medium Priority Risks

**5. Computational Resources**
- **Risk**: GPU memory constraints, long training times
- **Mitigation**:
  - Small batch sizes (4)
  - Mixed precision training
  - Gradient accumulation
  - Use of cloud GPUs if needed

**6. Time Management**
- **Risk**: Feature engineering may take longer than expected
- **Mitigation**:
  - Start early on Week 5-6 tasks
  - Have simple fallback features
  - Prioritize: U-Net > CNN > multi-channel
  - Regular progress tracking

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ U-Net classifier with density + P-P distance features
- ✅ 3D CNN baseline for comparison
- ✅ Classification accuracy > 70%
- ✅ Complete paper following ACM format
- ✅ Reproducible code with documentation

### Target Goals
- 🎯 All three PDB features implemented and tested
- 🎯 Multi-channel U-Net with component masks
- 🎯 Classification accuracy > 85%
- 🎯 Statistical comparison of models
- 🎯 Ablation studies showing feature importance
- 🎯 Publication-quality paper

### Stretch Goals
- 🚀 Ensemble method combining U-Net + CNN
- 🚀 Additional features (secondary structure, solvent accessibility)
- 🚀 Attention mechanisms
- 🚀 Comparison with existing methods from literature
- 🚀 Paper submission to ML/bioinformatics conference

---

## Dependencies & Prerequisites

### Technical Dependencies
- Python 3.8+
- PyTorch 2.0+ with CUDA
- BioPython
- mrcfile
- NumPy, SciPy, pandas
- scikit-learn
- matplotlib, seaborn

### Knowledge Prerequisites
- 3D deep learning architectures
- Structural biology basics (RNA, cryo-EM)
- PyTorch training loops
- Scientific paper writing
- LaTeX/ACM format

### External Dependencies
- Access to dataset (already obtained ✓)
- GPU for training (local or cloud)
- ACM paper template
- Reference papers for related work

---

## Weekly Time Allocation

**Recommended Hours**: 10-15 hours/week

**Distribution**:
- Literature review: 2-3 hours/week (Weeks 3-4)
- Implementation: 6-8 hours/week (Weeks 5-9)
- Experimentation: 8-10 hours/week (Weeks 11-12)
- Paper writing: 10-15 hours/week (Weeks 14-15)
- Code documentation: 2-3 hours/week (ongoing)

---

## Communication & Documentation

### Regular Tasks
- **Weekly**: Commit code to GitHub with descriptive messages
- **Bi-weekly**: Update roadmap with progress
- **Throughout**: Document experiments in lab notebook
- **Ongoing**: Take notes for paper sections

### Key Documents to Maintain
- `docs/progress-log.md` - Weekly progress notes
- `experiments/experiment-log.md` - All experiment configurations and results
- `docs/related-work.md` - Literature review notes
- Paper draft in Overleaf/LaTeX (start Week 10)

---

## Final Checklist (Week 16)

### Before Submission
- [ ] Paper PDF is exactly ACM format (double-check template)
- [ ] All figures are high resolution (300 DPI)
- [ ] All references properly formatted (no Wikipedia)
- [ ] Code runs on fresh environment (test clean install)
- [ ] README has complete reproduction instructions
- [ ] All files in single .zip package
- [ ] File names are clear and professional
- [ ] Final presentation ready

### Quality Assurance
- [ ] Spell check and grammar check
- [ ] Peer review by colleague
- [ ] Run code reproduction test
- [ ] Verify all claims in paper match results
- [ ] Check all figure captions and table headers
- [ ] Ensure ethical use of citations

---

## Post-Submission Opportunities

If the project goes well:
- Submit to ML/bioinformatics conference (instructor may support travel)
- Open-source the code on GitHub with proper documentation
- Write blog post about the work
- Present at lab meetings or seminars
- Extend for journal publication

---

**Last Updated**: November 20, 2025  
**Status**: Active Development  
**Next Major Milestone**: Week 5-6 Feature Engineering

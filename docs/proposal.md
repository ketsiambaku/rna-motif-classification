Project Proposal:

Goal: Given a density map segment in .mrc file and its features extracted from the deposited structure segment in a .pdb file, we want to predict the motif type of the segmented map. It can only fall into one motif class. We have 3 classes: bulge loops, internal loops and hairpin loops.

The two models: 
UNet
Simple 3D CNN

I was able to find some preliminary example code. Which extract the backbone atoms distance from the pdb structures and use it as a feature. The goal is to expand this model by providing 2-3 additional features extracted from the corresponding pdb file.
Those features can be:
original P–P distances (900)
torsion angles. 7 angles per residue. (7x30 = 210)
base pairing matrix (900) using a simple pairing detector. For example pairing detector:
For each residue pair (i,j):
if N1/N3 atoms distance < some distance → paired
else unpaired

We can also have a multi-channel U-Net instead of the single unet found in the example. We can have a ribose mask, a phosphate mask and a base mask of the .mrc map since these constitute the RNA. An example of how to extract these masks can be found in example/label folder. However the code contains some bugs that need to be fixed and it has a ribose and sugar label which is identical. We should have maybe backbone, ribose and base masks. So that logic needs to be corrected to create the masks which can then be passed to the Unet which will have 4-channels. The density map and its 3 masks.

Finally, any other suggestions for features or input channels to expand this example model and optimize the prediction accuracy are welcome. 

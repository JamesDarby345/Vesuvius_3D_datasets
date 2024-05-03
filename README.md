# Vesuvius 3D Datasets ReadMe

## What this repo does
The purpose of this repo is to produce more high quality, complete, 3d voxel, instance labelled data similar to the data tim skinner produced for the Vesuvius Scroll Challenge March Open Source Prize.  
![goodInstanceSeg](https://github.com/JamesDarby345/Vesuvius_3D_datasets/assets/49734270/30cb539e-1cb6-42ce-8182-dd97ca5a5a1c)
A 2d slice of a cube with the borders of the label overlaid that was produced with this repo.

Drive link to some preliminary 256^3 nrrd labels: https://drive.google.com/file/d/1V2uqOTY4xhpMMRTfpVftOv0LLgKCK-Ft/view?usp=drive_link

## Why more of this data is useful
The current best attempt at autosegmentation, Thaumato Anaklpytor relies on a 3d sobel kernel to extract sheet surface points and groups them into sheets with Mask3D. This has shown to be succesful, but I doubt it will generalize to dense scroll regions where nearly every pixel is scroll. The 3D sobel kernel activations would thus be weak.

<img width="707" alt="Dense vs sparse scroll regions" src="https://github.com/JamesDarby345/Vesuvius_3D_datasets/assets/49734270/04f9338d-99b5-4b0e-9a96-a5da4ff49b52">
Dense vs Sparse scroll regions.

Additionally this approach abstracts away all of the sheet texture information when it moves to point clouds. This may be more computationally efficient, but the sheet texture information could be useful for segmentation, espcially in dense regions where the air gap cannot be relied upon to create strong sheet surface signal. Creating voxel based instance labeled data, which this repo aims to do, allows for Supervised Machine Learning approaches to be trained to identify the sheets from the voxels directly. I theorize that this will be one of the paths to successfully autosegment entire scrolls, directly extracting instance segmentations of sheets, even in their dense regions, though many challenges will still need to be overcome for that to work. First among them being all the .obj labels, which this repo relies upon, have come from sparse regions as the manual VC approach doesnt work in dense, complicated scroll regions. 

Anouther possibility is that the representation of the data this repo produces and tuning the parameters, could be directly used to segment the scrolls. But from past experience I doubt that this will ever become robust enough. Though further exploration of graph min-cut, watershed segmentation, other grouping algorithms etc on the SLIC SuperVoxel representation could realistically yield more valuable training data where .obj labels do not exist. 

## How it works 
Currently this code is stilll in a very fast moving, iterative state with new techniques and improvements being tried, adopted, or commented out quickly. But the main ideas are:

1. Use a simple threshold to mask scroll and non-scroll regions, and apply post processing to clean up the semantic (scroll or not) mask. Example output:
![Screenshot 2024-05-03 at 6 33 19 AM](https://github.com/JamesDarby345/Vesuvius_3D_datasets/assets/49734270/45d0a579-1976-4ebf-85a3-6625f564d203)
2. This mask is then used as input to the skimage SLIC superpixel function, using superpixels was inspired by Spelufo's SNIC Superpixel monthly prize, though this is a different algorithm that essentailly does the same thing, but can take in a mask as input. This produces supervoxels of just the scroll sheets. Example Output:
![SLICSuperVoxels](https://github.com/JamesDarby345/Vesuvius_3D_datasets/assets/49734270/a4a90256-798c-4bd5-8a48-0d1862716c29)
3. Next we need to seperate the supervoxels into sheet instances. This is where some more experimentation could be done to do it without using the .obj labels, but I deicided to use them to make it easier. Though using them restricts the approach to areas where the .obj labels are densly specified like the gp region in scroll 1 and the the larger labelled sections of scroll 4. An example of dense .obj labels where the approach can work: 
<img width="254" alt="dense obj labels" src="https://github.com/JamesDarby345/Vesuvius_3D_datasets/assets/49734270/f0b982a2-dca6-4f12-ac8c-0ffa06e47ef2">
The code in voxel utils file is used to calculate rough voxel coordinates of where each .obj wrapping would intersect with the Region of Interest. Visualisation of a 2d slice of .obj intersections:
![ObjInstanceMaskIntersection](https://github.com/JamesDarby345/Vesuvius_3D_datasets/assets/49734270/3693e538-4155-4e4e-8504-66e650ecca3c)
4. The instance based obj intersection mask is used to assign groups of supervoxels roughly into sheets where they intersect. Some issues are that the .obj labels follow the sheet surface and the code to calculate their intersection seems to produce wider labels than it should, so when sheets are close together this causes issues as one .obj label is intersecting with two seperate sheets. And sometimes the .obj labels do wander around off of sheets a bit. More work could be done to address this issue, but for now I am focussing on sparse regions where the sheets are further apart. Visualisation of the supervoxel obj intersection operation:
![Screenshot 2024-05-03 at 6 40 10 AM](https://github.com/JamesDarby345/Vesuvius_3D_datasets/assets/49734270/26d6f859-8f68-4791-a427-f78a0adc4245)
5. Some post processing to assign supervoxels that dont intersect with the .obj label is done and the result is saved as a nrrd file whose name specifies the location of the cube in absolute voxel coordiantes, the block size, the resolution scale, and the scroll it is from. Visualisation of label borders overlaid on the raw data: 
![goodInstanceSeg](https://github.com/JamesDarby345/Vesuvius_3D_datasets/assets/49734270/30cb539e-1cb6-42ce-8182-dd97ca5a5a1c)

Other label:
![otherInstanceSeg](https://github.com/JamesDarby345/Vesuvius_3D_datasets/assets/49734270/e8fbaa11-c852-44f9-8bf6-605ad1d4c99a)

This approach is capable of producing more labelled data, but is not robust at creating the instance segmentation aspect. Though it is suprisingly effective at the semantic scroll or not seperation, though that is a far easier problem. Succesful examples need to be cherry picked to be used as training data. But further development of this approach will increase the regions it is applicable to. Example of a failure case: 
![poorlydoneInstanceSeg](https://github.com/JamesDarby345/Vesuvius_3D_datasets/assets/49734270/ccc349b5-aa0d-4bbb-b71c-5063cb031df6)

Further, as Supervised Machine Learning models are trained with this data, successful instance labels can be verified and then used as more training data, ideally making this approach outdated as the ML model will do a better job. 

May the Scrolls be read.


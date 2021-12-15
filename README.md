![Word Images](IMGUR5K_teaser.png)
## IMGUR5K Handwriting Dataset 
To run the code for downloading the urls and generate corresponding annotations : 

Usage: python download_imgur5k.py --dataset_info_dir <dir_with_annotaion_and_hashes> --output_dir <path_to_store_images>

## Requirements
IMGUR5K download code works with
* Python3
* Numpy
* Requests
* PIL


## Downloading images of IMGUR5K
Run the command and set <path_to_store_images> to the target image directory

## How IMGUR5K download works
The code checks the validity of urls by checking the hash of the url with the groundtruth md5 hash.
If the image is pristine, the annotations are added to the generated annotations file and the respective splits. 

## Full documentation
IMGUR5K is shared as a set of image urls with annotations. This code downloads
the images and verifies the hash to the image to avoid data contamination.
        
**REQUIRED FILES:**
* download_imgur5k.py : Code to download the URLs for the dataset building.
* <dataset_info_dir>/imgur5k_data.lst : File containing URLs with annotations and bounding box
* <dataset_info_dir>/imgur5k_hashes.lst : File containins URL indexes with groundtruth md5 hash.
* <dataset_info_dir>/train_index_ids.lst : File containins URL indexes belonging to train split.
* <dataset_info_dir>/val_index_ids.lst : File containins URL indexes belonging to val split.
* <dataset_info_dir>/test_index_ids.lst : File containins URL indexes belonging to test split.

**Output:**
* <path_to_store_images>/<index>.jpg : 
	* Images dowloaded to output_dir
* imgur5k_annotations.json : 
	* json file with image annotation mappings -> dowloaded to dataset_info_dir
		* Format: { "index_id" : {indexes}, "index_to_annotation_map" : { annotations ids for an index}, "annotation_id": { each annotation's info } }
		* Annotation ID: bounding_box in (xc,yc,w,h,a) format in absolute floating points coordinates.
			* (xc, yc) is the center of the rotated box
			* (w, h) is the width and height
			* (a) is the angle in degrees counterclockwise.
		* Bounding boxes with '.' mean the annotations were not done for various reasons
* imgur5k_annotations_train.json : 
	* json file with image annotation mappings of TRAIN split only -> dowloaded to dataset_info_dir
* imgur5k_annotations_val.json : 
	* json file with image annotation mappings of VAL split only -> dowloaded to dataset_info_dir
* imgur5k_annotations_test.json : 
	* json file with image annotation mappings of TEST split only -> dowloaded to dataset_info_dir

**[All imgur5k_annotations_*.json's format is similar to the format of imgur5k_annotations.json]**

**NOTE:**
Apart from the ~5K images employed in TextStyleBrush paper, ~4K more images are added to the dataset to foster the research in Handwritten Recognition.

# Statistics
| Description 	  	| Count  |
| :-------------------	| :---	 |
| # Page Images 	| 8,177 |
| # Word Images 	| 230,573|
| # Lexicons (case-sensitive) 	| 49,317|

The ratio for train/val/test splits is 80\%:10\%:10\% at the level of page images and the details are provided in the respective json files created as part of the output.

**Disclaimer:**
The dataset is provided using public links to each image, and the availability of these images is controlled by IMGUR or the original user (who uploaded it).

# Contribution
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
IMGUR5K is Creative Commons Attribution-NonCommercial 4.0 International Public licensed, as found in the LICENSE file.
	
## Citation
If you find this data useful, please consider citing our paper: 

Praveen Krishnan, Rama Kovvuri, Guan Pang, Boris Vassilev and Tal Hassner, [TextStyleBrush: Transfer of Text Aesthetics from a Single Example](https://arxiv.org/abs/2106.08385), arXiv: 2106.08385 2021.
```
@misc{krishnan2021textstylebrush,
      title={TextStyleBrush: Transfer of Text Aesthetics from a Single Example}, 
      author={Praveen Krishnan and Rama Kovvuri and Guan Pang and Boris Vassilev and Tal Hassner},
      year={2021},
      eprint={2106.08385},
      archivePrefix={arXiv},
}
```

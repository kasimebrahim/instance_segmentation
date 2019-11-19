# Instance Segmentation for Contract Documents

Semantic structure identification model for general contract documents using [MASK_RCNN](https://arxiv.org/abs/1703.06870). Generates JSON file containing semantic structures of each component in a document.

Semantic labels are below:
* Title
* Subtitle
* Paragraph
* Footnotes
* Header
* Footer
* Page
* Signature

## Installation
---------------
<pre>
  git clone https://github.com/kasimebrahim/instance_segmentation
  cd instance_segmentation
</pre>

### Option 1
<pre>
  # Open conda_environment.yml and change the last line to your own conda environment path.
  conda env create -f conda_environment.yml
</pre>

### Option 2
<pre>
  pip install -r requirements.txt
</pre>

## Prerequisite 
---------------
You will need python>3.7 and optionally conda>4.7.10

## Getting Started
------------------
### NoteBooks
To inspect and visualize your dataset use [dataset_inspect](https://github.com/kasimebrahim/instance_segmentation/blob/master/dataset_inspect.ipynb).

To evaluate your model or visualize your output use [model_eval](https://github.com/kasimebrahim/instance_segmentation/blob/master/model_eval.ipynb).

To train your model
<pre>
  # If you want to train from a pretrained model.
  python Segmentation.py train --datasets=datasets --log=log --model=models/mask_rcnn_pub_lay_seg_0100.h5
  # If you dont have.
  python Segmentation.py train --datasets=datasets --log=log
  # If you want to pickup from a stoped training.
  python Segmentation.py train --datasets=datasets --log=log --pickup=true
</pre>


To segment your documents
<pre>
  python Segmentation.py segment --model=models/mask_rcnn_doc_seg_0100.h5 --image=infer
  # --image is the directory where your documents to be segmented are stored.
  # Your documents should be stored in a directory in two ways.
  # One: Each documenat is in its own directory and every
  #      page of the document is in the directory.
  #      i:e ds/doc/p01.jpg
  # Two: All pages of all the documents are under one
  #      directory. And every image is named as document_page
  #      concatinated with page name/number.
  #      i:e ds/doc_p01.jpg
</pre>
The out put of the segmented documents will be stored in a json file named "documents.json".

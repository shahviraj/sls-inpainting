# sls-inpainting
Image inpainting of SLS images, and also with CelebA dataset

Instructions:

1. Please make sure that the data folder is placed in the parent directory of this code. Dataset directory should contain 3 folders named `train`, `test` and `valid`.

2. Train a projector by running: 
```
cd projector
source run_plate.sh
```
or, 
```
cd projector
source run_celeb.sh
```
3. Modify the filepath to your trained model and then run this: 
```
cd admm
python update_popmean.py 
```
4. Run the demo:
```
python plate_demo.py
```

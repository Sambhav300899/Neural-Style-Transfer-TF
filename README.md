# Neural-Style-Transform-TF
Neural Style Transfer based on https://arxiv.org/abs/1508.06576 by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

# Setup
## Download the github repo
```bash
git cline https://github.com/Sambhav300899/Neural-Style-Transfer-TF.git
```
### Dependencies
```bash
pip install tensorflow-gpu
pip install keras
pip install numpy
pip install matplotlib
pip install opencv-python
```
# Running style transfer
```bash
python style_transfer.py --style {style_image} --content {content_image} --backbone {vgg19/vgg16} --output {path_of_output_image}
```
To see other optional arguments which can be used use the -h and run the script

# Instance-segmentation-with-Mask-RCNN

This is the unofficial implementation of the paper [Mask-RCNN](https://arxiv.org/abs/1703.06870) using [pytorch-lightning](https://www.pytorchlightning.ai/). We demonstrate Mask RCNN for a subset of COCO dataset - which segments three categories namely people, animals and vehicles.

## Downoad dataset and required modules by running 
```
!gdown 101kiCyfTeGvs1p5nW3UxH9oZJ8ucJ3hx
!gdown 1fnbbHeZ3DZqsXwPDrq9YF3p8vI3PCK1K
!gdown 1YCK8Sfbj_UDp1mnSA6g55hlfcpcSklbv
!gdown 1KIf6jeMPpfvWqqiGqjc6Y4JmGHkbL9_C
!gdown 1h6VQWmbq41cJ9O1WdRalc8iOsox2HCeO 
```


## Setup instructions
It is recommended to create a virtual environment and run

```pip install -r requirements.txt```

## Running the code 
```python main.py```

It is important to note that the main function in `main.py`, trains three modules sequentially (RPNHead ,BoxHead, and MaskHead) -- the different training instances  are specified with appropriate comments. For better results it is recommended to train all three modules together after the provided sequential training regime. 

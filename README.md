# Vision transformer

Our implementation of paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), using [PyTorch](https://pytorch.org/)

Run it on VSCode:

<a href="https://code.visualstudio.com/download">
<img src= "https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white" width=80>
</a>

![Vision transformer](https://cdn-lfs.hf.co/datasets/huggingface/documentation-images/142f1b9c07445fbc30f72e4e8e629a9cf11488a78ca6babc8bdb49c11e42b672?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27vit_architecture.jpg%3B+filename%3D%22vit_architecture.jpg%22%3B&response-content-type=image%2Fjpeg&Expires=1732502749&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczMjUwMjc0OX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy8xNDJmMWI5YzA3NDQ1ZmJjMzBmNzJlNGU4ZTYyOWE5Y2YxMTQ4OGE3OGNhNmJhYmM4YmRiNDljMTFlNDJiNjcyP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=MlB3zQJSG9yNM%7E16wWUiGcN1hxVGdXU6qtD%7ES3Bj9OZdSG2LESoiior4DJKEFTS4dSLh-riufefe6kjjoKohZ-t8B1tafmxwIdp1xZYU-d1EdDrPeW4uFaAltQ%7EVIoHOnhwlP8UsMb8W8v0z672-OlXpzw8SOsaRm4OarCijrj1DPcGeWM%7EChCUgCMtdaRUjT%7EOFbn4myxF3VVeSVrZTvglgm7xdB16WmIzUR2kZIJjpjDw4MviaJDYVyXDW79MlvpX5%7ExRs9pd3IVXdbvmnY3Xlag3zGIreIjYH7rFjyZZ0qDWqun%7EKlqqK8EQ0I8mICwx6U1agyyyP-FZzZvWyRw__&Key-Pair-Id=K3RPWS32NSSJCE)

Author:

- Github: khoitran2003
- Email: anhkhoi246813579@gmail.com

### I. Set up environment

1. Make sure you have installed `Anaconda`. If not yet, see the setup document [here](https://www.anaconda.com/download).

2. Clone this repository: `git clone https://github.com/khoitran2003/ViT`
3. `cd` into `vit` and install dependencies package: `pip install -r requirements_cpu.txt` for CPU or `pip install -r requirements_cuda124.txt` for CUDA 124                               

### II. Set up your dataset.

Create 2 folders `train` and `validation` in the `data` folder (which was created already). Then `Please copy` your images with the corresponding names into these folders.

- `train` folder was used for the training process
- `validation` folder was used for validating training result after each epoch

Structure of these folders.

```
sample_data/
...train/
......class_a/
.........a_image_1.jpg
.........a_image_2.jpg
......class_b/
.........b_image_1.jpg
.........b_image_2.jpg
...val/
......class_a/
.........a_image_3.jpg
.........a_image_4.jpg
......class_b/
.........b_image_3.jpg
.........b_image_4.jpg
```

### III. Train your model by running this command line

We create `train.py` for training model.

```
usage: train.py [-h] [--model MODEL] [--num-classes CLASSES]
                [--patch-size PATH_SIZE] [--lr LEARNING_RATE] [--weight-decay WEIGHT_DECAY]
                [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                [--image-size IMAGE_SIZE]
                [--train-folder TRAIN_FOLDER] [--valid-folder VALID_FOLDER]
                [--model-folder MODEL_FOLDER]

optional arguments:
  -h, --help            
    show this help message and exit

  --model MODEL       
    Type of ViT model, valid option: base, large, huge

  --num-classes CLASSES     
    Number of classes
  
  --patch-size PATH_SIZE
    Size of image patch
  
  --lr LR               
    Learning rate
  
  --batch-size BATCH_SIZE
    Batch size
  
  --epochs EPOCHS       
    Number of training epoch
  
  --image-size IMAGE_SIZE
    Size of input image
  
  --train-folder TRAIN_FOLDER
    Where training data is located
  
  --valid-folder VALID_FOLDER
    Where validation data is located
  
  --model-folder MODEL_FOLDER
    Folder to save trained model
```

There are some `important` arguments for the script you should consider when running it:

- `train-folder`: The folder of training images. If you not specify this argument, the script will use the CIFAR-10 dataset for training.
- `valid-folder`: The folder of validation images
- `num-classes`: The number of your problem classes.
- `batch-size`: The batch size of the dataset
- `lr`: The learning rate of Adam Optimizer
- `model-folder`: Where the model after training saved
- `model`: The type of model you want to train. If you want to train with `base` or `large` or `huge` model, you need to specify `patch-size` argument.

Example:

You want to train a model in 10 epochs with Oxford_pet dataset, 37 classes:

```bash
!python train.py --train-folder ${oxford_pet/train} --valid-folder ${oxford_pet/val} --num-classes 37 --patch-size 16 --image-size 224 --lr 0.0001 --epochs 200
```

After training successfully, your model will be saved to `model-folder` defined before

### IV. Testing model with a new image

We offer a script for testing a model using a new image via a command line:

```bash
python predict.py --model_path ${model_path} --image_path ${image_path} --data_path ${data_path} 
```

where `test_image_path` is the path of your test image, `data_path` is the root training data to find class name

Example:

```bash
python predict.py --model_path ./output/ViTBase_best.pth --image_path ./data/test/cat.2000.jpg --data_path ./oxford_pet/
```
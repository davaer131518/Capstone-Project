# Interior Style Classification 


This project aims to fine-tune various pretrained CNN based models for our task of interior designb classification in Armenia.

## Image classification finetuning pipeline

Firstly we must seperate the data into training and testing based on our classes. We keep the main portion of data in the following order 

```
├──Images
    ├── Classic
    ├── Modern
    ├── Soviet
```

You can run the following command to obtain the train test splits based on a desired ratio

```
python data_make.py --split_ratio 0.1
```

After running the command the hierarchy should look like the following (Note we only copy from the original data and do not modify it, however the traiing is done using only trian/test)
```
├──Images
    ├── Classic
    ├── Modern
    ├── Soviet
    ├── train
    ├── test
```

In order to train a model on our newly created training data please run the following command.

```
python train.py --model_name YOU_MODEL_NAME --num_classes YOU_DATA_CLASSES_NUM --batch_size BUTCH_SIZE --num_epochs NUM_EPOCHS
```

To access all the input options please consult the documentation.
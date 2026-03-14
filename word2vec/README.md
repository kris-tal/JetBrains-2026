### Word2Vec Implementation 
This folder contains a Word2Vec implementation using the CBOW architecture and negative sampling, build with pure NumPy. 

- `model.py`: CBOW class logic;
- `utils.py`: preprocessing, data loading, and Vocabulary management;
- `testing.ipynb`: training loop and result visualization.

#### Dataset:

The model was trained on a sampled subset of the [Text Emotion](https://www.kaggle.com/datasets/prajwalnayakat/text-emotion) dataset from Kaggle.

#### Notes:
* since this implementation doesn't use standard ML libraries, the execution is obviously slower. To keep the training time practical, I've limited the dataset to approximately 15000 rows;

- the training loss demonstrates steady convergence, typically stabilizing around epoch 10;

* the loss on the test set (~3.48) remains comparable to the training loss (~3.44), indicating that the model generalizes well and is not overfitted.
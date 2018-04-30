# NLP-Project
> A Natural Langue Processing Project, classifying comments for an old
> Kaggle Competition

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge 

The data is from said competition, we make no claims to it. Private repo until competition winners announced, or terms
read more closely.

Usage:
    - extract the data.zip into folder 'data'
    - 
usage: ./toxic_comment_cnn.py [-h] [--embedding_dim EMBEDDING_DIM]
                            [--num_filters NUM_FILTERS]
                            [--batch_size BATCH_SIZE]
                            [--num_epochs NUM_EPOCHS]

For adjusting hyperparameters.

optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM, -e EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --num_filters NUM_FILTERS
                        Number of filters per filter size. (default: 250)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs. (default: 1)

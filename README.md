Code I wrote as part of the CMU 10701 Final Project. The task was to train a French to English machine translation model on a Kaggle dataset (https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset/) using multiple approaches. My contributions contained two parts:

1. Implementation and training of a transformer model (thenntransformer_10701project.py, adapted from https://pytorch.org/tutorials/beginner/translation_transformer.html). In addition, I implemented an automatic differentiation-driven hyperparameter optimization (from https://arxiv.org/abs/1909.13371) and tested the effects of starting with different initial conditions on train and test loss (transformer_hyperopt_test.py).

2. Finetuning of llama-2-7b (from https://github.com/facebookresearch/llama) via adapter to improve translation performance. llama__adapter.py finetunes the adapter module while keeping the remaining weights frozen. llama__adapter_inference.py is used to assess model quality via bleu score.

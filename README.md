# Fair Classifier that Abstain

An example to run the code:

```
python main.py --dataset=adult \
     --attribute=sex --fairness_notion=DP \
     --epsilon=0.03 --delta1=0.2 \
     --delta2=0.2 --sigma1=1 --sigma0=1 \
     --sigma=1 --eta1=0 --eta2=0 
```
The output of this code includes a file called "IP_results.npz", contains wn, hn solved by IP, and two folders named "AbstainClassifier" and "hClassifier". These folders contain the trained model states of classifiers used to predict wn and hn, respectively. All the output files will be saved in the directory "ROOT/result/output_folder", where "output_folder" is named using the following information separated by "_":
```
args.dataset,
args.attribute,
args.fairness_notion,
args.epsilon,
args.delta1,
args.delta2,
args.sigma1,
args.sigma0,
args.sigma,
args.eta1,
args.eta2,
args.sample_ratio,
args.patience,
args.lr,
args.batch_size,
args.seed
```

## Requirments
Install all the packages from requirments.txt
```
pip install -r requirements.txt  
```

## Data
<center>

| Dataset       | Train data         | Val data | Test data |
| ------------- |-------------| -----| -----|
| Adult      | 31641 | 7911 | 9888 |

</center>

## Options


### Training Parameters

- `max_epoch`: Maximum number of epochs to run the training for if early stopping does not occur.
- `patience`: The number of epochs to wait for improvement in the loss before stopping the training process.
- `batch_size`: The size of the batch used during training.
- `device`: The device to use for training. Default is "cuda:1".
- `lr`: The learning rate to use during training.
- `sample_ratio`: The ratio of data to be sampled for Integer Programming.
- `seed`: The random seed to use for reproducibility.

### Dataset and Fairness Parameters

- `dataset`: The dataset to be used for training and testing. The options are "adult", "XX", "XX".
- `attribute`: The sensitive attribute to protect. The default for the "adult" dataset is "sex".
- `fairness_notion`: The fairness notion to be used. The default is DP (Differential Privacy). The options are DP, EO (Equalized Odds), and EOs (Equalized Opportunity for a Single Threshold).
- `epsilon`: A fairness measurement that represents the disparity between the two groups.
- `delta1`: The abstention rate allowed for group 1.
- `delta2`: The abstention rate allowed for group 2.
- `sigma1`: The abstention rate disparity allowed for two groups within positive samples. Default 1 means no restriction.
- `sigma0`: The abstention rate disparity allowed for two groups within negative samples. Default 1 means no restriction.
- `sigma`: The abstention rate disparity allowed for two groups. Default 1 means no restriction.
- `eta1`: The error relaxation rate of group 1.
- `eta2`: The error relaxation rate of group 2.



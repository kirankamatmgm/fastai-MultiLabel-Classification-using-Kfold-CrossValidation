# fastai MultiLabel Classification using Kfold Cross Validation

The problem I have considered is Multi Label classification. In addition to having multiple labels in each image, the other challenge in this problem is the existence of rare classes and combinations of different classes. So in this situation normal split or random split doesnt work because you can end up putting rare cases in the validation set and your model will never learn about them. The stratification present in the scikit-learn is also not equipped to deal with multilabel targets. 

I have specifically choosen this problem because we may learn some techniques on the way, which we otherwise would not have thought of.

**There may be better or easy way of doing kfold cross validation but I have done it keeping in mind how to implement using fastai**, so if you know some better way so please mail or tweet the idea, i will try to implement and give you credit.

## Install all the necessary libraries

I am using fastai2 so import that. 



```python
!pip install -q fastai2
```

### Cross Validation

Cross-validation, how I see it, is the idea of minimizing randomness from one split by makings n folds, each fold containing train and validation splits. You train the model on each fold, so you have n models. Then you take average predictions from all models, which supposedly give us more confidence in results.
These we will see in following code. I found iterative-stratification package that provides scikit-learn compatible cross validators with stratification for multilabel data.

**My opinion**: 

---

In my opinion it's more important to make one right split, especially because CV takes n times more to train. Then why did I do it??

I wanted to explore classification using cross validation using fastai, which I didn't find many resources to learn. So if I write this blog it may help people.

fastai has no cross validation split(may be) in their library to work like other functions they provide. It may be because cross validation takes time, so may be it not that useful.

But still in this condition I feel its worth exploring using fastai.









so what is **stratification**??

The splitting of data into folds may be governed by criteria such as ensuring that each fold has the same proportion of observations with a given categorical value, such as the class outcome value. This is called stratified cross-validation


```python
!pip install -q iterative-stratification
```


```python
from fastai2.vision.all import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
```

Here dataset is of Zero to GANs - Human Protein Classification inclass jovian.ml hosted competition


```python
path = Path('../input/jovian-pytorch-z2g/Human protein atlas')

train_df = pd.read_csv(path/'train.csv')

train_df['Image'] = train_df['Image'].apply(str) + ".png"

train_df['Image'] = "../input/jovian-pytorch-z2g/Human protein atlas/train/" + train_df['Image']

train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/train/19567.png</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/train/29993.png</td>
      <td>6 4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/train/17186.png</td>
      <td>1 4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/train/29600.png</td>
      <td>6 2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/train/701.png</td>
      <td>3 4</td>
    </tr>
  </tbody>
</table>
</div>



The method I use here is if we have column called fold and with fold number it would be helpfull to split data using that.

fastai has IndexSplitter in datablock api so this would be helpful.




```python
strat_kfold = MultilabelStratifiedKFold(n_splits=3, random_state=42, shuffle=True)
train_df['fold'] = -1
for i, (_, test_index) in enumerate(strat_kfold.split(train_df.Image.values, train_df.iloc[:,1:].values)):
    train_df.iloc[test_index, -1] = i
train_df.head()
```

    /opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py:71: FutureWarning: Pass shuffle=True, random_state=42 as keyword args. From version 0.25 passing these as positional arguments will result in an error
      FutureWarning)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image</th>
      <th>Label</th>
      <th>fold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/train/19567.png</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/train/29993.png</td>
      <td>6 4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/train/17186.png</td>
      <td>1 4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/train/29600.png</td>
      <td>6 2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/train/701.png</td>
      <td>3 4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.fold.value_counts().plot.bar();
```


![png](fastai_MultiLabel_Classification_using_Kfold_CrossValidation_files/fastai_MultiLabel_Classification_using_Kfold_CrossValidation_15_0.png)


## DataBlock 

now that data is in dataframe and also folds are also defined for cross validation, we will build dataloaders, for which we will use datablock.

If you want to learn how fastai datablock see my blog series [Make code Simple with DataBlock api](https://kirankamath.netlify.app/blog/fastais-datablock-api/)

we will create a function get_data to create dataloader.

get_data uses fold to split data to be used for cross validation using IndexSplitter. 
for multiLabel problem compared to single only extra thing to be done is to add MultiCategoryBlock in blocks, this is how fastai makes it easy to work.


```python
def get_data(fold=0, size=224,bs=32):
    return DataBlock(blocks=(ImageBlock,MultiCategoryBlock),
                       get_x=ColReader(0),
                       get_y=ColReader(1, label_delim=' '),
                       splitter=IndexSplitter(train_df[train_df.fold == fold].index),
                       item_tfms=[FlipItem(p=0.5),Resize(512,method='pad')],
                   batch_tfms=[*aug_transforms(size=size,do_flip=True, flip_vert=True, max_rotate=180.0, max_lighting=0.6,max_warp=0.1, p_affine=0.75, p_lighting=0.75,xtra_tfms=[RandomErasing(p=0.5,sh=0.1, min_aspect=0.2,max_count=2)]),Normalize],
                      ).dataloaders(train_df, bs=bs)
```

## metrics

Since this is multi label problem normal accuracy function wont work, so we have accuracy_multi. fastai has this which we can directly use in metrics but I wanted to know how that works so took code of it.


```python
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()
```

F_score is way of evaluation for this competition so used this.


```python
def F_score(output, label, threshold=0.2, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)
```

## Gathering test set


```python
test_df = pd.read_csv('../input/jovian-pytorch-z2g/submission.csv')
tstpng = test_df.copy()
tstpng['Image'] = tstpng['Image'].apply(str) + ".png"
tstpng['Image'] = "../input/jovian-pytorch-z2g/Human protein atlas/test/" + tstpng['Image']
tstpng.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/test/24117.png</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/test/15322.png</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/test/14546.png</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/test/8079.png</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../input/jovian-pytorch-z2g/Human protein atlas/test/13192.png</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Training

I have used technique called mixup, its a data augmentation technique. 

In fastai Mixup is callback, and
this Callback is used to apply MixUp data augmentation to your training.
to know more read [this](http://dev.fast.ai/callback.mixup)

I have tried this first time, but this technique didnot improve my result in this problem. It usually improves accuracy after 80 epochs but I have trained for 20 epoches. so there was no difference in accuracy without it. so you can ignore this. 

But to know about how mixup works is good, I will separate blog on this, so follow my twitter for updates.


```python
mixup = MixUp(0.3)
```

gc is for garbage collection


```python
import gc
```

I have created 3 folds where I simply get the data from a particular fold, create a model, add metrics, I have used resnet34.
And that's the whole training process. I just trained model on each fold and saved predictions for the test set.

I have used a technique called progressive resizing. 

this is very simple: start training using small images, and end training using large images. Spending most of the epochs training with small images, helps training complete much faster. Completing training using large images makes the final accuracy much higher. this approach is called progressive resizing.

we should use the `fine_tune` method after we resize our images to get our model to learn to do something a little bit different from what it has learned to do before. 

I have used `cbs=EarlyStoppingCallback(monitor='valid_loss')` so that model doesnot overfit.

append all prediction to list so that we use it later.

I have run the model for less epochs to see code works and show result, or stopped model in between(it took so much time)

This method gave me F_score of `.77` and accuracy of `>91%` so you can try.

My Purpose here is to write blog and explain how to approach and how code works.

If GPU is out of memory delete learner and empty cuda cache done in last line of code.


```python
all_preds = []

for i in range(3):
    dls = get_data(i,256,64)
    learn = cnn_learner(dls, resnet34, metrics=[partial(accuracy_multi, thresh=0.2),partial(F_score, threshold=0.2)],cbs=mixup).to_fp16()
    learn.fit_one_cycle(10, cbs=EarlyStoppingCallback(monitor='valid_loss'))
    learn.dls = get_data(i,512,32)
    learn.fine_tune(10,cbs=EarlyStoppingCallback(monitor='valid_loss'))
    tst_dl = learn.dls.test_dl(tstpng)
    preds, _ = learn.get_preds(dl=tst_dl)
    all_preds.append(preds)
    del learn
    torch.cuda.empty_cache()
    gc.collect()
```

    Downloading: "https://download.pytorch.org/models/resnet34-333f7ec4.pth" to /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth



    HBox(children=(FloatProgress(value=0.0, max=87306240.0), HTML(value='')))


    



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>F_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.545962</td>
      <td>0.344541</td>
      <td>0.796662</td>
      <td>0.325269</td>
      <td>03:34</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>F_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.383005</td>
      <td>0.320621</td>
      <td>0.838335</td>
      <td>0.343538</td>
      <td>03:31</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.364457</td>
      <td>0.307868</td>
      <td>0.834389</td>
      <td>0.395574</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.339153</td>
      <td>0.288834</td>
      <td>0.851154</td>
      <td>0.424101</td>
      <td>03:30</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.321231</td>
      <td>0.276498</td>
      <td>0.860605</td>
      <td>0.500714</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.307015</td>
      <td>0.262529</td>
      <td>0.865237</td>
      <td>0.526143</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.303211</td>
      <td>0.252637</td>
      <td>0.869339</td>
      <td>0.542609</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.293627</td>
      <td>0.246137</td>
      <td>0.870898</td>
      <td>0.557622</td>
      <td>03:28</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.288911</td>
      <td>0.240489</td>
      <td>0.877324</td>
      <td>0.567245</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.285248</td>
      <td>0.237112</td>
      <td>0.876575</td>
      <td>0.589500</td>
      <td>03:28</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.286031</td>
      <td>0.236534</td>
      <td>0.877339</td>
      <td>0.590382</td>
      <td>03:27</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>F_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.290949</td>
      <td>0.315076</td>
      <td>0.849766</td>
      <td>0.462833</td>
      <td>04:23</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.295474</td>
      <td>0.320369</td>
      <td>0.826154</td>
      <td>0.472169</td>
      <td>04:22</td>
    </tr>
  </tbody>
</table>


    No improvement since epoch 0: early stopping







<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>F_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.554367</td>
      <td>0.340495</td>
      <td>0.808827</td>
      <td>0.318601</td>
      <td>03:22</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>F_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.388593</td>
      <td>0.319564</td>
      <td>0.840362</td>
      <td>0.331533</td>
      <td>03:32</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.366437</td>
      <td>0.307622</td>
      <td>0.839379</td>
      <td>0.363208</td>
      <td>03:28</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.343576</td>
      <td>0.290248</td>
      <td>0.840876</td>
      <td>0.402298</td>
      <td>03:27</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.325350</td>
      <td>0.281427</td>
      <td>0.846164</td>
      <td>0.467070</td>
      <td>03:28</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.314482</td>
      <td>0.266823</td>
      <td>0.859451</td>
      <td>0.469782</td>
      <td>03:27</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.305747</td>
      <td>0.257310</td>
      <td>0.865830</td>
      <td>0.517662</td>
      <td>03:29</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.295284</td>
      <td>0.248361</td>
      <td>0.871881</td>
      <td>0.546668</td>
      <td>03:27</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.287226</td>
      <td>0.242251</td>
      <td>0.874298</td>
      <td>0.563557</td>
      <td>03:27</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.288487</td>
      <td>0.239550</td>
      <td>0.878041</td>
      <td>0.570435</td>
      <td>03:27</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.284914</td>
      <td>0.238145</td>
      <td>0.879196</td>
      <td>0.570799</td>
      <td>03:27</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>F_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.294379</td>
      <td>0.344038</td>
      <td>0.817124</td>
      <td>0.550201</td>
      <td>04:22</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.296755</td>
      <td>0.330813</td>
      <td>0.830053</td>
      <td>0.424398</td>
      <td>04:20</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.295352</td>
      <td>0.260328</td>
      <td>0.866110</td>
      <td>0.477913</td>
      <td>04:23</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.280896</td>
      <td>0.259668</td>
      <td>0.857954</td>
      <td>0.598846</td>
      <td>04:19</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.273494</td>
      <td>0.218607</td>
      <td>0.878868</td>
      <td>0.645609</td>
      <td>04:22</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.259204</td>
      <td>0.200595</td>
      <td>0.905396</td>
      <td>0.653470</td>
      <td>04:20</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.247184</td>
      <td>0.209090</td>
      <td>0.889878</td>
      <td>0.687422</td>
      <td>04:19</td>
    </tr>
  </tbody>
</table>


    No improvement since epoch 5: early stopping








    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00<00:00]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>F_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='0' class='' max='200' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/200 00:00<00:00]
    </div>



stack all the prediction stored in list and average the values.


```python
subm = pd.read_csv("../input/jovian-pytorch-z2g/submission.csv")
preds = np.mean(np.stack(all_preds), axis=0)
```

You should have list of labels which we get using vocab.


```python
k = dls.vocab
```


```python
preds[0]
```




    array([0.04565947, 0.08774102, 0.0536039 , 0.04304133, 0.9251135 ,
           0.01606368, 0.15841891, 0.02610746, 0.09389433, 0.06638951],
          dtype=float32)



I found threshold of 0.2 works good for my code.

then all the labels predicted above 0.2 are labels of that image using vocab. 


```python
thresh=0.2
labelled_preds = [' '.join([k[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
```

put them in Labels column


```python
test_df['Label']=labelled_preds
```

this step is to submit result to kaggle.


```python
test_df.to_csv('submission.csv',index=False)
```

# WIP

Trying to translate the work from the [MiniPile
paper](https://arxiv.org/abs/2304.08442) into other subdomains. Specifically
this time I am trying to train a CLIP model that is better with less data.

As in the ConvNext2 paper they said:

> The performance of a visual representation learning system is largely
> influenced by three main factors: the neural network architecture chosen, the
> method used for training the network, and the data used for training.

This time around, I am really interested in the data, so that is why I am
currently using the DataComp dataset. I want to extract the soul of this
dataset, what makes it move. I want to be the alchemist with the right amount
of data to make it feel something.

This is the pipeline that i am working with:

```
+----------------------------------+    +----------------------------------+    +----------------------------------+    +------------------------------------+
|           cluster.py             | -> |           classify.py            | -> |    parse_classification.py       | -> |   filter_english_clusters.py       |
|----------------------------------|    |----------------------------------|    |----------------------------------|    |------------------------------------|
| cluster_experiments_elbow.json   |    | cluster_classifications.json     |    | classification_of_clusters.csv   |    |filtered_high_value_english_data.csv|
| cluster_samples.json             |    |                                  |    |                                  |    |                                    |
+----------------------------------+    +----------------------------------+    +----------------------------------+    +------------------------------------+
```

```
python datacomp/resharder.py -i small_datacomp/shards/ -o my_cluster -s subset_file.npy
```
Ok so my final subset was of 1649647 examples but in the end 333125 images in the subset were not found in the input!
So in the end i only had: 1649647 - 333125

So i actually have: 1316522

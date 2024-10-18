# simple-data-free-model-server
 This is an open source implementation of DirectAI's core service, provided both to give back to the open source community that we benefited greatly from, as well as to allow our clients to continue to have service in the event that we can no longer host our API.

We host zero-shot image models to allow clients to use computer vision at scale without having to collect / label lots of training data or train their own model. However, zero-shot models don't necessarily work out of the box for all cases. We introduce an algorithm for providing feedback to zero-shot models by extending the standard linear decision boundary in the model's embedding space into a two-stage nearest neighbors algorithm, which allows for much more fine-tuned control over what the model considers to belong in a particular class with minimal impact on runtime.

### On Pre-Commit Hooks
- Make sure you run `pip install pre-commit` followed by `pre-commit install` before attempting to commit to the repo.

### Launching Production Service
- Set your logging level preference in `directai_fastapi/.env`. See options on [python's logging documentation](https://docs.python.org/3/library/logging.html#levels). An empty string input defaults to `logging.INFO`.
- `docker compose build && docker compose up`

### Launching Integration Tests
- `docker compose -f testing-docker-compose.yml build && docker compose -f testing-docker-compose.yml up`

### Hardware Requirements
This repository is designed to require access to an Nvidia GPU with Ampere architecture. The Ampere architecture is used by the flash attention integration in the object detector. However, it could be modified to run on older Nvidia GPUs or on CPU. Feel free to submit a pull request or raise an issue if you need that support!

### Running Offline Batch Classification
We've built infrastructure to make it easy to quickly run an arbitrary classifier against a dataset. If your images are organized like so:

    /dataset_directory
    │
    ├── image1.jpg
    ├── image2.jpg
    ├── image3.jpg
    ├── ...
    └── imageN.jpg
and you have a JSON file defining the image classifier you'd like to run at `classifier_config.json`, you can dump classification labels to an `output.csv` via:

 - `docker-compose build && docker-compose run local_fastapi
   python classify_directory.py --root=dataset_directory
   --classifier_json_file=classifier_config.json --output_file=output.csv`

Make sure that all the files are mounted within the Docker container. You can do that by either modifying the volumes specified in `docker-compose.yml`, or by placing them all within the `.cache` directory which is mounted by default.

If your images have labels and are organized like so:

    /dataset_directory
    │
    ├── /label1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    ├── /label2
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    └── /labelN
        ├── image1.jpg
        ├── image2.jpg
        └── ...
You can run an evaluation against the labels by running the command

 - `docker compose build && docker compose run local_fastapi
   python classify_directory.py --root=dataset_directory
   --classifier_json_file=classifier_config.json --eval_only=True`

If you want to run classifications on a custom dataset, you can either use our API or build a custom Ray Dataset and use the utilities defined in `batch_processing.py`.

### A Quick Start for Self-Hosting on AWS
To launch a self-hosted version of this service in AWS, we'll spin up a fresh EC2 instance. Choose "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4 (Ubuntu 22.04)" as your AMI and g5.xlarge as your instance size. After that you should be able to just run:

```
git clone https://github.com/DirectAI/simple-data-free-model-server
cd simple-data-free-model-server
docker compose build && docker compose up
```

### Method TLDR
This repository presents the idea of a _semantic nearest neighbors_ for building custom decision boundaries with late-fusion zero-shot models. We use CLIP and OWL-ViT-ST as our base late-stage zero-shot image classifier and object detector, respectively. See the code for implementation details.

##### Image classifier
The standard approach for a late-stage zero-shot image classifier is to take a set of _n_ labels, embed them via the associated language model, and then append those embeddings to generate a linear classification layer on top of the image embedding from the associated image model. This head can also be interpreted as a nearest neighbors layer, as the predicted class is just the class with text embedding most similar to the image embedding.

Let $d(t_i, v) > 0$ be the relevancy function between text embedding $i$ and the image embedding $v$. Then our class prediction for $v$ (denoted $f(v)$) via the traditional approach is:

$f(v) = \text{argmax}_i d(t_i, v)$

We define a _meta class_ which contains both positive and negative examples. A score is computed for the _meta class_ based on a goodness-of-fit estimate for the image given the positive and negative examples in the class. Then, the _meta class_ with the highest score is predicted to be the true class.

In the simplest case of a single positive example per _meta class_, this is the same as the traditional zero-shot approach. To extend the paradigm to the case where there are multiple positive examples per _meta class_, we can run the traditional zero-shot approach over the set of all positive examples and then let the predicted class be the _meta class_ which includes the highest-scoring positive example. This can be viewed as an _n_-way nearest neighbors problem where the samples are semantically-relevant text embeddings, and the correct class is the one with the most semantically-relevant example.

Let $m(i)$ be the _meta class_ associated with sample $i$. Then we have:

$f(v) = m(\text{argmax}_i d(t_i, v))$

We can reinterpret the above as a two-stage process. Instead of running an _n_-way nearest neighbors problem, we can take for each _meta class_ the max of the relevancy scores of the provided examples, and then take the argmax over this _meta class_ level score.

Let $S_j$ refer to the samples associated with the $j$th _meta class_. Then:

$f(v) = \text{argmax}_j \text{max}_{i \in S_j} d(t_i, v)$

To extend this to allow for negative examples, we can replace the first stage max with a two-class nearest neighbors boundary between the positive and negative examples. Then, we can let our _meta class_ score be the score of the most relevant example if that example is positive, and the negative score if that example is negative. In other words, a _meta class_ that has more relevant positive examples will result in a higher score, and a _meta class_ with more relevant negative examples will result in a lower score.

Let $P_j$ and $N_j$ refer to the positive and negative examples for _meta class_ $j$, respectively. In other words, if $t_i \in P_j$, then the $i$th text example is a positive example for _meta class_ $j$. Then, let $t_j$ refer to the example in $P_j \cup N_j$ that is most relevant, and let $\hat{d}(j, v)$ be the result of the two-class nearest neighbor problem run for each _meta class_ $j$. We have:

$t_j = \text{argmax}_{t_i \in P_j \cup N_j} d(t_i, v)$
$\hat{d}(j, v) = d(t_j, v)$ if $t_j \in P_j$ else $-d(t_j, v)$
$f(v) = \text{argmax}_j \hat{d}(j, v)$

Note that if there are no negative examples, this is the same as the previous case, and if there are no negative examples and exactly one positive example per _meta class_ this is equivalent to the traditional method. Also note that if the most relevant example for all of the _meta classes_ is a negative example, then this function attempts to choose the 'least irrelevant' prediction.

This is our final prediction function, a two-layered nearest neighbors problem that incorporates positive and negative semantic evidence into its decision boundary. This can be run efficiently with an optimized scatter max function.

##### Object Detector

To extend late-fusion zero-shot object detectors to be able to incorporate positive and negative examples, we first compute for each _meta class_ and each proposed bounding box the $\hat{d}(j, v)$ as defined above. However, this is not sufficient to incorporate negative samples into the prediction, as an object may have overlapping bounding boxes, some of which score highly on the negative examples for a class while others score highly on the positive examples. To address this, we extend $\hat{d}$ to include NMS between neighboring boxes. As we'll be running NMS many times over the same set of boxes with this approach, we pre-compute the IoU graph and reuse the same graph on each NMS instance. We then run a final NMS between _meta classes_ with the per-class boxes that survived being suppressed by the NMS with the negative examples, and then threshold on detection confidence as usual.

In other words, we first run a two-way object detection problem for each _meta class_ between its positive and negative examples, and then run an $n$-way object detection problem between the survivors from the previous problem for each _meta class_. This can be done efficiently by using an optimized scatter max function and by caching the IoU graph for reuse between NMS subproblems. In the absence of any negative examples, this is the same as assigning each box's score for a _meta class_ the max of the relevancy scores for each positive example belonging to that _meta class_ and then running NMS. If there are no negative examples and exactly one positive example per class, this is the same as the traditional method.

### On Pre-Commit Hooks
- Make sure you run `pip install pre-commit` followed by `pre-commit install` before attempting to commit to the repo.

### Acknowledgements
Special thanks to [Neo](https://neo.com) for funding DirectAI! Thank you to [OpenCLIP](https://github.com/mlfoundations/open_clip) and [Huggingface](https://huggingface.co) for providing the model implementations that we use here.

### Contact
If you have any questions or comments, raise an issue or reach out to Isaac Robinson at isaac@directai.io.

### Contributing
If you're interested in contributing, raise an issue or email Isaac and we'll write a contributing guide!

### Citing
If you find this useful for your work, please cite this repository!
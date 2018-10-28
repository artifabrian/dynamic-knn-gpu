***

# GPU based Dynamic k-Nearest Neighbours

I built a relatively simple implementation of a Dynamic k-NN using TensorFlow with GPU support. 

![knn](https://raw.githubusercontent.com/artifabrian/dynamic-knn-gpu/master/knn.png)
from [datacamp](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn)

>In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, >with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, >typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

from [wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

A dynamic k-NN allows for some parameters (such as the k parameter) to be defined on a per instance basis, instead of being fixed in the model.
This can be useful in cases where the k is influenceed by some independent variable or whenever you need to add new training instances without requiring a fit step to your model.

Note: running TensorFlow on GPU is formally supported for only NVIDIA cards (CUDA backend).

### Benchmark Results

For more detailed information abuot the benchmark check out the [notebook with the experimental information.](https://github.com/artifabrian/dynamic-knn-gpu/blob/master/notebooks-exploration/dynamic-knn-experimental.ipynb)

These were done on a laptop with a GTX 870m, the results could've been even better with a faster GPU.

| TF GPU speedup over TF CPU: |
|-----------------------------|
| 1.4x  |


On large datasets, the GPU gains are more noticeable, however, even on a laptop, with a low powered GPU, the GPU performance gain can be around 40%.
   
| TF GPU speedup over Scikit CPU (dynamic k): |
|---------------------------------|
| 50x        |


As seen above, the results are quite favorable towards the k-NN implementation using TensorFlow with GPU support.

In the scenario where the k is static or when no new training instances are introduced this implementation of k-NN using TensorFlow with GPU support can be around 20% slower than the Scikit implementation (assuming you are using their efficient implementation with Ball Tree). However, this is also the worst scenario for this model and with a better GPU this difference might decrease.

In any case, it's impressive that we can obtain such results for such a simple implementation!

### Dependencies
* [TensorFlow](https://www.tensorflow.org/) >= 1.11

### Instructions

Run example with command:  
```
python example.py
```

### Author

[João Leal](http://www.joao-leal.com/)

### Related research:
- [An Improved k-NN Classification with Dynamic k](https://www.researchgate.net/publication/317595386_An_Improved_k-NN_Classification_with_Dynamic_k)

- [Adaptive k-Nearest-Neighbor Classification Using a Dynamic Number of Nearest Neighbors](https://link.springer.com/chapter/10.1007/978-3-540-75185-4_7)

- [Dynamic K-Nearest-Neighbor with Distance and attribute weighted for classification](https://ieeexplore.ieee.org/document/5559858?reload=true)


### Code License

Code is licensed under the Apache License 2.0  
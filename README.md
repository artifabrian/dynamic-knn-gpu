***

# GPU based Dynamic k-Nearest Neighbours

I built a relatively simple implementation of a Dynamic k-NN using TensorFlow with GPU support. (it can also run on the CPU)

>In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, >with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, >typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

from [wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

A dynamic k-NN allows for some parameters (such as the k parameter) to be defined on a per instance basis, instead of being fixed in the model.
This can be useful in cases where the k is influenceed by some independent variable.

Note: running TensorFlow on GPU is formally supported for only NVIDIA cards (CUDA backend).

### Benchmark Results

| TF GPU speedup over TF CPU: |
|-----------------------------|
| 1.4x  |


On large datasets, the GPU gains are more noticeable, however, even on a laptop, with a low powered GPU, the GPU performance gain can be around 40%.
   
| TF GPU speedup over Scikit CPU: |
|---------------------------------|
| 50x        |


As seen above, the results are quite favorable towards the k-NN implementation using TensorFlow with GPU support.
This makes it a great solution if we are dealing with any problem where a dynamic k value is needed.
But what about when the k is static?

| TF GPU speedup over Scikit CPU (using static k): |
|--------------------------------------------------|
| 0.7x                      |


In this scenario, this implementation of k-NN using TensorFlow with GPU support is actually around 20-30% slower than the Scikit implementation.
However, this is also the worst scenario for this implementation and with a better GPU this difference might decrease. 
In any case, it's impressive that we can obtain such results for such a naíve/simple implementation and there's a lot of room for improvement!

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
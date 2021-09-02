# Convolutional Exponential


## Literature

**[Glow: Generative flow with invertible 1x1 convolutions](https://arxiv.org/abs/1807.03039)** - Kingma & Dhariwal (2018)

> Original paper showcasing a 1x1 Invertible Convolution

* **Problem** 1: It's invertible but it is not orthogonal. So it violates some of the problems with the original RBIG construction and the relationship between the KLD and information.
* **Problem** 2: I don't think it account for the spatial variability. I could be wrong but it may only be channel-wise. They mentioned it in the *convolutional exponential paper*.

---

**[Emerging convolutions for generative normalizing flows](https://arxiv.org/abs/1901.11137)** - Hoogeboom et al (2019)

> A paper that creating an orthogonal convolutional which **does** account for the spatial characteristics. They also improve the original 1x1Conv Method by parameterized it with householder rotations. So it makes it correspond with RBIG convergence specifics.

* **Problem** 1: It is fast for evaluating densities, but it is slow for generating samples. It uses a sort of autoregressive approach.

---


**[The Convolution Exponential and Generalized Sylvester Flows](https://arxiv.org/abs/2006.01910)** - Hoogeboom et al (2020)

> They create the *exponential* convolution transform. It basically uses the exponential of the convolutions. It approximates the transform via a Taylor series. It generalizes most of the householder approaches, i.e. Sylvester Flows.

* **Problem** 1: I saw in the code that they did some spectral normalization which greatly increases the complexity of the code. I don't understand if it is necessary for our application or maybe it's only for Dequantization.

---

# 🎤 PRESENTATION SCRIPT: Project Defense (April 23rd)

## Slide 1: Title Slide
**Speaker:** 
"Good morning, everyone. Today, I am excited to present our work on 'High-Fidelity CT Reconstruction: Bridging Physics and Deep Learning'. Our core mission was to investigate how we can fundamentally improve medical imaging by removing the reliance on legacy algorithms that induce visual artifacts, ultimately moving toward direct raw-data deep learning."

## Slide 2: The Core Problem - FBP Artifacts
**Speaker:**
"To understand our contribution, we must first look at the legacy standard: Filtered Back Projection, or FBP. FBP relies on strict mathematical assumptions. When a patient receives a low-dose CT scan to minimize radiation risk, the resulting raw data—called a sinogram—becomes highly noisy and sparse. 
Because FBP maps this noise globally across the image, it creates severe streaking artifacts. Current State-of-the-Art Deep Learning approaches try to fix this by post-processing the already corrupted FBP images. We recognized this as conceptually flawed: why task a Neural Network with un-learning artifacts when we possess the raw source data?"

## Slide 3: The Threat to End-to-End Deep Learning
**Speaker:**
"Our immediate next thought was: 'Let's just feed the raw sinogram into a standard Deep Learning Vision model like a U-Net.' 
However, this defies the physics of the Radon Transform. A single physical point in a patient's body manifests as an entire sine wave across the sinogram data. Standard Convolutions are local—they look at small 3x3 pixel grids. They cannot geometrically connect a sine wave stretching across an entire image. Attempting to force a standard CNN to learn this geometry results in massive parameter explosion, making the model untrainable and inherently unstable."

## Slide 4: Our Dual-Track Solution Strategy
**Speaker:**
"To solve this, we initiated a two-pronged strategy. 
Option 'A' was the Learned Primal-Dual algorithm, the theoretical holy grail that completely embeds physical forward and backward physics operators into the neural network layers. However, this carries massive structural risks regarding C++ CUDA interoperability. 
Therefore, our primary deliverable for today is Option 'B': The **FreqHybridNet**."

## Slide 5: Understanding FreqHybridNet (Reference Poster Infographic)
**Speaker:**
"As shown in our implementation poster, FreqHybridNet bridges explicit physics with adaptive deep learning in pure, native frameworks. 
1. We take the raw sonogram and perform a 1D Fast Fourier Transform. 
2. Instead of a static mathematical filter, our model utilizes a trainable frequency-domain branch that learns the optimal filtering strategy mapped entirely to our exact noise profile. 
3. We then map this back into the spatial domain and perform geometric backprojection.
4. Only after the global mapping has been resolved by geometry, do we deploy standard CNNs to refine the localized textures."

## Slide 6: Results and Conclusion
**Speaker:**
"By not forcing a local convolutional layer to solve a global geometric physics equation, we achieve structurally solid, artifact-free medical imagery reconstructed entirely from raw data. We proved that we can bypass classical post-processing pipelines while simultaneously de-risking the complex infrastructure dependencies of massive unrolled physical networks. Thank you, and I look forward to your questions."
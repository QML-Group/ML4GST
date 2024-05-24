# ML4QGST: Transformer Models for Quantum Gate Set Tomography

This repository hosts the accompanying software for the following research article. 

### Research article: [Transformer Models for Quantum Gate Set Tomography](https://arxiv.org/abs/2405.02097)

#### Abstract:
Quantum computation represents a promising frontier in the domain of high-performance computing, blending quantum information theory with practical applications to overcome the limitations of classical computation. This study investigates the challenges of manufacturing high-fidelity and scalable quantum processors. Quantum gate set tomography (QGST) is a critical method for characterizing quantum processors and understanding their operational capabilities and limitations. This paper introduces Ml4QGST as a novel approach to QGST by integrating machine learning techniques, specifically utilizing a transformer neural network model. Adapting the transformer model for QGST addresses the computational complexity of modeling quantum systems. Advanced training strategies, including data grouping and curriculum learning, are employed to enhance model performance, demonstrating significant congruence with ground-truth values. We benchmark this training pipeline on the constructed learning model, to successfully perform QGST for $3$ gates on a $1$ qubit system with over-rotation error and depolarizing noise estimation with comparable accuracy to pyGSTi.
This research marks a pioneering step in applying deep neural networks to the complex problem of quantum gate set tomography, showcasing the potential of machine learning to tackle nonlinear tomography challenges in quantum computing.

### Citation:
```
@article{yu2024transformer,
  title={Transformer Models for Quantum Gate Set Tomography},
  author={Yu, King Yiu and Sarkar, Aritra and Ishihara, Ryoichi and Feld, Sebastian},
  journal={arXiv preprint arXiv:2405.02097},
  year={2024}
}
```

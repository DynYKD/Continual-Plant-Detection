# Continual Plant Detection

## Abstract
This paper investigates the problem of class-incremental object detection for agricultural applications where a model needs to learn new plant species and diseases incrementally without forgetting the previously learned ones. We adapt two public datasets to include new categories over time, simulating a more realistic and dynamic scenario. We then compare three class-incremental learning methods that leverage different forms of knowledge distillation to mitigate catastrophic forgetting. Our experiments show that all three methods suffer from catastrophic forgetting, but the Dynamic Y-KD approach, which additionally uses a dynamic architecture that grows new branches to learn new tasks, outperforms ILOD and Faster-ILOD in most settings both on new and old classes. 

These results highlight the challenges and opportunities of continual object detection for agricultural applications. In particular, we hypothesize that the large intra-class and small inter-class variability that is typical of plant images exacerbate the difficulty of learning new categories without interfering with previous knowledge.

# Acknowledgments
Our repository is based on the inspiring work of [MMA](https://github.com/fcdl94/MMA) @fcdl94. We thank the authors and the contibutors for releasing their code.
# Implementation of IMPALA with Distributed Tensorflow


## Information

* These results are from only 32 threads.
* A total of 32 CPUs were used, 4 environments were configured for each game type, and a total of 8 games were learned.
* Tensorflow Implementation
* Use DQN model to inference action
* Use distributed tensorflow to implement Actor
* Training with 1 day
* Same parameter of [paper](https://arxiv.org/abs/1802.01561)
```
start learning rate = 0.0006
end learning rate = 0
learning frame = 1e6
gradient clip norm = 40
trajectory = 20
batch size = 32
reward clipping = -1 ~ 1
```


## Dependency

```
tensorflow==1.14.0
gym[atari]
numpy
tensorboardX
opencv-python
```

## Overall Schema

<div align="center">
  <img src="source/image2.png" width="50%" height='300'>
</div>

## Model Architecture

<div align="center">
  <img src="source/model_architecture.png" width="100%" height='300'>
</div>

## How to Run

* show [start.sh](start.sh)
* Learning 8 types of games at a time, one of which uses 4 environments.

## Result

### Video

|||||
|:---:|:---:|:---:|:---:|
| ![Breakout](source/breakout/openaigym.video.0.25421.video001440.gif) | ![Pong](source/pong/openaigym.video.0.20974.video000440.gif) | ![Seaquest](source/seaquest/openaigym.video.0.27728.video001090.gif) | ![Space-Invader](source/spaceinvader/openaigym.video.0.30111.video004130.gif) |
| Breakout | Pong | Seaquest | Space-Invader |
| ![Boxing](source/boxing/openaigym.video.0.3092.video000930.gif) | ![Star-Gunner](source/gunner/openaigym.video.0.11850.video001760.gif) | ![KungFu](source/kungfu/openaigym.video.0.22816.video000740.gif)|![Demon](source/demon/openaigym.video.0.31852.video000310.gif) |
| Boxing | Star-Gunner | Kung-Fu | Demon |

### Plotting

![abs_one](source/multitaskrecurrent/result1.png)
![abs_one](source/multitaskrecurrent/result2.png)


## Compare reward clipping method

### Video

|||
|:---:|:---:|
| ![abs_one](source/gunner/openaigym.video.0.11850.video001760.gif) | ![Pong](source/reward_clipping/asyn_gunner.gif) |
| abs_one | soft_asymmetric |

### Plotting
||
|:---:|
| ![abs_one](source/gunner/gunner_1.png)
![abs_one](source/gunner/gunner_2.png) |
| abs_one |
| ![soft_asymmetric](source/reward_clipping/asyn_1.png)
![soft_asymmetric](source/reward_clipping/asyn_2.png) |
| soft_asymmetric |

## Is Attention Really Working?

|||
|:---:|:---:|
| ![abs_one](source/attention/attention_1.gif) |

* Above Blocks are ignored.
* Ball and Bar are attentioned.
* Empty space are attentioned because of less trained.

# Todo

- [x] Only CPU Training method
- [x] Distributed tensorflow
- [x] Model fix for preventing collapsed
- [x] Reward Clipping Experiment
- [x] Parameter copying from global learner
- [x] Add Relational Reinforcement Learning
- [x] Add Action information to Model
- [x] Multi Task Learning
- [x] Add Recurrent Model
- [ ] Training on GPU, Inference on CPU

# Reference

1. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
2. [deepmind/scalable_agent](https://github.com/deepmind/scalable_agent)
3. [Asynchronous_Advatnage_Actor_Critic](https://github.com/alphastarkor/distributed_tensorflow_a3c)

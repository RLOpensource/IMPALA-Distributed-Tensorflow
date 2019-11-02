# Implementation of IMPALA with Distributed Tensorflow


## Information

* These results are from only 20 threads.
* Tensorflow Implementation
* Use distributed tensorflow to implement Actor
* Training with 2 hours
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

## Schema

<div align="center">
  <img src="source/image1.png" width="50%" height='300'>
</div>

## How to Run

* learner

```shell
python trainer.py --num_actors=20 --task=0 --batch_size=32 --queue_size=128 --trajectory=20 --learning_frame=1000000000 --start_learning=0.0006 --end_learning=0.0 --discount_factor=0.99 --entropy_coef=0.05 --baseline_loss_coef=1.0 --gradient_clip_norm=40.0 --job_name=learner --reward_clipping=abs_one &
```

* actor

```shell
for i in $(seq 0 19); do
    python trainer.py --num_actors=20 --task=$i --batch_size=32 --queue_size=128 --trajectory=20 --learning_frame=1000000000 --start_learning=0.0006 --end_learning=0.0 --discount_factor=0.99 --entropy_coef=0.05 --baseline_loss_coef=1.0 --gradient_clip_norm=40.0 --job_name=actor --reward_clipping=abs_one &
done;
wait
```

## Result

### Video

<center>
|||||
|:---:|:---:|:---:|:---:|
| ![Breakout](source/breakout/openaigym.video.0.25421.video001440.gif) | ![Pong](source/pong/openaigym.video.0.20974.video000440.gif) | ![Seaquest](source/seaquest/openaigym.video.0.27728.video001090.gif) | ![Space-Invader](source/spaceinvader/openaigym.video.0.30111.video004130.gif) |
| Breakout | Pong | Seaquest | Space-Invader |
| ![Boxing](source/boxing/openaigym.video.0.3092.video000930.gif) | ![Star-Gunner](source/gunner/openaigym.video.0.11850.video001760.gif) | ![KungFu](source/kungfu/openaigym.video.0.22816.video000740.gif)| |
| Boxing | Star-Gunner | Kung-Fu | ... |
</center>

### Plotting

* Breakout
  
<div align="center">
  <img src="source/breakout/breakout_1.png" width="100%" height='150'>
  <img src="source/breakout/breakout_2.png" width="100%" height='150'>
</div>

* Pong
  
<div align="center">
  <img src="source/pong/pong_2.png" width="100%" height='150'>
  <img src="source/pong/pong_1.png" width="100%" height='150'>
</div>

* Seaquest

<div align="center">
  <img src="source/seaquest/seaquest_1.png" width="100%" height="150">
  <img src="source/seaquest/seaquest_2.png" width="100%" height="150">
</div>

* Space Invader

<div align="center">
  <img src="source/spaceinvader/invader_1.png" width="100%" height="150">
  <img src="source/spaceinvader/invader_2.png" width="100%" height="150">
</div>

* Boxing

<div align="center">
  <img src="source/boxing/boxing_1.png" width="100%" height="150">
  <img src="source/boxing/boxing_2.png" width="100%" height="150">
</div>

* Gunner

<div align="center">
  <img src="source/gunner/gunner_1.png" width="100%" height="150">
  <img src="source/gunner/gunner_2.png" width="100%" height="150">
</div>

* KungFu

<div align="center">
  <img src="source/kungfu/kungfu_1.png" width="100%" height="150">
  <img src="source/kungfu/kungfu_2.png" width="100%" height="150">
</div>


# Todo

- [x] Only CPU Training method
- [x] Distributed tensorflow
- [x] Model fix for preventing collapsed
- [ ] Training on GPU, Inference on CPU

# Reference

1. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
2. [deepmind/scalable_agent](https://github.com/deepmind/scalable_agent)
3. [Asynchronous_Advatnage_Actor_Critic](https://github.com/alphastarkor/distributed_tensorflow_a3c)

Learning to Drive Smoothly in Minutes in Carla Simulator
===
For simple and plain virtual environment like donkey simulator, or simple and plain real environment like wayve experiment, car can both learn to drive smoothly in minutes. And I want to apply this technology in the complex and vivid virtual environment like Carla simulator.

schedule
----
| features                                      | status        |
| --------                                      | :-----:       |
| PSPNet: semantic segmentation                 | Complete      |
| VAE: abstract features                        | Complete      |
| SAC: RL                                       | Complete      |
| PSPNet + VAE: abstract useful features        | Ongoing       |
| PSPNet + VAE + SAC: fast RL                   | Ongoing       |
| YOLO: identity pedestrian, car, guidepost...  | TODO          |
| PSPNet + VAE + SAC + YOLO                     | TODO          |

performance
---
TODO

requirement
---
Carla server  
anaconda  
mpi4py: need install mpi in win10  
gym  
stable-baselines  
opencv-python  
pygame  
imgaug  

Reference
---
Carla simulator: https://github.com/carla-simulator/carla & https://github.com/bitsauce/Carla-ppo  
PSPNet: https://github.com/meetshah1995/pytorch-semseg  
VAE & SAC: https://github.com/araffin/learning-to-drive-in-5-minutes  
YOLO: https://github.com/pjreddie/darknet & https://pjreddie.com/darknet/yolo/  


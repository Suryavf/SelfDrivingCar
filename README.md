Self-driving car
================

End-To-End models for autonomous driving. Imitation learning model trained with <a href="https://github.com/carla-simulator/imitation-learning">CoRL2017</a>. The reinforcement learning agents were trained in the <a href="https://github.com/carla-simulator/carla">CARLA</a> environment:

## Necessary Packages
* Pytorch  1.2.0
* Numpy    1.16.4
* ImgAug   0.2.9 
* tsnecuda 2.1.0
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install tsnecuda cuda101 -c cannylab
conda install -c anaconda numpy
conda install -c anaconda pandas
conda install -c anaconda opencv
conda install -c anaconda scipy
conda install -c anaconda h5p
pip install matplotlib
pip install imgaug
pip install tqdm
pip install einops
pip install ipython
pip install tensorboard
```

## CARLA 0.9.6
<a href="https://github.com/carla-simulator/carla">CARLA</a> environment instructions, all credits to <a href="https://github.com/dianchen96">dianchen96</a>. SOURCE: <a href="https://github.com/dianchen96/LearningByCheating">link</a>.

CARLA download:
```
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
mkdir carla
tar -xvzf CARLA_0.9.6.tar.gz -C carla
cd carla
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town01.bin
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town02.bin
mv Town*.bin CarlaUE4/Content/Carla/Maps/Nav/
```

CARLA installation:
```
cd PythonAPI/carla/dist
rm carla-0.9.6-py3.5-linux-x86_64.egg
wget http://www.cs.utexas.edu/~dchen/lbc_release/egg/carla-0.9.6-py3.5-linux-x86_64.egg
easy_install carla-0.9.6-py3.5-linux-x86_64.egg
```

## Execution
Basic exectution:
```
$ python main.py
```
Or select execution mode:
```
$ python main.py --mode train
```
Mode options: train, continue, plot, play

### Data path
**trainpath** and **validpath** should point to where the <a href="https://github.com/carla-simulator/imitation-learning">CoRL2017</a> dataset located.
```
$ python main.py --trainpath ./data/h5file/SeqTrain/
                 --validpath ./data/h5file/SeqVal/
                 --savedpath ./Saved
```

### Train settings
Basic training settings. More options in `config.py`.
```
$ python main.py --mode       train
                 --model      Kim2017
                 --n_epoch    150
                 --batch_size 120
                 --optimizer  Adam
                 --scheduler  True
```

Check the training log through tensorboard.
```
$ tensorboard --logdir runs
```

You can continue the training with:
```
$ python main.py --mode continue --name 1910141754
```


## References

1. Codevilla, F., Miiller, M., López, A., Koltun, V., & Dosovitskiy, A. (2018). <a href="https://arxiv.org/pdf/1710.02410">End-to-end driving via conditional imitation learning</a>. In 2018 IEEE International Conference on Robotics and Automation (ICRA) (pp. 1-9). 
2. Codevilla, F., Santana, E., López, A. M., & Gaidon, A. (2019). <a href="https://arxiv.org/pdf/1904.08980">Exploring the Limitations of Behavior Cloning for Autonomous Driving</a>. arXiv preprint arXiv:1904.08980.
3. Kim, J., & Canny, J. (2017). <a href="http://openaccess.thecvf.com/content_ICCV_2017/papers/Kim_Interpretable_Learning_for_ICCV_2017_paper.pdf">Interpretable learning for self-driving cars by visualizing causal attention</a>. In Proceedings of the IEEE international conference on computer vision (pp. 2942-2950).
4. Kim, J., Misu, T., Chen, Y. T., Tawari, A., & Canny, J. (2019). <a href="http://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Grounding_Human-To-Vehicle_Advice_for_Self-Driving_Vehicles_CVPR_2019_paper.pdf">Grounding Human-To-Vehicle Advice for Self-Driving Vehicles</a>. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 10591-10599).
5. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). <a href="https://arxiv.org/pdf/1511.05952">Prioritized experience replay</a>. arXiv preprint arXiv:1511.05952.
6. Liang, X., Wang, T., Yang, L., & Xing, E. (2018). <a href="http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaodan_Liang_CIRL_Controllable_Imitative_ECCV_2018_paper.pdf">CIRL: Controllable imitative reinforcement learning for vision-based self-driving</a>. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 584-599).

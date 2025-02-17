<<<<<<< HEAD
# Bayesian_optimization
Peer review on Bayesian optimization which was part of my coursework CSE5835 at University of Connecticut
=======
# Benchmarking

Project Name: Benchmarking the Performance of Bayesian Optimization across Multiple Experimental Materials Science Domains

## Authors
||                    |
| ------------- | ------------------------------ |
| **AUTHORS**      | Harry Qiaohao Liang     | 
| **VERSION**      | 2.0 / Aug, 2021     | 
| **EMAILS**      | hqliang@mit.edu | 
||                    |


Abstract: 
In the field of machine learning (ML) for materials optimization, active learning algorithms, such as Bayesian Optimization (BO), have been leveraged for guiding autonomous and high-throughput experimentation systems. However, very few studies have evaluated the efficiency of BO as a general optimization algorithm across a broad range of experimental materials science domains. In this work, we evaluate the performance of BO algorithms with a collection of surrogate model and acquisition function pairs across five diverse experimental materials systems, namely carbon nanotube polymer blends, silver nanoparticles, lead-halide perovskites, as well as additively manufactured polymer structures and shapes. By defining acceleration and enhancement metrics for general materials optimization objectives, we find that Gaussian Process (GP) with anisotropic kernels (automatic relevance detection, ARD) and Random Forests (RF) have comparable performance in BO as surrogate models, and both outperform the commonly used GP with isotropic kernels. While GP with anisotropic kernel has shown to be more robust as a surrogate model across most design spaces, RF is a close alternative and warrants more consideration because of it being free of distribution assumptions, having lower time complexities, and requiring less effort in initial hyperparameter selection. We also raise awareness about the benefits of using GP with anisotropic kernels over GP with isotropic kernels in future materials optimization campaigns.

GitHub Repo: https://github.com/PV-Lab/Benchmarking

Collaborators: Aldair Gongora, Danny Zekun Ren, Armi Tiihonen, etc.

Status: Published in npj Computational Materials (2021).
See PDF at: https://rdcu.be/cByoD


## Attribution
This work is under BSD-2-Clause License. Please, acknowledge use of this work with the appropiate citation to the repository and research article.

## Citation 

Liang, Q., Gongora, A.E., Ren, Z. et al. Benchmarking the performance of Bayesian optimization across multiple experimental materials science domains. npj Comput Mater 7, 188 (2021). https://doi.org/10.1038/s41524-021-00656-9
    
## Usage

run `Example use of framework with GP type surrogate models.ipynb` or `Example use of framework with RF type surrogate models.ipynb` with any of the given datasets to benchmark the performance of a selected BO algorithm using pool-based active learning framework. 
run `Manifold Visualization.ipynb` with any of the given datasets to visualize its design space. 
run `Performance Visualization.ipynb` with given demo benchmarking results or any benchmarking results from running the framework locally.

The package contains the following module and scripts:

| Module | Description |
| ------------- | ------------------------------ |
| `Example use of framework with GP type surrogate models.ipynb`      | Script for running framework      |
| `Example use of framework with RF type surrogate models.ipynb`      | Script for running framework       |
| `Manifold Visualization.ipynb`      | Script for visualizing materials dataset design spaces   |
| `Performance Visualization.ipynb`      | Script for visualizing performance   |
| `requirements.txt`      | Required packages   |




## Datasets
**For reuse for code and materials datasets in this repo, please cite both this study and the following authors for sharing their datasets.**

Materials datasets used to benchmark BO performance in this repository are provided by:

(1) Crossed barrel dataset

    @article{gongora2020bayesian,
      title={A Bayesian experimental autonomous researcher for mechanical design},
      author={Gongora, Aldair E and Xu, Bowen and Perry, Wyatt and Okoye, Chika and Riley, Patrick and Reyes, Kristofer G and Morgan, Elise F and Brown, Keith A},
      journal={Science advances},
      volume={6},
      number={15},
      pages={eaaz1708},
      year={2020},
      publisher={American Association for the Advancement of Science}
    }
    
    link: https://advances.sciencemag.org/content/6/15/eaaz1708
    
(2) Perovskite dataset
     
     @article{sun2021data,
       title={A data fusion approach to optimize compositional stability of halide perovskites},
       author={Sun, Shijing and Tiihonen, Armi and Oviedo, Felipe and Liu, Zhe and Thapa, Janak and Zhao, Yicheng and Hartono, Noor Titan P and Goyal, Anuj and Heumueller, Thomas and Batali, Clio and others},
       journal={Matter},
       volume={4},
       number={4},
       pages={1305--1322},
       year={2021},
       publisher={Elsevier}
     }
     
     link: https://www.sciencedirect.com/science/article/pii/S2590238521000084
     
(3) AutoAM dataset

     @article{deneault2021toward,
       title={Toward autonomous additive manufacturing: Bayesian optimization on a 3D printer},
       author={Deneault, James R and Chang, Jorge and Myung, Jay and Hooper, Daylond and Armstrong, Andrew and Pitt, Mark and Maruyama, Benji},
       journal={MRS Bulletin},
       pages={1--10},
       year={2021},    
       publisher={Springer}
     }
     
     link: https://link.springer.com/article/10.1557/s43577-021-00051-1
     
(4) P3HT/CNT dataset

    @article{bash2021multi,
    title={Multi-Fidelity High-Throughput Optimization of Electrical Conductivity in P3HT-CNT Composites},
    author={Bash, Daniil and Cai, Yongqiang and Chellappan, Vijila and Wong, Swee Liang and Yang, Xu and Kumar, Pawan and Tan, Jin Da and Abutaha, Anas and Cheng, Jayce JW and Lim, Yee-Fun and others},
    journal={Advanced Functional Materials},
    pages={2102606},
    year={2021},
    publisher={Wiley Online Library}
    }
    
    link: https://onlinelibrary.wiley.com/doi/abs/10.1002/adfm.202102606
    
(5) AgNP dataset

    @article{mekki2021two,
      title={Two-step machine learning enables optimized nanoparticle synthesis},
      author={Mekki-Berrada, Flore and Ren, Zekun and Huang, Tan and Wong, Wai Kuan and Zheng, Fang and Xie, Jiaxun and Tian, Isaac Parker Siyu and Jayavelu, Senthilnath and Mahfoud, Zackaria and Bash, Daniil and others},
      journal={npj Computational Materials},
      volume={7},
      number={1},
      pages={1--10},
      year={2021},
      publisher={Nature Publishing Group}
    }
    
    link: https://www.nature.com/articles/s41524-021-00520-w
    
    








>>>>>>> a36ebd9 (Peer review on Bayesian optimization which was part of my coursework CSE5835 at University of Connecticut)

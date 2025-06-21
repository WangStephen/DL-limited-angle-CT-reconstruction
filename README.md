# Project Title

Deep Learning for Limited-angle CT Reconstruction

**Master Thesis for MSc Data Science and Machine Learning at UCL**

**Research Assistant Funded by A\*STAR Singapore**

## Reference

This work is published on EMBC **(Oral)** by [Wang, Y. 2020](https://ieeexplore.ieee.org/document/9176040):

````bash
Wang, Y., Yang, T. and Huang, W., 2020, July. Limited-Angle Computed Tomography Reconstruction using Combined FDK-Based Neural Network and U-Net. In 2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) (pp. 1572-1575). IEEE.
````

For reference use:

````bash
@inproceedings{wang2020limited,
  title={Limited-Angle Computed Tomography Reconstruction using Combined FDK-Based Neural Network and U-Net},
  author={Wang, Yiying and Yang, Tao and Huang, Weimin},
  booktitle={2020 42nd Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={1572--1575},
  year={2020},
  organization={IEEE}
}
````

## Instructions

The folder '3Dircadb1' contains the data for our project which is a public resource that can be downloaded from [here](https://www.ircad.fr/research/3d-ircadb-01/).

The 'data_preprocessing' folder contains the necessary tools for CT preprocessing, sinograms generation and reconstrution via the FDK algorithm from any specific angular range.

The folder 'train_models' contains a series of models including the baseline model proposed by [Syben et al. (2019)](https://arxiv.org/abs/1904.13342), and models based on this basline with additional networks in projection and reconstruction domain. The reconstructed CT results are stored in each model foler as well as the numerical evaluation results. The best training session is also saved for each model.

An overview of files tree in this project is shown below:

```bash
Limited-Angle-CT-Reconstruction-Code
├── 3Dircadb1 [Containing all the CT samples]
│   ├── 3Dircadb1.1 [Sample one]
│   │   └── PATIENT_DICOM [Containing all the slices for sample one in DICOM format]
│   .
│   .
│   .
│   ├── 3Dircadb1.20 [Sample twenty]
│   │   └── PATIENT_DICOM [Containing all the slices for sample twenty in DICOM format]
├── data_preprocessing
│   ├── ct_preprocessing.py
│   ├── forward_projection.py [Generating training data]
│   ├── fdk_back_projection.py [Generating training ground truth]
│   ├── recon_proj_parameters.py [Parameters set for projection]
│   ├── normalized_ct_phantoms [Containing the CT data after preprocessing]
│   ├── recon_145 [Reconstructed CT results from 145 degrees via the FDK algorithm]
│   ├── recon_180 [Reconstructed CT results from 180 degrees via the FDK algorithm]
│   ├── recon_360 [Reconstructed CT results from 360 degrees via the FDK algorithm]
│   └── sinograms [Generated projection data from preprocessed CT]
└── train_models
    ├── compare_results_figures [Several stored figures for comparing the results of these models]
    ├── cnn_projection_model [the FDK NN trained with CNN in projection domain together]
    │   ├── eval_recon [reconstructed CT images for test data from this model]
    │   ├── eval_result [numerical results for the performance of this model on test data]
    │   ├── model_cnn_projection.py [Code for this model]
    │   └── saved_session [Saved session of the best performance for this model]
    ├── cnn_reconstruction_model [the FDK NN trained with CNN in reconstruction domain together]
    │   ├── eval_recon
    │   ├── eval_result
    │   ├── model_cnn_reconstruction.py
    │   └── saved_session
    ├── dense_cnn_reconstruction_model [the FDK NN trained with dense CNN in reconstruction domain together]
    │   ├── eval_recon
    │   ├── eval_result
    │   ├── model_dense_cnn_reconstruction.py
    │   └── saved_session
    ├── unet_projection_model [the FDK NN trained with U-Net in projection domain together]
    │   ├── eval_recon
    │   ├── eval_result
    │   ├── model_unet_projection.py
    │   └── saved_session
    ├── unet_proposed_reconstruction_model [the FDK NN trained with proposed U-Net in reconstruction domain together]
    │   ├── eval_recon
    │   ├── eval_result
    │   ├── model_unet_proposed_reconstruction.py
    │   └── saved_session
    ├── combined_projection_reconstruction_model [the FDK NN trained with NN in both domains together]
    │   ├── eval_recon
    │   ├── eval_result
    │   ├── model_combined_proj_recon.py
    │   └── saved_session
    ├── pure_fdk_model [the FDK algorithm without any networks]
    │   ├── eval_recon
    │   └── eval_result
    ├── fdk_nn_model [the FDK neural network (the FDK NN )]
    │   ├── eval_recon
    │   ├── eval_result
    │   ├── model_fdk_nn.py
    │   └── saved_session
    ├── train_models.py [To train models with the FDK NN and NN together]
    ├── train_slices_model.py [To train the model with the FDK NN and U-Net separately (The Best Approach)]
    ├── evaluation.py [Evaluate the model performance]
    ├── load_data.py [Load training, validation and test data for models]
    ├── geometry_parameters.py [Parameters set for training]
    ├── Models_Performance [A brief conclusion of performance on the models with the FDK NN and NN together]
    └── unet_reconstruction_model 
        ├── eval_recon
        │   ├── generation_recons [The reconstructed CT from the FDK NN used for later U-Net training]
        │   ├── slices_batch_10_lev4_bott256 [Training results for the best approach]
        ├── eval_result
        ├── model_unet_reconstruction.py [the FDK NN trained with U-Net in reconstruction domain together]
        ├── model_unet_slices.py [The model with the FDK NN and U-Net trained separately (The best Approach)]
        └── saved_session
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.






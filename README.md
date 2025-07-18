# MHIAIFormer
# MHIAIFormer: Multihead Interacted and Adaptive Integrated Transformer With Spatial-Spectral Attention for Hyperspectral Image Classification
## Abstract:
Deep learning is an effective method for hyperspectral image (HSI) classification, where CNN-based and Transformer-based meth-ods have achieved excellent performance. However, there are some drawbacks to the existing CNN-based and Transformer-based HSI classification approaches: 1) CNN-based methods are deficient in showing the extraction of multiscale features and lo-calized features owing to the fixed-size input patch. 2) the MHSA module ignores the interaction capability between multiple atten-tion heads, which leads to insufficient feature fusion in various directions. 3) The weights of attention heads in various directions are disregarded in the MHSA and attention heads are simply concatenated horizontally. To address the above limitations, a novel Multi-Head Interacted and Adaptive Integrated Trans-former (MHIAIFormer) with Spatial-Spectral Attention which integrates the respective advantages of convolutions and trans-formers is proposed in this study. A pyramidal spatial-spectral attention feature extraction (PS2A) module is adopted to efficient-ly capture the localized and multiscale feature information of HSI. The output of PS2A is then sent to the transformer encoder stage through a grouped multiscale cross-dimension (GMCD) embed-ding module, which includes additive self-attention using multi-head interaction and MHSA with adaptive multi-head merging to capture the long-range dependencies of the features. Extensive experiments on four datasets verify that our proposed approach achieves more satisfactory classification accuracy when compared with state-of-the-art models. The overall accuracy (OA) of the proposed model achieved 95.97%, 98.68%, 92.68%, and 99.49% on four datasets. The source code is available at https://github.com/Delon1364/MHIAIFormer.

![image](https://github.com/Delon1364/MHIAIFormer/assets/55641779/84738fb4-51e5-4ec6-b83f-452e0c3edd7f)
Fig. 1. Architecture of the proposed MHIAIFormer for HSI classification. (a) Overall pipeline. (b) Illustration of the GMCD, 

## Citation
Please kindly cite the papers if this code is useful and helpful for your research.

D. Kong, J. Zhang, S. Zhang, X. Yu and F. A. Prodhan, "MHIAIFormer: Multihead Interacted and Adaptive Integrated Transformer With Spatial-Spectral Attention for Hyperspectral Image Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 17, pp. 14486-14501, 2024, doi: 10.1109/JSTARS.2024.3441111.

  
    @ARTICLE{10632582,
      author={Kong, Delong and Zhang, Jiahua and Zhang, Shichao and Yu, Xiang and Prodhan, Foyez Ahmed},
      journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
      title={MHIAIFormer: Multihead Interacted and Adaptive Integrated Transformer With Spatial-Spectral Attention for Hyperspectral Image Classification}, 
      year={2024},
      volume={17},
      number={},
      pages={14486-14501},
      keywords={Transformers;Feature extraction;Data mining;Attention mechanisms;Adaptation models;Deep learning;Hyperspectral imaging;Image classification;Deep learning (DL);hyperspectral image (HSI) classification;multihead interacted and adaptive integrated transformer (MHIAIFormer);multihead self-attention (MHSA)},
      doi={10.1109/JSTARS.2024.3441111}}

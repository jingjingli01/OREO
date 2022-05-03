This repository contains the code and resources of our AAAI 2022 paper [Text Revision by On-the-Fly Representation Optimization](https://arxiv.org/pdf/2204.07359.pdf)



# Datasets & Models 

To request the GYAFC corpus, please follow the instructions [here](https://github.com/raosudha89/GYAFC-corpus). 

To request the Newsela-auto dataset, please follow the instructions [here](https://github.com/chaojiang06/wiki-auto) to obtain the access first. Newsela-turk can be requested from [here](https://github.com/mounicam/controllable_simplification).

After you get the access to the corpus, please email me <llee.jingjing@gmail.com> to get the preprocessed version.



The implementation of multi-task roberta is based on [huggingface transformers](https://github.com/huggingface/transformers). We provide the fine-tuned models for text simplicity and text formalization [here](https://drive.google.com/drive/folders/1SkihgkKxu6LG2GBXmyKAiYMInshU343g?usp=sharing) .

```bash
models
|-- simplicity
|   `-- for_inference
`-- formality
    |-- for_evaluation
    `-- for_inference
```



# Text revise
## Text simplification
Inference 
```bash
python3 text_revise.py --rbt_path path_to_simplicity_model_for_inference --cls_thld 0.30 --infile input_file  --outfile output_file --attribute formality 
```

The evaluation scripts for simplification is based on [this repository](https://github.com/chaojiang06/wiki-auto/tree/41e0e7f60c6216abc6e1bbeb573b44190b01291b/simplification/system_output/metrics).

Evaluation 
```bash
python3 simp_eval/metrics.py --complex complex_file  --reference path_to_reference_files  --simplified output_file
```

## Text formalization
Inference 
```bash
python3 text_revise.py --rbt_path path_to_formality_model_for_inference  --cls_thld 0.50 --infile input_file  --outfile output_file --attribute simplicity 
```


Evaluation
```bash
python3 formal_eval/metrics.py --ref_dir path_to_reference_files --cls_model path_to_formality_model_for_evaluation --batch_size 512 --result_file output_file
```
# Citation
Please cite if you use the above resources for your research
```
@article{li2022text,
  title={Text Revision by On-the-Fly Representation Optimization},
  author={Li, Jingjing and Li, Zichao and Ge, Tao and King, Irwin and Lyu, Michael R},
  journal={arXiv preprint arXiv:2204.07359},
  year={2022}
}
```


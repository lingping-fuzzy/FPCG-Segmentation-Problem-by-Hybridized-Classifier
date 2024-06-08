# FPCG-Segmentation-Problem-by-Hybridized-Classifier

This is the source code for paper [Analysis on Fetal Phonocardiography Segmentation Problem by Hybridized Classifier](https://) \\


Table of Contents:
- [The main results](#the-main-results)
        - [Data](#data)

- [Understanding the repository](#understanding-the-repository)
    - [Running scripts](#running-scripts)



### Data

(***License:** we do not impose any new license restrictions in addition to the original licenses of the used dataset.
See the paper to learn about the dataset sources*)

Navigate to the repository root and run the following commands:
```
\url{https://physionet.org/content/challenge-2016/1.0.0/}
\url{https://ag-datasets-89f203ac-44ed-4a06-9395-1e069e8e662d.s3-us-west-2.amazonaws.com/springer_dataset.mat} from paper citation 12
paper [cited](https://github.com/alvgaona/heart-sounds-segmentation/tree/matlab?tab=readme-ov-file)
```
The synthetic data will be available here:
```
\url{https://drive.google.com/drive/folders/1LJCS5XZb7yZG3VLqGIva-XvRnL4MuDSn?usp=drive_link}
```

After that, the `data/` directory should appear.

## Tutorial

Here, we reproduce the results for.... to be continued

# Understanding the repository

Read this if you are going to do more experiments/research in this repository.

## Code overview
- `main` contains scripts which produce the main results
    - main_pcg_signal_classification.py
    - main_pcg_Tree_classification.py
    - main_combination.py
    - Models
        - `nets\` is the "main" network setup inside the paper
        - `.py` to be finished--- later

    - data
        - `signal.py` how to create signal data
        - `mixSignal.py` how to create numpy or torch version of signal data from .mat file.

- `others` to be continued...


## Running scripts

For most scripts in `main` folder, the pattern is as follows:

```
python some_script.py conf/*.json
```

When the run is successfully finished, the result will be the `out/method/*` folder.
In particular, the `out/(logs, results/checkpoints)` folder will be created.
Usually, the main part of the result is the `out/method/result/*.txt` file.



# How to cite<!-- omit in toc -->

```
@article{kong2024analysis,
  title={Analysis on fetal phonocardiography segmentation problem by hybridized classifier},
  author={Kong, Lingping and Barnova, Katerina and Jaros, Rene and Mirjalili, Seyedali and Snasel, Vaclav and Pan, Jeng-Shyang and Martinek, Radek},
  journal={Engineering Applications of Artificial Intelligence},
  volume={135},
  pages={108621},
  year={2024},
  publisher={Elsevier}
}t
```

# contacts

if you have questions, please email to the author lingping_kong@yahoo.com

# AI for City Districts: Measuring City Districts’ Sustainability using Satellite Data

This repository contains codes and data for performing city index prediction (regression) task.

Original dataset contains raw 1302 images of 4800 x 4800 resolution (14 GB). 
- Raw data and processed dataset can be downloaded at our [Hugging Face dataset link](https://huggingface.co/datasets/issai/SCI_prediction).
- Models (both pre-trained, and fine-tuned) can be accessed at [Hugging Face models link](https://huggingface.co/issai/SCI_prediction_models).

Dataset consists of 45 cities from various locations, and mostly chosen from [Arcadis Index 2022](https://www.arcadis.com/en/knowledge-hub/perspectives/global/sustainable-cities-index). Cities and their corrresponding index values from 9 different sustainability ranking systems which were used for model training:

| City IATA Code | Overall Arcadis SCI | Planet Arcadis SCI | People Arcadis SCI | Profit Arcadis SCI | Sustainable cities by Corporate Knights | Resilient Cities by Grosvenor | Global Cities by AT Kearney | European Green City Index | US and Canada Green City Index |
|----------------|---------------------|--------------------|--------------------|--------------------|-----------------------------------------|-------------------------------|-----------------------------|---------------------------|--------------------------------|
| ALA            | -                   | -                  | -                  | -                  | -                                       | -                             | 118                         | -                         | -                              |
| ESB            | -                   | -                  | -                  | -                  | -                                       | -                             | 86                          | -                         | -                              |
| NQZ            | -                   | -                  | -                  | -                  | -                                       | -                             | 128                         | -                         | -                              |
| GYD            | -                   | -                  | -                  | -                  | -                                       | -                             | -                           | -                         | -                              |
| BKK            | 72                  | 92                 | 58                 | 73                 | -                                       | -                             | 35                          | -                         | -                              |
| PEK            | 73                  | 91                 | 71                 | 53                 | 30                                      | 39                            | 6                           | -                         | -                              |
| FRU            | -                   | -                  | -                  | -                  | -                                       | -                             | -                           | -                         | -                              |
| BOG            | 78                  | 20                 | 82                 | 82                 | 36                                      | -                             | 63                          | -                         | -                              |
| BOS            | 22                  | 54                 | 54                 | 3                  | -                                       | 7                             | 21                          | -                         | 6                              |
| BNE            | 64                  | 60                 | 57                 | 50                 | -                                       | 27                            | -                           | -                         | -                              |
| AEP            | 82                  | 62                 | 84                 | 84                 | 35                                      | 36                            | 32                          | -                         | -                              |
| CAI            | 86                  | 89                 | 79                 | 91                 | -                                       | 48                            | 59                          | -                         | -                              |
| CHI            | 52                  | 67                 | 70                 | 8                  | -                                       |                               | 8                           | -                         | 11                             |
| DUB            | 37                  | 28                 | 19                 | 65                 | -                                       | 29                            | 45                          | 21                        | -                              |
| HAN            | 85                  | 93                 | 80                 | 85                 | -                                       | -                             | -                           | -                         | -                              |
| HKG            | 63                  | 56                 | 65                 | 45                 | -                                       | 30                            | 7                           | -                         | -                              |
| IST            | 74                  | 55                 | 74                 | 79                 | 46                                      | -                             | 27                          | 25                        | -                              |
| CGK            | 83                  | 68                 | 81                 | 86                 | -                                       | 49                            | 67                          | -                         | -                              |
| FIH            | 100                 | 99                 | 95                 | 100                | -                                       | -                             | 136                         | -                         | -                              |
| KUL            | 71                  | 73                 | 62                 | 69                 | -                                       | -                             | -                           | -                         | -                              |
| LOS            | 99                  | 88                 | 100                | 99                 | 40                                      | -                             | 113                         | -                         | -                              |
| LHE            | 94                  | 95                 | 85                 | 97                 | -                                       | -                             | 127                         | -                         | -                              |
| LIS            | 57                  | 24                 | 56                 | 66                 | -                                       | -                             | 46                          | 18                        | -                              |
| MNL            | 93                  | 83                 | 97                 | 89                 | -                                       | 47                            | 69                          | -                         | -                              |
| MEL            | 60                  | 50                 | 61                 | 43                 | -                                       | -                             | 12                          | -                         | -                              |
| MEX            | 79                  | 53                 | 83                 | 77                 | -                                       | 44                            | 31                          | -                         | -                              |
| MIL            | 51                  | 21                 | 39                 | 71                 | -                                       | 33                            | 44                          | -                         | -                              |
| BOM            | 91                  | 81                 | 89                 | 96                 | 44                                      | 46                            | 62                          | -                         | -                              |
| MUC            | 19                  | 12                 | 25                 | 27                 | -                                       | 24                            | 26                          | -                         | -                              |
| NBO            | 96                  | 82                 | 98                 | 95                 | -                                       | -                             | 89                          | -                         | -                              |
| OSL            | 1                   | 1                  | 17                 | 39                 | 2                                       | -                             | 54                          | 3                         | -                              |
| PAR            | 8                   | 2                  | 43                 | 31                 | 17                                      | 23                            | 3                           | 10                        | -                              |
| RIX            | 44                  | 18                 | 30                 | 62                 | -                                       | -                             | -                           | 15                        | -                              |
| SFO            | 9                   | 35                 | 38                 | 4                  | 16                                      | 16                            | 11                          | -                         | 1                              |
| GRU            | 84                  | 44                 | 94                 | 78                 | 42                                      | 41                            | 40                          | -                         | -                              |
| ICN            | 26                  | 43                 | 4                  | 44                 | 25                                      | 35                            | 17                          | -                         | -                              |
| CIT            | -                   | -                  | -                  | -                  | -                                       | -                             | -                           | -                         | -                              |
| SIN            | 35                  | 69                 | 5                  | 28                 | 45                                      | 32                            | 9                           | -                         | -                              |
| SYD            | 33                  | 42                 | 15                 | 46                 | 26                                      | 19                            | 15                          | -                         | -                              |
| TPE            | 46                  | 71                 | 20                 | 29                 | -                                       | 34                            | 49                          | -                         | -                              |
| TAS            | -                   | -                  | -                  | -                  | -                                       | -                             | -                           | -                         | -                              |
| TKY            | 2                   | 7                  | 7                  | 20                 | -                                       | 26                            | 4                           | -                         | -                              |
| YVR            | 17                  | 26                 | 13                 | 30                 | 8                                       | 2                             | 39                          | -                         | 2                              |
| IAD            | 20                  | 45                 | 37                 | 15                 | 24                                      | 9                             | 14                          | -                         | 8                              |




# 📥 Downloading the repository

```
git clone https://github.com/IS2AI/city-classification-and-index-prediction

```

# ⚙️ Pre-processing 

Prior to training it is necessary to perform pre-processing on raw images. To generate patches out of raw images needed for training, and to perform train-val-test split launch the following script:

```
python3 preprocessing.py
```

# ⚡ Training

To launch training for city index predition use:
```
python3 train_regression.py
```

# 🎯 Inference

To test city index prediciton on unseen patches run:
```
python3 test_regression.py
```

# 🌱🗺️ Sustainability map creation

To create sustainability color map, there is available another script:

```
python3 make_sustainability_map.py
```

# 🔍🤖 Explainable AI

To run Relevance-CAM and receive Vizual Explanations of decision making process of models, run:

```
python3 rel_cam.py
```

Full list of Vizual Explanations produced using Relevance-CAM can be found [here](https://drive.google.com/drive/folders/1E-PdSc0JoByzvo4230AOaYD6hte7SWUY?usp=share_link)


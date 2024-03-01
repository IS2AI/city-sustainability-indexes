# AI for City Districts: Measuring City Districts’ Sustainability using Satellite Data

This repository contains codes and data for performing city classification prediction tasks.

Original dataset contains raw 1585 images of 4800 x 4800 resolution (16 GB). Raw data, processed dataset, and models altogether can be downloaded by filling out [this form](https://forms.gle/vsg8SqTB1V6iqXx3A).
Alternatively, models only can be accessed at [this google drive link](https://drive.google.com/drive/folders/1-7C7YY3ejCsLZlXKM5o0E8kT5IY2ROyK?usp=sharing).

Dataset consists of 45 cities from various locations, and mostly chosen from [Arcadis Index 2022](https://www.arcadis.com/en/knowledge-hub/perspectives/global/sustainable-cities-index). Additional 8 cities (Almaty, Ankara, Ashgabat, Astana, Baku, Bishkek, Shymkent, and Tashkent) underrepresented in Arcadis Index is also part of the dataset.

Cities and their corrresponding index values from 9 different sustainability ranking systems which were used for model training:
| City IATA Code | Overall Arcadis SCI |
| -------------- | ------------------- |
| Content Cell   | Content Cell  |
| Content Cell   | Content Cell  |
|      ALA       | 
|      ESB       |
|      ASB       |
|      NQZ       |
|      GYD       |
|      BKK       |
|      PEK       | 
|      FRU       |
|      BOG       |
|      BOS       |
|      BNE       |
|      AEP       |
|      CAI       |
|      CHI       |
|      DUB       |
|      HAN       |
|      HKG       | 
|      IST       |
|      CGK       |
|      FIH       |
|      KUL       |
|      LOS       |
|      LHE       |
|      LIS       |
|      MNL       |
|      MEL       |
|      MEX       |
|      MIL       |
|      BOM       |
|      MUC       |
|      NBO       |
|      OSL       |
|      PAR       |
|      RIX       | 
|      SFO       |
|      GRU       |
|      ICN       |
|      CIT       |
|      SIN       |
|      SYD       |
|      TPE       |
|      TAS       |
|      TKY       |
|      YVR       |
|      IAD       |



# Downloading the repository

```
$ git clone https://github.com/IS2AI/city-classification-and-index-prediction
```

# Pre-processing 

Prior to training it is necessary to perform pre-processing on raw images. To generate patches out of raw images needed for training, and to perform train-val-test split launch the following script:

```
python3 preprocessing.py
```

# Training

To launch training for city index predition use:
```
python3 train_regression.py
```

# Inference

To test city index prediciton on unseen patches run:
```
python3 test_regression.py
```

# Sustainability map creation

To create sustainability color map, there is available another script:

```
python3 make_sustainability_map.py
```

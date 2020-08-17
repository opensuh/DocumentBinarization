## Two-Stage Generative Adversarial Networks for Document Image Binarization with Color Noise and Background Removal
<img width="600" alt="Figure1" src="https://user-images.githubusercontent.com/57751687/90395881-8dae9800-e095-11ea-944e-ff69e2bc1de2.png">

Abstract

Document image enhancement and binarization methods are often used to improve the accuracy and efficiency of document image analysis tasks such as text recognition. Traditional non-machine-learning methods are constructed on low-level features in an unsupervised manner but have difficulty with binarization on documents with severely degraded backgrounds. Convolutional neural network (CNN)––based methods focus only on grayscale images and on local textual features. In this paper, we propose a two-stage color document image enhancement and binarization method using generative adversarial neural networks. In the first stage, four color-independent adversarial networks are trained to extract color foreground information from an input image for document image enhancement. In the second stage, two independent adversarial networks with global and local features are trained for image binarization of documents of variable size. For the adversarial neural networks, we formulate loss functions between a discriminator and generators having an encoder--decoder structure. Experimental results show that the proposed method achieves better performance than many classical and state-of-the-art algorithms over the Document Image Binarization Contest (DIBCO) datasets, the LRDE Document Binarization Dataset (LRDE DBD), and our shipping label image dataset.

## Models

The performance of each model

<table>
  <tr align="center">
    <td colspan="2">H-DIBCO 2016</td>
    <td>FM</td>
    <td>p-FM</td>
    <td>PSNR</td>
    <td>DRD</td>
  </tr>
  <tr align="center">
    <td colspan="2">Otsu</td>
    <td>86.59</td>
    <td>89.92</td>
    <td>17.79</td>
    <td>5.58</td>
  </tr>
  <tr align="center">
    <td colspan="2">Niblack</td>
    <td>72.57</td>
    <td>73.51</td>
    <td>13.26</td>
    <td>24.65</td>
  </tr>
  <tr align="center">
    <td colspan="2">Sauvola</td>
    <td>84.27</td>
    <td>89.10</td>
    <td>17.15</td>
    <td>6.09</td>
  </tr>
  <tr align="center">
    <td colspan="2">Vo</td>
    <td>90.01</td>
    <td>93.44</td>
    <td>18.74</td>
    <td>3.91</td>
  </tr>
  <tr align="center">
    <td colspan="2">He</td>
    <td>91.19</td>
    <td>95.74</td>
    <td>19.51</td>
    <td>3.02</td>
  </tr>
  <tr align="center" style="bold">
    <td colspan="2">Zhao</td>
    <td>89.77</td>
    <td>94.85</td>
    <td>18.80</td>
    <td>3.85</td>
  </tr>
  <tr align="center" style="bold">
    <td colspan="2">Ours</td>
    <td>92.24</td>
    <td>95.95</td>
    <td>19.93</td>
    <td>2.77</td>
  </tr>
</table>

<table>
<thead>
<tr><th>Evaluation of binarization</th><th>OCR accuracy in Levenshetin distance</th></tr>
</thead>
<tr><td>

  <table>
    <tr align="center">
      <td colspan="2">Shipping Label</td>
      <td>FM</td>
      <td>p-FM</td>
      <td>PSNR</td>
      <td>DRD</td>
    </tr>
    <tr align="center">
      <td colspan="2">Otsu</td>
      <td>88.31</td>
      <td>89.42</td>
      <td>14.73</td>
      <td>6.17</td>
    </tr>
    <tr align="center">
      <td colspan="2">Niblack</td>
      <td>86.61</td>
      <td>89.46</td>
      <td>13.59</td>
      <td>6.61</td>
    </tr>
    <tr align="center">
      <td colspan="2">Sauvola</td>
      <td>87.67</td>
      <td>89.53</td>
      <td>14.18</td>
      <td>5.75</td>
    </tr>
    <tr align="center">
      <td colspan="2">Vo</td>
      <td>91.20</td>
      <td>92.92</td>
      <td>16.14</td>
      <td>2.20</td>
    </tr>
    <tr align="center">
      <td colspan="2">He</td>
      <td>91.09</td>
      <td>92.26</td>
      <td>16.03</td>
      <td>2.33</td>
    </tr>
    <tr align="center" style="bold">
      <td colspan="2">Zhao</td>
      <td>92.09</td>
      <td>93.83</td>
      <td>16.29</td>
      <td>2.37</td>
    </tr>
    <tr align="center" style="bold">
      <td colspan="2">Ours</td>
      <td>94.65</td>
      <td>95.94</td>
      <td>18.02</td>
      <td>1.57</td>
    </tr>
  </table>
  
</td><td>

  <table>
    <tr align="center">
      <td colspan="2">Shipping Label</td>
      <td>Total</td>
      <td>Korean</td>
      <td>Alphabet</td>
    </tr>
    <tr align="center">
      <td colspan="2">Input Image</td>
      <td>77.20</td>
      <td>73.86</td>
      <td>94.47</td>
    </tr>
    <tr align="center">
      <td colspan="2">Ground Truth</td>
      <td>84.62</td>
      <td>85.88</td>
      <td>96.66</td>
    </tr>
    <tr align="center">
      <td colspan="2">Otsu</td>
      <td>74.45</td>
      <td>70.72</td>
      <td>93.79</td>
    </tr>
    <tr align="center">
      <td colspan="2">Niblack</td>
      <td>69.00</td>
      <td>66.31</td>
      <td>82.94</td>
    </tr>
    <tr align="center">
      <td colspan="2">Sauvola</td>
      <td>72.84</td>
      <td>68.81</td>
      <td>93.73</td>
    </tr>
    <tr align="center">
      <td colspan="2">Vo</td>
      <td>77.14</td>
      <td>74.69</td>
      <td>89.86</td>
    </tr>
    <tr align="center">
      <td colspan="2">He</td>
      <td>75.15</td>
      <td>72.45</td>
      <td>89.13</td>
    </tr>
    <tr align="center" style="bold">
      <td colspan="2">Zhao</td>
      <td>77.33</td>
      <td>74.56</td>
      <td>91.69</td>
    </tr>
    <tr align="center" style="bold">
      <td colspan="2">Ours</td>
      <td>83.40</td>
      <td>81.15</td>
      <td>95.09</td>
    </tr>
  </table>

</td></tr>
</table>

## Prerequisites
- Linux (Ubuntu)
- Python >= 3.6
- NVIDIA GPU + CUDA CuDNN

## Installation

<!--
- Clone this repo:
```bash
git clone https://github.com/
cd dfg
```
-->

- Install [PyTorch](http://pytorch.org)
- Install [segmentation_models](https://github.com/qubvel/segmentation_models.pytorch)
- Install [pytesseract](https://github.com/madmaze/pytesseract)
- download [tesseract data](https://github.com/tesseract-ocr/tessdata_best)
  <!--
  - For pip users, please type the command `pip install -r requirements.txt`
  -->
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`

## Model train/eval
- Prepare datasets 
  - [DIBCO](https://vc.ee.duth.gr/dibco2019/)
  - [Persian](http://www.iapr-tc11.org/mediawiki/index.php/Persian_Heritage_Image_Binarization_Dataset_(PHIBD_2012))
  - [PLM](http://amadi.univ-lr.fr/ICFHR2016_Contest/index.php/download-123)
  - [S-MS](http://tc11.cvc.uab.es/datasets/SMADI_1)
  - [LRDE-DBD](https://www.lrde.epita.fr/dload/olena/datasets/dbd/1.0/)
  - [Label](https://www.kist-europe.de/portal/main/main.do)
  
 - Patch per datasets
 ```bash
 (In the case of dibco)
 python3 ./Common/make_ground_truth_dibco.py
 python3 ./Common/make_ground_truth_512_dibco.py
 ```


- Train a model per datasets
```bash
(In the case of Label)
1) sh ./Label/train_5_fold_step1.sh
2) sh ./Label/predict_for_step2_5_fold.sh
3) sh ./Label/train_5_fold_step2.sh
4) sh ./Label/train_5_fold_resize.sh
```

- Evaluate the model per datasets
<!--
(our pre-trained models are in ./pretrained_model)
- We plan to upload the pre-trained models on our Github page.
-->
```bash
(In the case of Label)
sh ./Label/predict_step2_5_fold.sh
```

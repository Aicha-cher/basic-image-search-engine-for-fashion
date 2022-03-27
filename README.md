# basic-image-search-engine-for-fashion
## Introduction

Often ecommerce clients find it difficult to  find the exact terms to describe apparel  they wish to buy

Sometime the clients don’t have a product  on mind but they are looking for a garment that complement an other when they already own for eg finding a garment that complement well the orange t-shirt and the client is looking for inspiration.

Filtering by product category and manually searching are both extremely time-consuming, and sometimes not successful specially in big webshops that often mis categorize their products

The above problems can be solved by allowing an inverse image search which will  improve customer’s shopping experience
 This have the potential to increase the conversion rates, as customers can quickly and interactively find the items they’re looking for and as well it will increase upselling and cross-selling 
 
 ## How to use this code
 
* Download the dataset from [here](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) and extract it to a folder
* Indexing the images using one or multiple descriptor, each descriptor will create a json file with image id and feature 
  * we need to provide the descriptor name : - moment, -histogram, -orb
  * as well the name of the output json file
  
```python
python index.py -d 'image_fashion'  -moment "index_mom2.json" --> for one descriptor 

python index.py -d 'image_fashion' -ci 'index2.json' -orbi "index_orb2.json" --> for multiple descriptor 

```

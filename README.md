# Basic image search engine for fashion
## Introduction

Often ecommerce clients find it difficult to  find the exact terms to describe apparel  they wish to buy

Sometime the clients don’t have a product  on mind but they are looking for a garment that complement an other when they already own for eg finding a garment that complement well the orange t-shirt and the client is looking for inspiration.

Filtering by product category and manually searching are both extremely time-consuming, and sometimes not successful specially in big webshops that often mis categorize their products

The above problems can be solved by allowing an inverse image search which will  improve customer’s shopping experience
 This have the potential to increase the conversion rate, as customers can quickly and interactively find the items they’re looking for and as well it will increase upselling and cross-selling 
 
 ## How to use this code
 
 1. Download the dataset from [here](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) and extract it to a folder
 2. Indexing the images using one or multiple descriptor, each descriptor will create a json file with image id and feature 
  * the dataset folder path before -d
  * we need to provide the descriptor name : - moment, -histogram, -orb followed by the name of the output json file
  
```python
python index.py -d 'image_fashion'  -moment "index_moment.json" --> for one descriptor 

python index.py -d 'image_fashion' -histogram 'index2.json' -orbi "index_orb.json" --> for multiple descriptor 
```
 3. Running the search : 
  * here again we need to provide the name of the descriptor or descriptors we want to use, possible options are : histogram, color moment, orb, and combined(combining color moment and orb)
  * we can use the flag -P if we want to only plot the resulted images
  * or use the flag -S if we want to save the results to disk
  * After -Q we provide the query image path, we can provide more than one query image path 
 
 ```python
 python search.py -P -Q image_fashion/1526.jpg -M combined histogram --> two descriptor histogram abd combined and we plot them only
 python search.py -P -Q image_fashion/1526.jpg  image_fashion/1527.jpg -M histogram --> one descriptor histogram abd combined and we plot them only
 ```
## Contributor
Aicha cheridi

import pandas as pd
import numpy as numpy


yelp = pd.read_csv('newreviews.csv')
print(yelp.shape)
print(yelp.head)
# Used to save the truncated data set by filtering based on dates one year ago
#yelp = pd.read_csv('yelp_review.csv')
#yelp = yelp[(yelp['date'] > '2017-05-01')]

#yelp.to_csv('newreviews.csv', encoding = 'utf-8', index=False)
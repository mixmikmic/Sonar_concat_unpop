get_ipython().magic('load_ext autotime')


import gzip
import csv


# ### Somewhat optimized version of reading and writing book-reviews file:
# 

# #### Basic info about the file:
# 1. Total #reviews in file: 12,886,488
# 2. Total #rows in file: 141,751,368
# 

col_names = ['productID', 'title', 'price', 'userID', 'profileName', 
             'helpfulness', 'score', 'time', 'summary', 'text']


def process_data(n):
    
    """
    The function reads the txt.gz file, which contains all the reviews, line-by-line, and when it 
    gathers all the data for a sinngle review, it writes (or adds) a single row for that review in a 
    csv file, and does this for all ~13M reviews.
    
    Estimated completion time for ~13M reviews: 20 min
    Estimated size of file for ~13M reviews: 10 GB 
    
    Args:
        n (int): #rows to read, provide n as a multiple of 11.
                    If n = 1100, then fetches data for 1100/11 = 100 users.
                    
    Returns:
        None
    
    """
    
    cnt = 0
    temp = []
    
    with gzip.open('E:\\MRP\\0102\\Books.txt.gz') as rf:
        with open('E:\\MRP\\0102\\Books.csv', 'w', newline = '') as cw:
            csv_writer = csv.writer(cw, delimiter = ',')
            
            while cnt < n:
                l1 = rf.readline().decode('utf-8')
                
                if len(l1) > 1:                
                    value = l1.split(':')[1].strip('\n')                  
                    temp.append(value)
                else:
                    csv_writer.writerow(temp)
                    temp = []

                cnt+=1          
            
    return None

all_records = process_data(141751368)


# ### Need to strip white-spaces from some of the columns: 1) title
# 

sample = pd.read_csv('E://MRP//0102//Books.csv', nrows = 500, names=col_names)


sample.head()


pd.pivot_table(data = sample, index = 'userID', columns = 'title', values = 'score')





# ### Collect data from goodreads.com:
# 

from bs4 import BeautifulSoup as bs
import pandas as pd
import re
import urllib
import time


p1 = 'https://www.goodreads.com/list/show/11.Best_Crime_Mystery_Books?page='
page_id = [str(n) for n in range(1,51)]


books_all = []  ## Store names of all the books from all the pages
meta_ratings = [] ## Store ratings for all the books
meta_votes = []  ## Store users' votes for all the books

for i in page_id:
    link = p1+i
    print ('page'+str(i))
    
    page = urllib.request.urlopen(url = link)
    soup = bs(page, 'lxml')
    time.sleep(2)

    ## Get book title:
    book_titles = []
    title = soup.find_all(name = 'a', class_ = 'bookTitle')
    for t in title:
        book_titles.append(t.span.text)
  
    ## Get metadata-1:
    metadata_ratings = []
    metadata = soup.find_all(name = 'span', class_ = 'minirating')
    for meta in metadata:
        metadata_ratings.append(meta.text.strip())

    ## Get metadata-2:
    metadata_score = []
    metadata_votes = []
    metadata_2 = soup.find_all(name = 'span', class_ = 'smallText uitext')
    for meta in metadata_2:
        temp = meta.text.strip('\n')
        score = temp.split('\n')[0]
        metadata_score.append(score)

        num_votes = temp.split('\n')[2]
        metadata_votes.append(num_votes)        
    
    books_all.extend(book_titles)
    meta_ratings.extend(metadata_ratings)
    meta_votes.extend(metadata_votes)


print ('Total books getched:', len(books_all))
print ('Total ratings fetched:', len(meta_ratings))
print ('Total votes fetched:', len(meta_votes))


# ### Data Post-processing:
# #### Extract and clean meta-data
# 

## For ratings:
avg_ratings = []
total_ratings = []

for rat in meta_ratings:
    avg_ratings.append(re.findall('([0-9]+[.]+[0-9]+)', rat))
    total_ratings.append(rat.split(' ')[-2])


## For votes:
total_votes = []

for vote in meta_votes:
    total_votes.append(vote.split(' ')[0])


books_df = pd.DataFrame({'Books': books_all, 'Avg_Rating': avg_ratings, 
                          'Total_Num_Ratings': total_ratings, 'Total_Num_Votes': total_votes})

books_df.head()


books_df.shape


books_df.to_csv('books_df.csv', index=False)


books_df = pd.read_csv('../data/books_df.csv', encoding='latin1')
books_df.head()





get_ipython().magic('load_ext autotime')


get_ipython().magic('unload_ext autotime')


## Import required modules:
import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


# ## Data Post-processing:
# 

# ### Import reviews file (i.e. pre-processed amazon-book-reviews created by `reviews_preprocessing.ipynb` ):
# 

col_names = ['productID', 'title', 'price', 'userID', 'profileName', 
             'helpfulness', 'score', 'time', 'summary', 'text']
books_amazon = pd.read_csv('E://MRP//0102/Books.csv', nrows = 10, encoding = 'utf-8', names=col_names)


books_amazon


## Get all the book-names from Amaozn-data (later this list will be used to find crime-mystery books):
book_names = []
i = 0
with open ('E://MRP//0102/Books.csv') as books:
    for line in books:
        book_names.append(line.split(',')[1])


## Total unique books:
len(set(book_names))


# ## Clean the book-names:
# 

unique_books = list(set(book_names))
unique_books = [book.strip().lower() for book in unique_books]
unique_books.sort()


# ### Import goodreads file (i.e. data collected from `goodreads_data.ipynb`):
# 

books_goodreads = pd.read_csv('E:/MRP/0331/Book-Recommender-System/data/books_df.csv', encoding = 'latin1')

books_goodreads['Books'] = [book.strip().lower() for book in books_goodreads['Books']]
books_goodreads['Total_Num_Ratings'] = [int(rating.replace(',','')) for rating in books_goodreads['Total_Num_Ratings']]
books_goodreads['Total_Num_Votes'] = [int(vote.replace(',','')) for vote in books_goodreads['Total_Num_Votes']]
books_goodreads['Avg_Rating'] = [float(rating.strip("['']")) for rating in books_goodreads['Avg_Rating']]

books_goodreads.head()


# ### Found matches for only 1200 books out of 5000:
# 

common_books = list(set(unique_books) & set(books_goodreads['Books']))
print (len(common_books))
common_books[0:10]


# ### Start building a recommendation system:
# 

# #### Efficient but incomplete method!
# 

temp_book = ['Dr. Seuss', 'Its Only Art If Its Well Hung!']
# temp_books = set(common_books)

list_df = []
with open ('E://MRP//0102//Books.csv') as books:
    for i in range(5):
        line = books.readline()
        book = line.split(',')[1].strip()
        
        if book in temp_book:
            list_df.append(line)


# #### Quick fix for incomplete-method:
# 

## Load only required columns to avoid memory-issues:
books_amazon_whole = pd.read_csv('E://MRP//0102/Books.csv', encoding = 'utf-8', header=None, usecols=[0,1,3,6])
new_cols = ['BookID', 'BookTitle', 'UserID', 'Score']
books_amazon_whole.columns = new_cols
books_amazon_whole['BookTitle'] = [title.strip().lower() for title in books_amazon_whole['BookTitle']]
books_amazon_whole.head()


## Subset the previous data-frame to keep records for only crime-mystery books:
books_amazon_whole = books_amazon_whole[books_amazon_whole['BookTitle'].isin(common_books)]
books_amazon_whole.head()


## Unique users
len(set(books_amazon_whole['UserID']))


amazon_rating = pd.DataFrame(books_amazon_whole.groupby(by = 'BookTitle').mean()).reset_index(drop = False)

fig, (ax1, ax2) = plt.subplots(figsize= (14,7), ncols = 2, sharey = True)
ax1.hist(amazon_rating['Score'], bins = [1,2,3,4,5], normed = True)
ax1.set_xlabel('Average Rating', fontsize = 15)
ax1.set_ylabel('Frequency (Normalized)', fontsize = 15)
ax1.set_title('Distribution of Average Rating \n (Amazon)', fontsize = 20)

ax2.hist(books_goodreads['Avg_Rating'], bins = [1,2,3,4,5], normed = True)
ax2.set_xlabel('Average Rating', fontsize = 15)
# ax2.set_ylabel('Frequency (Normalized)', fontsize = 15)
ax2.set_title('Distribution of Average Rating \n (Goodreads)', fontsize = 20)

plt.show()


## Consider users who rated >1 books. Two reasons: 1) Expect to get better results; 2) Speeds-up calculation and easy to manage.
filter_users = pd.DataFrame(books_amazon_whole.groupby(by = 'UserID').size(), columns = ['count'])
filter_users = filter_users.loc[filter_users['count'] > 1]
filter_users.reset_index(drop = False, inplace = True)

filter_users = filter_users[filter_users['UserID'] != ' unknown']  ## Remove unknown user-ID which has given 949 ratings
filter_users = books_amazon_whole.loc[books_amazon_whole['UserID'].isin(filter_users['UserID'])]


## Unique users who rated more than 1 books
len(set(filter_users['UserID']))


filter_users.head()


# books_amazon_whole.drop_duplicates(subset = ['BookTitle', 'UserID'], inplace = True)
user_item_df = pd.pivot_table(data = filter_users, index = 'UserID', columns = 'BookTitle', values = 'Score')
user_item_df.head()


## Fill NAs with 0 and get the new list of book-titles:
user_item_df.fillna(0, inplace=True)
new_book_names = user_item_df.columns
user_item_df.head()


user_item_df.shape


true_user_id = list(enumerate(user_item_df.index))
true_user_id[0:10]


## Test mapping of user-ids:
user_0 = pd.DataFrame(user_item_df.loc[' A0134066213WYQXLTVGYT'])
user_0[user_0[' A0134066213WYQXLTVGYT'] > 0]


# ### In practise, the user-item matrices are very sparse. Following two plots help in understanding the sparsity of the data in this case.
# 
# ### Ratings per user (i.e. number of books rated by a user) is very less. Most users rated only 1 book. So, user-user model may not result in a good recommendation system.
# 

user_data = [np.count_nonzero(user_item_df.iloc[i,:]) for i in range(user_item_df.shape[0])]
book_data = [np.count_nonzero(user_item_df.iloc[:,i]) for i in range(user_item_df.shape[1])]


fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14,6))
ax1.hist(user_data, bins = range(0,15,1))
ax2.hist(book_data, bins = range(0, 2000,50))

ax1.set_title('Distribution of User-Ratings', fontsize = 15)
ax2.set_title('Distribution of Book-Ratings', fontsize = 15)

ax1.set_xlabel('#Books rated by user', fontsize = 15)
ax2.set_xlabel('#Ratings each book received', fontsize = 15)

ax1.set_ylabel('#Users', fontsize = 15)
ax2.set_ylabel('#Books', fontsize = 15)

plt.show()


# ### Item-item recommendation system:
# 

user_item_mat = csr_matrix(user_item_df)
user_item_mat


# def get_dense_users(row):
#     if np.count_nonzero(user_item_mat[row,:].toarray()) > 1:
#         return row        
    
# subset_matrix_ind = [get_dense_users(row) for row in range(user_item_mat.shape[0])]


book_similarity_mat = cosine_similarity(user_item_mat.transpose(), dense_output = False)
book_similarity_mat = book_similarity_mat.toarray()
book_similarity_mat.shape


# ### Make recommendations (item-item similarity based on user-ratings):
# 

def most_similar_books(book_ind, book_names):
    
    most_sim_books = np.argsort(book_similarity_mat[book_ind,:])[::-1][0:20]
    recommended_books = [book_names[i] for i in most_sim_books]
    
    recommendation_df = pd.DataFrame({'BookIndex': most_sim_books, 'BookTitle': recommended_books})
    
    print ('Target book: ', book_names[book_ind])
    
    return recommendation_df

most_similar_books(483, new_book_names)  ## Try: 837, 483


most_similar_books(book_ind = 483, book_names = new_book_names)


# ### Making it a bit more personalized, that is, consider other books that a user likes and recommend books accordingly:
# 

# ### Matrix-Factorization Approach
# 

# ### Splitting data into train / test set and evaluating performance
# 

import random
from sklearn.decomposition import TruncatedSVD


random.seed(11)

user_ids_all = list(user_item_df.index)

train_ind = random.sample(k = 50000, population = range(58991))
train_user_ids = [user_ids_all[i] for i in train_ind]
train_data = csr_matrix(user_item_df.iloc[train_ind, :])

test_user_ids = [user_ids_all[j] for j in range(len(user_ids_all)) if j not in train_ind]
test_data = csr_matrix(user_item_df.loc[test_user_ids, :])

print ('Shape of training data: ', train_data.shape)
print ('Shape of testing data: ', test_data.shape)


# ### Sklearn implementation of SVD:
# 

def fit_svd(train, n_compo = 10, random_state = 11):
    
    tsvd = TruncatedSVD(n_components = n_compo, random_state = random_state)
    tsvd.fit(train)
    
    return tsvd

def predict_train(tsvd_obj, train):
    
    train_predictions = np.dot(tsvd_obj.transform(train), tsvd_obj.components_)
    
    return train_predictions

def predict_test(tsvd_obj, test):
    
    test_predictions = np.dot(tsvd_obj.transform(test), tsvd_obj.components_)
        
    return test_predictions


tsvd = fit_svd(train = train_data, n_compo = 25)
predict_ratings_train = predict_train(tsvd_obj = tsvd, train = train_data)
predict_ratings_train.shape


predict_ratings_test = predict_test(tsvd_obj = tsvd, test = test_data)
predict_ratings_test.shape


def plot_rmse(rmse_list, n_users = 500):
    
    fig, ax = plt.subplots(figsize = (14,8))
    ax.plot(rmse_list[0:n_users])
    ax.axhline(y = np.mean(rmse_list), label = 'Avg. RMSE: {}'.format(round(np.mean(rmse_list), 3)), 
               color = 'r', linestyle = 'dashed')
    ax.set_ylabel('RMSE', fontsize = 15)
    ax.set_xlabel('UserId', fontsize = 15)
    ax.set_title('RMSE for each user', fontsize = 20)
    ax.legend()
    plt.show()    
    
    return None


rmse_train = np.sqrt(np.mean((train_data.toarray() - predict_ratings_train)**2, axis = 1))


plot_rmse(rmse_train)


def get_recommended_books(user_id, books_list, latent_ratings, ui_mat, top_n = 15):
    
    ## Get recommendations for a given user:
    ind_top_rated_books = np.argsort(latent_ratings.iloc[user_id])[::-1][0:top_n]
    recommended_books = [books_list[ind] for ind in ind_top_rated_books]    
    recommendation_df = pd.DataFrame({'UserID': user_id, 'BookID': ind_top_rated_books, 
                                     'Recommended_Books': recommended_books})
    
    ## Get actual books that the user rated:
    user_rated_books = ui_mat[user_id,:].toarray()
    rated_books_ind = np.argwhere(user_rated_books != 0)[:,1]
    rated_books = [books_list[ind] for ind in rated_books_ind]
    user_rated_books_df = pd.DataFrame({'BookID': rated_books_ind, 'RatedBooks': rated_books, 'UserID': user_id})
    
    return user_rated_books_df, recommendation_df

## Try: 211
user_rated_books, recommended_books = get_recommended_books(user_id = 211, books_list = new_book_names, 
                                      latent_ratings = predict_ratings, ui_mat = user_item_mat)


rated_books, recommended_books = get_recommended_books(user_id = 100, books_list = new_book_names, 
                                      latent_ratings = pd.DataFrame(predict_ratings_train), ui_mat = train_data)


rated_books


recommended_books


rmse_test = np.sqrt(np.mean((test_data.toarray() - predict_ratings_test)**2, axis = 1)) 
plot_rmse(rmse_test)


# ## Notes:
# 
# 1. Increasing **#singual values** decreases RMSE. This also increases the overlap between the recommended books and the books that a user has already rated.
# 
# 2. Currently, not using actual **BookID** and **UserID**, but they can easily be traced back (example given).
# 
# 3. Sparsity of the user-item-matrix is still an issue.
# 
# 4. Only considering users who rated **at least two** books.
# 
# 5. As most of the predicted ratings are close to 0 and most of the actual ratings are also 0 (i.e. users didn't rate a lot of books), we eventually end-up with lower RMSE. In other words, sparsity (i.e. a lot of 0s)decreases RMSE. If we were to have dense matrix then, probably, RMSE will be a bit higher than current value.
# 





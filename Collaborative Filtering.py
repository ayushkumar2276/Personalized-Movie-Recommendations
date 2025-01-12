%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import imp
import numpy as np

from zipfile import ZipFile
try:
    from urllib.request import urlretrieve
except ImportError:  # Python 2 compat
    from urllib import urlretrieve

# this line need to be changed if not on colab:
data_folder = '/content/'


ML_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_1M_FILENAME = op.join(data_folder,ML_1M_URL.rsplit('/', 1)[1])
ML_1M_FOLDER = op.join(data_folder,'ml-1m')
if not op.exists(ML_1M_FILENAME):
    print('Downloading %s to %s...' % (ML_1M_URL, ML_1M_FILENAME))
    urlretrieve(ML_1M_URL, ML_1M_FILENAME)

if not op.exists(ML_1M_FOLDER):
    print('Extracting %s to %s...' % (ML_1M_FILENAME, ML_1M_FOLDER))
    ZipFile(ML_1M_FILENAME).extractall(data_folder)
import pandas as pd
all_ratings = pd.read_csv(op.join(ML_1M_FOLDER, 'ratings.dat'), sep='::',
                          names=["user_id", "item_id", "ratings", "timestamp"],engine='python')
all_ratings.head()
list_movies_names = []
list_item_ids = []
with open(op.join(ML_1M_FOLDER, 'movies.dat'), encoding = "ISO-8859-1") as fp:
    for line in fp:
        list_item_ids.append(line.split('::')[0])
        list_movies_names.append(line.split('::')[1])

movies_names = pd.DataFrame(list(zip(list_item_ids, list_movies_names)),
               columns =['item_id', 'item_name'])
movies_names.head()
movies_names['item_id']=movies_names['item_id'].astype(int)
all_ratings['item_id']=all_ratings['item_id'].astype(int)
all_ratings = all_ratings.merge(movies_names,on='item_id')

#number of entries
len(all_ratings)

# statistics of ratings
all_ratings['ratings'].describe()
# The ratings are 1, 2, 3, 4, 5
all_ratings['ratings'].unique()
all_ratings['user_id'].describe()

# number of unique users
total_user_id = len(all_ratings['user_id'].unique())
print(total_user_id)
list_user_id = list(all_ratings['user_id'].unique())
list_user_id.sort()
for i,j in enumerate(list_user_id):
    if j != i+1:
        print(i,j)
all_ratings['user_num'] = all_ratings['user_id'].apply(lambda x :x-1)
all_ratings.head()
all_ratings['item_id'].describe()

# number of unique rated items
total_item_id = len(all_ratings['item_id'].unique())
print(total_item_id)
itemnum_2_itemid = list(all_ratings['item_id'].unique())
itemnum_2_itemid.sort()
itemid_2_itemnum = {c:i for i,c in enumerate(itemnum_2_itemid)}
all_ratings['item_num'] = all_ratings['item_id'].apply(lambda x: itemid_2_itemnum[x])
def check_ratings_num(df):
    item_num = set(df['item_num'])
    if item_num == set(range(len(item_num))):
        return True
    else:
        return False
check_ratings_num(all_ratings)
all_ratings.head()
counts = all_ratings.groupby("item_name").size()
top10 = counts.sort_values(ascending=False).head(10)

top10
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.hist(counts, bins=30, edgecolor='black', alpha=0.9)
plt.title("Movie Popularity")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Movies")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()
users = all_ratings.groupby("user_id").size()

plt.figure(figsize=(12, 6))
plt.hist(users, bins=30, edgecolor='black', alpha=0.7)
plt.title("User Activity")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Users")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()


plt.show()
average_ratings = all_ratings.groupby("item_name")["ratings"].mean()
top_20rate = average_ratings.sort_values(ascending=False).head(20)

top_20rate
#Split the data into train, validation and test
from sklearn.model_selection import train_test_split

ratings_trainval, ratings_test = train_test_split(all_ratings, test_size=0.1, random_state=42)

ratings_train, ratings_val = train_test_split(ratings_trainval, test_size=0.1, random_state=42)
user_id_train = ratings_train['user_id']
item_id_train = ratings_train['item_id']
rating_train = ratings_train['ratings']

user_id_test = ratings_test['user_id']
item_id_test = ratings_test['item_id']
rating_test = ratings_test['ratings']
movies_not_train = list(set(all_ratings['item_id']) -set(item_id_train))
movies_not_train_name=set(all_ratings.loc[movies_not_train]['item_name'])
print(movies_not_train_name)
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def df_2_tensor(df, device):
    # return a triplet user_num, item_num, rating from the dataframe
    user_num = np.asarray(df['user_num'])
    item_num = np.asarray(df['item_num'])
    rating = np.asarray(df['ratings'])
    return torch.from_numpy(user_num).to(device), torch.from_numpy(item_num).to(device), torch.from_numpy(rating).to(device)
train_user_num, train_item_num, train_rating = df_2_tensor(ratings_train,device)
val_user_num, val_item_num, val_rating = df_2_tensor(ratings_val,device)
test_user_num, test_item_num, test_rating = df_2_tensor(ratings_test,device)
from torch.utils.data import DataLoader, Dataset

def tensor_2_dataset(user,item,rating):
    dataset = list(zip(user, item, rating))
    return dataset

def make_dataloader(dataset,bs,shuffle):
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=shuffle)
    return dataloader
train_dataset = tensor_2_dataset(train_user_num,train_item_num, train_rating)
val_dataset = tensor_2_dataset(val_user_num,val_item_num,val_rating)
test_dataset = tensor_2_dataset(test_user_num, test_item_num, test_rating)
train_dataloader = make_dataloader(train_dataset,1024,True)
val_dataloader = make_dataloader(val_dataset,1024, False)
test_dataloader = make_dataloader(test_dataset,1024,False)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """
    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0.0)


class ZeroEmbedding(nn.Embedding):
    """
    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0.0)
class DotModel(nn.Module):

    def __init__(self,
                 num_users,
                 num_items,
                 embedding_dim=32):

        super(DotModel, self).__init__()

        self.embedding_dim = embedding_dim

        # TODO: generate user and item embeddigns using ScaledEmbedding
        # your code
        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)

        # TODO: generate bias embeddigns using ZeroEmbedding
        # your code
        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)


    def forward(self, user_ids, item_ids):

        # TODO: compute and return the predicted rating based on the embedding vectors and biases.
        # your code
        user_vector = self.user_embeddings(user_ids)
        item_vector = self.item_embeddings(item_ids)

        product = (user_vector * item_vector).sum(dim=-1)

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        predicted_rating = product + user_bias + item_bias

        return predicted_rating
net = DotModel(total_user_id,total_item_id).to(device)
net
batch_user, batch_item, batch_rating = next(iter(train_dataloader))
batch_user, batch_item, batch_rating = batch_user.to(device), batch_item.to(device), batch_rating.to(device)
predictions = net(batch_user, batch_item)
predictions.shape
def regression_loss(predicted_ratings, observed_ratings):
    return ((observed_ratings - predicted_ratings) ** 2).mean()
loss=regression_loss(predictions,batch_rating)
class FactorizationModel(object):

    def __init__(self, embedding_dim=32, n_iter=10, l2=0.0,
                 learning_rate=1e-2, device=device, net=None, num_users=None,
                 num_items=None,random_state=None):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._device = device
        self._num_users = num_users
        self._num_items = num_items
        self._net = net
        self._optimizer = None
        self._loss_func = None
        self._random_state = random_state or np.random.RandomState()


    def _initialize(self):
        if self._net is None:
            self._net = DotModel(self._num_users, self._num_items, self._embedding_dim).to(self._device)

        self._optimizer = optim.Adam(
                self._net.parameters(),
                lr=self._learning_rate,
                weight_decay=self._l2
            )

        self._loss_func = regression_loss

    @property
    def _initialized(self):
        return self._optimizer is not None


    def fit(self, dataloader, val_dataloader, verbose=True):
        if not self._initialized:
            self._initialize()

        valid_loss_min = np.Inf # track change in validation loss
        train_losses, valid_losses, valid_maes =[], [], [] # track train losses, valid loss, and valid maes over epoches

        for epoch_num in range(self._n_iter):
            tot_train_loss = 0.0
            ###################
            # train the model #
            ###################
            #Trainining loop:
            self._net.train()
            for batch_user, batch_item, batch_rating in dataloader:
                batch_user, batch_item, batch_rating = (
                    batch_user.to(self._device),
                    batch_item.to(self._device),
                    batch_rating.to(self._device))

                self._optimizer.zero_grad()
                predictions = self._net(batch_user, batch_item)
                loss = self._loss_func(predictions, batch_rating)
                loss.backward()
                self._optimizer.step()

                tot_train_loss += loss.item()

            train_loss = tot_train_loss /len(dataloader)

            # Go to the validation loop
            valid_loss, valid_mae = self.test(val_dataloader)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_maes.append(valid_mae)

            if verbose:
                print('Epoch {}: loss_train {}, loss_val {}'.format(epoch_num, train_loss,valid_loss))

            if np.isnan(train_loss) or train_loss == 0.0:
                raise ValueError('Degenerate train loss: {}'.format(train_loss))

            #TODO: Saving model if validation loss has decreased
            # your code
            if valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
                torch.save(self._net.state_dict(), 'case2best_model.pt')


        return train_losses, valid_losses, valid_maes


    ######################
    # validate/Test the model #
    ######################
    def test(self,dataloader, verbose = False):
        self._net.eval()
        L1loss = torch.nn.L1Loss()
        tot_test_loss = 0.0
        tot_test_mae = 0.0

        # Validation/testing loop
        # (mae = mean absolute error)
        with torch.no_grad():
            for batch_user, batch_item, batch_rating in dataloader:
                batch_user, batch_item, batch_rating = (
                    batch_user.to(self._device),
                    batch_item.to(self._device),
                    batch_rating.to(self._device))

                predictions = self._net(batch_user, batch_item)
                loss = self._loss_func(predictions, batch_rating)
                mae = L1loss(predictions, batch_rating)

                tot_test_loss += loss.item()
                tot_test_mae += mae.item()

        test_loss = tot_test_loss / len(dataloader)
        test_mae = tot_test_mae / len(dataloader)
        if verbose:
            print(f"RMSE: {np.sqrt(test_loss)}, MAE: {test_mae}")
        return test_loss, test_mae
#Constructing model using FactorizationModel
model = FactorizationModel(
    embedding_dim=32,
    n_iter=20,
    l2=1e-5,
    learning_rate=1e-3,
    device=device,
    num_users=total_user_id,
    num_items=total_item_id)
train_losses, valid_losses, valid_maes = model.fit(train_dataloader, val_dataloader, verbose=True)

plt.figure(figsize=(12, 6))
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
plt.plot(range(len(valid_losses)), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
best_epoch = valid_losses.index(min(valid_losses))
print(f"Stop training at epoch {best_epoch + 1} with validation loss {min(valid_losses)}")
test_loss, test_mae = model.test(test_dataloader, verbose=True)
test_rmse = np.sqrt(test_loss)
hyperparams = {
    "embedding_dim": 64,
    "n_iter": 30,
    "learning_rate": 5e-4,
    "l2": 1e-6
}

model = FactorizationModel(
    embedding_dim=hyperparams["embedding_dim"],
    n_iter=hyperparams["n_iter"],
    l2=hyperparams["l2"],
    learning_rate=hyperparams["learning_rate"],
    device=device,
    num_users=total_user_id,
    num_items=total_item_id
)

train_losses, valid_losses, valid_maes = model.fit(train_dataloader, val_dataloader, verbose=True)

test_loss, test_mae = model.test(test_dataloader, verbose=True)
test_rmse = np.sqrt(test_loss)

torch.save(model._net.state_dict(), "model_cf.pt")
# Retreiving the bias of the movies from your optimized model and store it in the numpy array item_bais_np
item_bias_np = model._net.item_biases.weight.data.cpu().numpy()
item_bias_np = item_bias_np.squeeze()
# Constructing a dictionary that maps item_num to item_name, and vice versa
numitem_2_name = {i:name for name,i in np.asarray(all_ratings[['item_name', 'item_num']])}
name_2_numitem = {name:i for name,i in np.asarray(all_ratings[['item_name', 'item_num']])}
# Constructing a list of movie names and the corresponding bias.
list_name_bias = [[name, item_bias_np[name_2_numitem[name]]] for name in list(ratings_train['item_name'].unique())]
# Sorting the movie names by biases and output the top 10 movie names with the largest biases.
list_name_bias.sort(key=lambda x: x[1], reverse=True)
top_10_movies_bias = list_name_bias[:10]

top_10_movies_bias
from sklearn.decomposition import PCA
from operator import itemgetter
# Retriving your movie embedding vectors and store them as a numpy matrix
item_emb_np = model._net.item_embeddings.weight.data.cpu().numpy()
#Here we perform PCA to extract the 4 principal components
pca = PCA(n_components=4)
latent_fac = pca.fit_transform(item_emb_np)
#Here we get the top 1000 mostly rated movies
g = all_ratings.groupby('item_name')['ratings'].count()
most_rated_movies = g.sort_values(ascending=False).index.values[:1000]
# we get the corresponding movie numbers
most_rated_movies_num = [name_2_numitem[n] for n in most_rated_movies]
nums = most_rated_movies_num[:80]
txt_movies_names = [numitem_2_name[i] for i in nums]
X = latent_fac[nums,1]
Y = latent_fac[nums,2]
plt.figure(figsize=(15,15))
plt.scatter(X, Y)
for i, x, y in zip(txt_movies_names, X, Y):
    plt.text(x+0.01,y-0.01,i, fontsize=11)
plt.show()
my_ratings = {
    "One Flew Over the Cuckoo's Nest (1975)": 5.0,
    'James and the Giant Peach (1996)': 4.0,
    'My Fair Lady (1964)': 4.5,
    'Erin Brockovich (2000)': 3.0,
    "Bug's Life, A (1998)": 4.0,
    'Princess Bride, The (1987)': 5.0,
    'Ben-Hur (1959)': 4.5,
    'Christmas Story, A (1983)': 3.5,
    'Snow White and the Seven Dwarfs (1937)': 4.0,
    'Wizard of Oz, The (1939)': 5.0
}
Computing embedding vector and bias
from sklearn.linear_model import Ridge
# Map team ratings to movie numbers
rated_movie_nums = [name_2_numitem[name] for name in my_ratings.keys()]

#Extracting the corresponding movie embedding vector and bias
rated_movie_embeddings = item_emb_np[rated_movie_nums]
rated_movie_biases = item_bias_np[rated_movie_nums]

X = np.hstack([rated_movie_embeddings, np.ones((len(rated_movie_embeddings), 1))])  # 加一列偏置
y = np.array(list(my_ratings.values())) - rated_movie_biases  # 减去电影偏置


#Fitting user embeddings using Ridge Regression
ridge = Ridge(alpha=1e-2)  # L2
ridge.fit(X, y)

my_emb_np = ridge.coef_[:-1]
my_emb_bias = ridge.intercept_
# compute pred_ratings
pred_ratings = np.dot(item_emb_np, my_emb_np) + item_bias_np + my_emb_bias
# output the top 10 movies with the highest predicted ratings.
unrated_movie_nums = list(set(range(len(item_emb_np))) - set(rated_movie_nums))
unrated_movie_ratings = {numitem_2_name[num]: pred_ratings[num] for num in unrated_movie_nums}
top_10_movies = sorted(unrated_movie_ratings.items(), key=lambda x: x[1], reverse=True)[:10]

#top10
print("Top 10 recommended movies:")
for movie, rating in top_10_movies:
    print(f"Movie: {movie}, Predicted Rating: {rating:.2f}")
# Get indices for top 1000 active users and top 1000 mostly rated movies
top1000_user_num = pd.Series(all_ratings["user_num"].value_counts()[:1000].index, name="user_num")
top1000_item_num = pd.Series(all_ratings["item_num"].value_counts()[:1000].index, name="item_num")
# Construct dataframes for storing predicted ratings
pred_ratings = pd.merge(top1000_user_num, top1000_item_num, how="cross")
cartesian_user_num = torch.from_numpy(np.asarray(pred_ratings["user_num"])).to(device)
cartesian_item_num = torch.from_numpy(np.asarray(pred_ratings["item_num"])).to(device)
# sanity check: # of rows in pred_ratings == # of active users (1000) x # of popular items (1000)
pred_ratings.shape[0] == 1000 * 1000
#Optimized model to compute the predicted ratings among the top 1000 active users and top 1000 popular movies
pred_ratings["pred_ratings"] = model._net(
    torch.from_numpy(np.asarray(pred_ratings["user_num"])).to(device),
    torch.from_numpy(np.asarray(pred_ratings["item_num"])).to(device)
).cpu().detach().numpy()
# Each user's total viewtime is stored in Series user_total_viewtime
user_total_viewtime = all_ratings["user_num"].value_counts()[:1000] / all_ratings["user_num"].value_counts()[:1000].sum()
user_total_viewtime.name = "user_total_viewtime"
user_total_viewtime.index.name = "user_num"
pred_ratings = pd.merge(pred_ratings, user_total_viewtime, left_on="user_num", right_on="user_num", how="left")
# Each user per each movie viewtime: pred_all_ratings["user_item_viewtime"] = ["1/user_rank"] / ["sum(1/user_rank)"] * ["user_total_viewtime"]
# these wordy codes are to reduce RAM consumption otherwise Colab may crush
pred_ratings["1/user_rank"] = 1/pred_ratings.groupby("user_num")["pred_ratings"].rank(method = "min", ascending = False)
sum_inverse_user_rank = pred_ratings.groupby("user_num")["1/user_rank"].sum()
sum_inverse_user_rank.name = "sum(1/user_rank)"
pred_ratings = pd.merge(pred_ratings, sum_inverse_user_rank, left_on="user_num", right_on="user_num", how="left")
pred_ratings["user_item_viewtime"] = pred_ratings["1/user_rank"] / pred_ratings["sum(1/user_rank)"] * pred_ratings["user_total_viewtime"]
# sanity check: sum of user_item_viewtime == 1
pred_ratings["user_item_viewtime"].sum()
# compute each movie's value
budget =  1000000000
#TODO: compute and store the values of movies based on pred_ratings
movie_values = pred_ratings.groupby("item_num")["user_item_viewtime"].sum() * budget
movie_values.name = "item_value"
# replace item_num by item_name
movie_values = movie_values.to_frame()
movie_values["item_name"] = [numitem_2_name[item_num] for item_num in movie_values.index]
movie_values = movie_values.set_index("item_name").squeeze(axis=1)
# sanity check: sum of movie values == budget
movie_values.sum() == budget
# your code
# estimated value of Toy Story (1995
toy_story_value = movie_values.get("Toy Story (1995)", "Not Found")
print(f"Estimated Value of Toy Story (1995): ${toy_story_value:,.2f}" if toy_story_value != "Not Found" else "Toy Story (1995) not found")

# top 10 mostly valued movies
top_10_valued_movies = movie_values.sort_values(ascending=False).head(10)
print("Top 10 mostly valued movies:")
for movie, value in top_10_valued_movies.items():
    print(f"Movie: {movie}, Estimated Value: ${value:,.2f}")
# your code
#top 30 rated
top_30_rated_movies = all_ratings["item_name"].value_counts().head(30)
print("Top 30 mostly rated movies:")
print(top_30_rated_movies)
#top 30 valued
top_30_valued_movies = movie_values.sort_values(ascending=False).head(30)
print("\nTop 30 mostly valued movies:")
print(top_30_valued_movies)
#top 30 rated but not in top 30 valued
rated_not_valued = set(top_30_rated_movies.index) - set(top_30_valued_movies.index)
print("\nMovies in Top 30 Rated but NOT in Top 30 Valued:")
print(rated_not_valued)

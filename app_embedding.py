import pandas as pd
import numpy as np
from model_wgan_gp_embedding_private import ModelWGANGP_PRIVATE
from model_wgan_gp_embedding_noisy import ModelWGANGPNoisy
from model_wgan_gp_embedding import ModelWGANGP, EmbeddingMapping, EmbeddingNetwork
from model_wgan_noisy import ModelWGANNoisy
from model_acgan_noisy import ModelACGAN
from processing import Processing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import os
import time
import geopy.distance


# ========================
# load data sources
# ========================
train_data = pd.read_csv('gan/data/sample_geo_270719.csv')
label_data = pd.read_csv('gan/data/label_data.csv')

# ========================
# Initialize variables
# ========================
pr = Processing()
le = preprocessing.LabelEncoder()
scaler_data = StandardScaler()
scaler_label = StandardScaler()

# class level variables
scaled_data = scale_label = None
encoding = None


def train_gan(data):

    gan = ModelWGANGP()
    # gan = ModelWGANGP_PRIVATE()
    
    # gan.epochs = 10001
    # gan.train(data)
    gan.train(data, epochs=10001)

def generate_data():
    gan_mix = ModelWGANGP()
    # gan_mix = ModelWGANGP_PRIVATE()
    # gan_mix = ModelWGANNoisy()
    gan_mix.load_model('10000')

    noise = np.random.normal(0, 1, (40000, 100))

    generated_data = gan_mix.generator.predict(noise)
    
    pr.save_data(generated_data, file_name='wgan-data-200')


def preprocess(data, labels):

    process_data = data

    # normalize numerical variables
    numcols = ['P_GRAGE', 'P_AGE']

    # loop through numeric columns and normalize
    num_scale_params = {}
    numeric_scale = MinMaxScaler()
    
    # fit the scaler
    numeric_scale.fit(process_data[numcols])
    
    num_scale_params["min"] = numeric_scale.min_
    num_scale_params["scale"] = numeric_scale.scale_
    num_scale_params["data_min"] = numeric_scale.data_min_
    num_scale_params["data_max"] = numeric_scale.data_max_
    num_scale_params["data_range"] = numeric_scale.data_range_

    # save scalar configs to file
    pr.save_data(num_scale_params, "numeric_scales")

    # apply scaler transform to data
    normalized = numeric_scale.transform( process_data[numcols] )

    # drop the existing numeric columns
    process_data.drop(numcols, axis=1, inplace=True)

    # append normalized columns
    df = pd.DataFrame( normalized, columns = [col for col in numcols] )
    process_data = pd.concat( [process_data, df], axis=1 )

     # encode categorical variables
    category_cols = ['P_SEXE', 'P_STATUT', 'P_MOBIL']
    process_data = pr.process_categories(process_data, category_cols)

    return process_data

def post_process():
    
    trip_data = pd.read_csv('gan/gen-data/wgan-data-200.csv')

    orig_lat = orig_lng = dest_lat = dest_lng = []

    A = pd.DataFrame(columns=['ORIG_LAT', 'ORIG_LON', 'DEST_LAT', 'DEST_LON', 'DIST'])
    for index, row in trip_data.iterrows():
       
        olat, olng = pr.get_LatLng(row.ORIG)
        dlat, dlng = pr.get_LatLng(row.DEST)

        dist = pr.get_distance (olat, olng, dlat, dlng)

        newrow = [olat, olng, dlat, dlng, dist]
        A.loc[index] = newrow
        # print ('{} of {} completed.'.format(index, len(trip_data.iterrows())))
    
    trip_data = pd.concat( [trip_data, A], axis=1 )
    trip_data.to_csv('gan/gen-data/wgan-out-200.csv', index = None, header=True)

    print ("Data successfully saved!")

def normalize_data():
    file_name = 'gan/gen-data/scaler' + '.npy'

    num_file = np.load('gan/gen-data/numeric_scales.npy', allow_pickle=True)
    cat_file = np.load('gan/gen-data/category_encodings.npy', allow_pickle=True)
    emb_file = np.load('gan/gen-data/embedding_weights.npy', allow_pickle=True)
    map_file = np.load('gan/gen-data/embedding_mappings.npy', allow_pickle=True)

    num_file = num_file.item()
    cat_file = cat_file.item()
    emb_file = emb_file.item()
    map_file = map_file.item()

    # retrieve numeric scaling setting
    num_scaler = MinMaxScaler()
    num_scaler.min_ = num_file.get('min')
    num_scaler.scale_ = num_file.get('scale')
    num_scaler.data_min_ = num_file.get('data_min')
    num_scaler.data_max_ = num_file.get('data_max')
    num_scaler.data_range_ = num_file.get('data_range')

    # retrieve generated data
    generated_data = pr.load_data('wgan-data-200')

    # get embedding weights for dictionary
    orig_embeddings = {}
    orig_weights = emb_file.get('ORIG')
    for idx, emb in enumerate(orig_weights):
        emb = str(emb)
        orig_embeddings[emb] = idx
    
    dest_embeddings = {}
    dest_weights = emb_file.get('DEST')
    for idx, emb in enumerate(dest_weights):
        emb = str(emb)
        dest_embeddings[emb] = idx

    
    # # split numeric and category columns
    num_cols, cat_sex, cat_statut, cat_mobil, col_orig, col_dest = np.split(generated_data, [2,4,12,15,20], axis=1)
    reversed_num_cols = num_scaler.inverse_transform( num_cols )

    # # split each feature
    col_grage, col_age = np.split(reversed_num_cols, [ 1], axis=1)

    # # convert to expected data types
    col_age = np.absolute(col_age.astype(int))
    col_grage = np.absolute(col_grage.astype(int))

    # decode categories to default
    in_sex = np.round(cat_sex), cat_file.get('P_SEXE')[0]
    cat_sex = pr.decode_categories( in_sex ).reshape(-1, 1)
    in_mobil = np.round(cat_mobil), cat_file.get('P_MOBIL')[0]
    cat_mobil = pr.decode_categories( in_mobil ).reshape(-1, 1)
    in_statut = np.round(cat_statut), cat_file.get('P_STATUT')[0]
    cat_statut = pr.decode_categories( in_statut ).reshape(-1, 1)

    # retrieve encoding mappings
    orig_mapping = map_file.get('ORIG')
    dest_mapping = map_file.get('DEST')

    col_orig = pr.get_embedding_idx(orig_weights, col_orig, orig_embeddings, orig_mapping).reshape((-1,1))
    col_dest = pr.get_embedding_idx(dest_weights, col_dest, dest_embeddings, dest_mapping).reshape((-1,1))

    normalized_data = np.concatenate((col_age, col_grage, cat_sex, cat_mobil, cat_statut, col_orig, col_dest ), axis=1)
    
    columns = ['P_AGE', 'P_GRAGE',  'P_SEXE', 'P_MOBIL', 'P_STATUT', 'ORIG', 'DEST']
    normalized_df = pd.DataFrame( normalized_data, columns=columns )

    normalized_df.to_csv ('gan/gen-data/wgan-data-200.csv', index = None, header=True)


# **********************************
#  Select data sample
# **********************************

# sample_geo = train_data.sample(n=20000, random_state=1)
# sample_geo.to_csv('gan/data/sample_geo.csv')


# *********************************
#  Load and process data
# *********************************

# # create and retrieve embeddings
# embedding_cols = ['ORIG', 'DEST']
# cols = ['P_GRAGE', 'P_AGE', 'P_SEXE', 'P_STATUT', 'P_MOBIL', 'ORIG', 'DEST' ]
# train_data = train_data [cols]


# # set embedding mapping
# print (' ***** Creating mappings *********')

# orig_mapping = EmbeddingMapping(train_data['ORIG'])
# mp_orig = train_data.assign(orig_mapping=train_data['ORIG'].apply(orig_mapping.get_mapping))
# dest_mapping = EmbeddingMapping(train_data['DEST'])
# mp_dest = train_data.assign(dest_mapping=train_data['DEST'].apply(dest_mapping.get_mapping))

# encoded_objects = {}
# encoded_objects['ORIG'] = orig_mapping
# encoded_objects['DEST'] = orig_mapping

# pr.save_data(encoded_objects, "embedding_mappings")

# print (' ***** Mappings successfully saved! *********')


# # create embeddings
# print (' ***** Creating embeddings *********')

# emb = EmbeddingNetwork()
# emb.orig_size = orig_mapping.num_values
# emb.dest_size = dest_mapping.num_values
# model = emb.build_network();

# # fit data to model embeddings
# embedding_objects = {}

# cols = [mp_orig.orig_mapping, mp_dest.dest_mapping]
# embeddings = model.predict(cols)

# embedding_objects['ORIG'] = model.get_weights()[0]
# embedding_objects['DEST'] = model.get_weights()[1]

# pr.save_data(embedding_objects, "embedding_weights")

# print (' ***** Embeddings successfully saved! *********')

# # split embeddings
# col_orig, col_dest = np.split(embeddings, [5], axis=1)
# emb_cols = ['ORIG', 'DEST']

# orig_df = pd.DataFrame( col_orig, columns = ["ORIG_" + str(int(i)) for i in range(col_orig.shape[1])] )
# dest_df = pd.DataFrame( col_dest, columns = ["DEST_" + str(int(i)) for i in range(col_dest.shape[1])] )


# # process non-embedding fields
# cols = ['P_GRAGE', 'P_AGE', 'P_SEXE', 'P_STATUT', 'P_MOBIL' ]
# un_cols = train_data[cols]
# un_data = preprocess(un_cols, label_data)

# process_data = pd.concat( [un_data, orig_df, dest_df], axis=1 )

start_time = time.time()

# # training the gan 
# train_gan ( process_data.values )

# generate_data()
# normalize_data()
post_process()

print("--- %s seconds ---" % (time.time() - start_time))

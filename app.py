import pandas as pd
import numpy as np
# from model_gan import ModelGAN
# from model_wgan import ModelWGAN
from model_wgan_gp_1 import ModelWGANGP
from processing import Processing, EmbeddingMapping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
# import category_encoders as ce


# ========================
# load data sources
# ========================
# train_data = pd.read_csv('gan/data/train_data.csv')
label_data = pd.read_csv('gan/data/label_data.csv')
# geo_data = pd.read_csv('gan/data/geo2.csv')
train_data = pd.read_csv('testing.csv')
# train_data = geo_data[['P_AGE', 'P_GRAGE', 'P_SEXE', 'P_STATUT', 'P_MOBIL', 'ORIG', 'DEST']]
# train_data = train_data.head(1000)

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
    gan.train(data, epochs=50001)

def generate_data():
    gan_mix = ModelWGANGP()
    gan_mix.load_model('20000')

    noise = np.random.normal(0, 1, (10000, 200))

    generated_data = gan_mix.generator.predict(noise)
    
    pr.save_data(generated_data, file_name='wgan-data-11')


def preprocess(data, labels):

    process_data = data

    # normalize numerical variables
    numcols = ['P_AGE', 'P_GRAGE', 'ORIG_LON', 'ORIG_LAT', 'DEST_LON', 'DEST_LAT']

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
    # process_data = encoding.get("data")

    return process_data

def postprocess(inTrain, inLabel):
    # decode categorical variables
    # train.Sex = le.inverse_transform(train.Sex)

    # reverse numeric variables
    inv_train = scale_train.inverse_transform( inTrain )
    inv_label = scale_label.inverse_transform( inLabel )

def normalize_data():
    file_name = 'gan/gen-data/scaler' + '.npy'

    num_file = np.load('gan/gen-data/numeric_scales.npy', allow_pickle=True)
    cat_file = np.load('gan/gen-data/category_encodings.npy', allow_pickle=True)

    num_file = num_file.item()
    cat_file = cat_file.item()

    # retrieve numeric scaling setting
    num_scaler = MinMaxScaler()
    num_scaler.min_ = num_file.get('min')
    num_scaler.scale_ = num_file.get('scale')
    num_scaler.data_min_ = num_file.get('data_min')
    num_scaler.data_max_ = num_file.get('data_max')
    num_scaler.data_range_ = num_file.get('data_range')

    # retrieve generated data
    generated_data = pr.load_data('wgan-data-11')

    # split numeric and category columns
    num_cols, cat_sex, cat_statut, cat_mobil = np.split(generated_data, [6,8,16], axis=1)
    reversed_num_cols = num_scaler.inverse_transform( num_cols )

    # # split each feature
    col_grage, col_age, orig_lon, orig_lat, dest_lon, dest_lat = np.split(reversed_num_cols, [1,1,1,1,1], axis=1)

    # # convert to expected data types
    col_age = col_age.astype(int)
    col_grage = col_grage.astype(int)

    # decode categories to default
    in_sex = np.round(cat_sex), cat_file.get('P_SEXE')[0]
    cat_sex = pr.decode_categories( in_sex ).reshape(-1, 1)
    in_mobil = np.round(cat_mobil), cat_file.get('P_MOBIL')[0]
    cat_mobil = pr.decode_categories( in_mobil ).reshape(-1, 1)
    in_statut = np.round(cat_statut), cat_file.get('P_STATUT')[0]
    cat_statut = pr.decode_categories( in_statut ).reshape(-1, 1)
    
    normalized_data = np.concatenate((col_grage, col_age, cat_sex, cat_mobil, cat_statut, orig_lon, orig_lat, dest_lon, dest_lat), axis=1)
    
    columns = ['P_GRAGE', 'P_AGE',  'P_SEXE', 'P_MOBIL', 'P_STATUT', 'ORIG_LON', 'ORIG_LAT', 'DEST_LON', 'DEST_LAT']
    normalized_df = pd.DataFrame( normalized_data, columns=columns )

    normalized_df.to_csv ('gan/gen-data/wgan-data-11.csv', index = None, header=True)



cols = ['P_GRAGE', 'P_AGE', 'ORIG_LON', 'ORIG_LAT', 'DEST_LON', 'DEST_LAT', 'P_SEXE', 'P_STATUT', 'P_MOBIL' ]
train_data = train_data [cols]

data = preprocess(train_data, label_data)

train_gan ( data.values )

# generate_data()
# normalize_data()


from load_data import train_data, test_data, train_label, test_label, positive_train, negative_train
from svm_original import train_svm
from genetic import genetic_algorithm
import random
import numpy as np
import matplotlib.pyplot as plt
from skbio.stats.composition import ilr, ilr_inv
from skbio.stats.composition import clr, clr_inv
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA, SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Params ::
# iterations : Number of times to run the algorithm to take average
# sample size : sample size of training data
# reduce : The target dimension to reduce to
# train_data_svm : training data
# train_label_svm : training label

def split_train_test( positive_train, negative_train, sample_size ):

    k = random.randrange( 0 , len(positive_train) -  ( sample_size // 2 ) )
    train_positive = positive_train[k: (k + (sample_size // 2) ) , :]
    train_postive_label = np.ones( ( sample_size // 2 )  )

    p = random.randrange( 0 , len(negative_train) -  ( sample_size // 2 ) )
    train_negative = negative_train[k: (k + (sample_size // 2) ) , :]
    train_negative_label = np.zeros( ( sample_size // 2 )  )

    train_sample_data = np.append( train_positive, train_negative, axis = 0)
    train_sample_label = np.append( train_postive_label, train_negative_label, axis = 0)

    data_label = np.append( train_sample_data, train_sample_label.reshape((( len(train_sample_label) ) , 1 )), axis = 1)
    np.random.shuffle( data_label )

    features = len(data_label[0])
    train_sample_data = data_label[:, 0:features - 1]
    train_sample_label = data_label[:,features - 1]

    return train_sample_data, train_sample_label

def train( iterations, sample_size, reduce, positive_train, negative_train, test_data, test_label):
    f1_original_clr = []
    f1_original = []
    f1_dca = []
    f1_clr = []
    f1_ilr = []

    roc_original_clr = []
    roc_original = []
    roc_dca = []
    roc_clr = []
    roc_ilr = []
    for _ in range( iterations ):
    # Select a smaller size
        #Select a random set from the train data
        train_sample_data, train_sample_label = split_train_test( positive_train, negative_train, sample_size )

        f1_original_data, roc_original_data = train_svm(train_sample_data, train_sample_label, test_data,
                                                            test_label)
        f1_original.append( f1_original_data )
        roc_original.append( roc_original_data )

        train_sample_data[train_sample_data == 0] = 0.1e-32
        test_data[test_data == 0] = 0.1e-32

        clr_original_train = clr(train_sample_data)
        clr_original_test = clr(test_data)

        scaler = StandardScaler()
        clr_original_train = np.nan_to_num(scaler.fit_transform(clr_original_train))
        clr_original_test = np.nan_to_num(scaler.fit_transform(clr_original_test))

        f1_original_data_clr, roc_original_data_clr = train_svm( clr_original_train, train_sample_label, clr_original_test, test_label )
        f1_original_clr.append ( f1_original_data_clr )
        roc_original_clr.append( roc_original_data_clr )

        matrices = genetic_algorithm( train_sample_data, reduce )
        roc_dca_iterations = []
        for br_matrix in matrices:
            #br_matrix = matrices[0]
            reduced_data = np.matmul(br_matrix, train_sample_data.transpose()).transpose()
            reduced_test = np.matmul(br_matrix, test_data.transpose()).transpose()

            f1_dca_data, roc_dca_data = train_svm( reduced_data, train_sample_label, reduced_test, test_label )
            #f1_dca.append( f1_dca_data )
            roc_dca_iterations.append( roc_dca_data )
        #print ("DCA max", max(roc_dca_iterations) )
        roc_dca.append( max(roc_dca_iterations) )
        #print ( " PCA CLR train shape ", train_sample_data.shape )
        # Do ILR and CLR transformation
        # Set zeros to small values
        train_sample_data[train_sample_data == 0] = 0.1e-32
        test_data[test_data == 0] = 0.1e-32

        clr_data_train = clr(train_sample_data)
        clr_test = clr(test_data)

        ilr_data_train = ilr( train_sample_data )
        ilr_test = ilr( test_data )
        np.savetxt("ilr_data.csv", ilr_data_train, delimiter=",")

        # Do PCA to reduce dimensions
        pca_clr = PCA(n_components = reduce)
        pca_ilr = PCA(n_components = reduce)
        #print ( "reduce ", reduce )

        fit_train_clr = np.ascontiguousarray( pca_clr.fit_transform(clr_data_train) )
        fit_test_clr = np.ascontiguousarray( pca_clr.transform(clr_test) )

        fit_train_ilr = np.ascontiguousarray( pca_ilr.fit_transform(ilr_data_train) )
        fit_test_ilr = np.ascontiguousarray( pca_ilr.transform(ilr_test) )
        np.savetxt("ilr_data_pca.csv", fit_train_ilr, delimiter=",")

        pca_clr_reduced_train = np.nan_to_num( fit_train_clr )
        pca_ilr_reduced_train = np.nan_to_num( fit_train_ilr )

        fit_test_clr = np.nan_to_num( fit_test_clr )
        fit_test_ilr = np.nan_to_num( fit_test_ilr )

        f1_pca_clr_data, roc_pca_clr_data = train_svm( pca_clr_reduced_train, train_sample_label, fit_test_clr, test_label )
        f1_pca_ilr_data, roc_pca_ilr_data = train_svm( pca_ilr_reduced_train, train_sample_label, fit_test_ilr, test_label )
        f1_clr.append( f1_pca_clr_data )
        roc_clr.append( roc_pca_clr_data )

        f1_ilr.append( f1_pca_ilr_data )
        roc_ilr.append( roc_pca_ilr_data )

        #print ( roc_original, roc_dca, roc_clr, roc_ilr)

    return ( sum ( roc_original ) / iterations ) , ( sum ( roc_original_clr ) / iterations ),  ( sum( roc_dca ) / iterations ) , ( sum( roc_clr ) / iterations ) , ( sum( roc_ilr ) / iterations )

original = []
original_clr_loss_array = []
dca = []
pca_clr = []
pca_ilr = []

for sample_size in range(20, 200, 10):
    original_loss, original_clr_loss , dca_loss, clr_loss, ilr_loss = train( 10, sample_size, 20, positive_train, negative_train, test_data, test_label)
    print ( original_loss, original_clr_loss, dca_loss, clr_loss, ilr_loss )
    original.append( original_loss )
    original_clr_loss_array.append( original_clr_loss )
    dca.append( dca_loss )
    pca_clr.append( clr_loss )
    pca_ilr.append( ilr_loss )
    print ( original_loss )
    print ( dca_loss )
    print ( clr_loss )
    print ( ilr_loss )
    print (" ")

x = list(range(20, 200, 10))
plt.title(" Needle Leaf : Landcover Classification: Dimensions 250 - 20 : Threshold : 0.5 ")
plt.plot(x, dca, label = ' DCA')
plt.plot(x, original, label = 'Original')
plt.plot(x, original_clr_loss_array, label = ' CLR ')
plt.plot(x, pca_clr, label = 'PCA_CLR ')
plt.plot(x, pca_ilr, label = 'PCA_ILR ')

plt.xlabel('Training sample size')
plt.ylabel('Area Under ROC ')
plt.legend()

plt.savefig("clr_z_normalised_needle_leaf.png")
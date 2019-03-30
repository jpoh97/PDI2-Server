#-----------------------------------------------------------------------------------------------------------------------
#---Archivo python utilizado para entrenar la maquina de soporte vectorial que clasifique las manos---------------------
#---Este lee las imagenes de la carpeta DB, y exporta el modelo (SVM) ya entrenado, para posteriormente poder-----------
#---Importarlo en el servidor-------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#---Curso: Procesamiento digital de imagenes 2 - UdeA-------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#---Por: Juan Pablo Ospina Herrera jpoh97@gmail.com---------------------------------------------------------------------
#--------Julián David Almanza Velásquez julian.almanza@udea.edu.co------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#---Versión 1, Mayo 2018------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#---Importo las librerias necesarias------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import sklearn.metrics as sm
from sklearn.externals import joblib
import cv2

#Confusion matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

#-----------------------------------------------------------------------------------------------------------------------
#---Inicializo variables------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
path="DB"
label=0
img_descs=[]
y=[]
labels=[]

#-----------------------------------------------------------------------------------------------------------------------
#---Funcion que separa los conjunto de entrenamiento, tests y validción-------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
def perform_data_split(X, y, training_idxs, test_idxs, val_idxs):
    X_train = X[training_idxs]
    X_test = X[test_idxs]
    X_val = X[val_idxs]

    y_train = y[training_idxs]
    y_test = y[test_idxs]
    y_val = y[val_idxs]

    return X_train, X_test, X_val, y_train, y_test, y_val


#-----------------------------------------------------------------------------------------------------------------------
#---Separo los identificadores de los registros de entrenamiento, tests y validación------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
def train_test_val_split_idxs(total_rows, percent_test, percent_val):
    if percent_test + percent_val >= 1.0:
        raise ValueError('percent_test and percent_val must sum to less than 1.0')

    row_range = range(total_rows)

    no_test_rows = int(total_rows*(percent_test))
    test_idxs = np.random.choice(row_range, size=no_test_rows, replace=False)
    # remove test indexes
    row_range = [idx for idx in row_range if idx not in test_idxs]

    no_val_rows = int(total_rows*(percent_val))
    val_idxs = np.random.choice(row_range, size=no_val_rows, replace=False)
    # remove validation indexes
    training_idxs = [idx for idx in row_range if idx not in val_idxs]

    print('Train-test-val split: %i training rows, %i test rows, %i validation rows' % (len(training_idxs), len(test_idxs), len(val_idxs)))

    return training_idxs, test_idxs, val_idxs


#-----------------------------------------------------------------------------------------------------------------------
#---Clusterizo las caracteristicas y convierto cada descriptor de las imagenes un visual Bag of Histogram---------------
#-----------------------------------------------------------------------------------------------------------------------
def cluster_features(img_descs, training_idxs, cluster_model):
    n_clusters = cluster_model.n_clusters

    # Concatenate all descriptors in the training set together
    training_descs = [img_descs[i] for i in training_idxs]
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)

    print ('%i descriptors before clustering' % all_train_descriptors.shape[0])

    # Cluster descriptors to get codebook
    print ('Using clustering model %s...' % repr(cluster_model))
    print ('Clustering on training set to get codebook of %i words' % n_clusters)

    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print ('done clustering. Using clustering model to generate BoW histograms for each image.')

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print ('done generating BoW histograms.')
    print(X)
    print(cluster_model)

    return X, cluster_model


#-----------------------------------------------------------------------------------------------------------------------
#---Imprimo los resultados del entrenamiento----------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
def calc_accuracy(method,label_test,pred):
    print("accuracy score for ",method,sm.accuracy_score(label_test,pred))
    print("precision_score for ",method,sm.precision_score(label_test,pred,average='micro'))
    print("f1 score for ",method,sm.f1_score(label_test,pred,average='micro'))
    print("recall score for ",method,sm.recall_score(label_test,pred,average='micro'))


#-----------------------------------------------------------------------------------------------------------------------
#---Metodo que predice mediante la maquina de soporte vectorial---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
def predict_svm(X_train, X_test, y_train, y_test):
    svc = SVC(kernel='linear')
    print("svm started")
    svc.fit(X_train, y_train)

    _ = joblib.dump(svc, 'hands_classifier2.joblib.pkl')
    clf2 = joblib.load('hands_classifier2.joblib.pkl')

    y_pred = clf2.predict(X_test)
    calc_accuracy("SVM", y_test, y_pred)
    print(svc)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    np.savetxt('submission_surf_svm2.csv', np.c_[range(1, len(y_test) + 1), y_pred, y_test], delimiter=',',
               header='ImageId,Label,TrueLabel', comments='', fmt='%d')


#-----------------------------------------------------------------------------------------------------------------------
#---Funcion que utiliza morfologia para procesar las imagenes. Extrae las caracteristicas SURF--------------------------
#-----------------------------------------------------------------------------------------------------------------------
def func(path):
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (128, 128))
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert from RGB to HSV
    # cv2.imshow("original",converted2)

    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)
    # cv2.imshow("masked",skinMask)

    skinMask = cv2.medianBlur(skinMask, 5)

    skin = cv2.bitwise_and(converted2, converted2, mask=skinMask)
    # frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    # skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    # skinGray=cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("masked2",skin)
    img2 = cv2.Canny(skin, 60, 60)
    # cv2.imshow("edge detection",img2)

    ''' 
    hog = cv2.HOGDescriptor()
    h = hog.compute(img2)
    print(len(h))

    '''
    surf = cv2.xfeatures2d.SURF_create()
    # surf.extended=True
    img2 = cv2.resize(img2, (256, 256))
    kp, des = surf.detectAndCompute(img2, None)
    # print(len(des))
    img2 = cv2.drawKeypoints(img2, kp, None, (0, 0, 255), 4)
    # plt.imshow(img2),plt.show()
    # cv2.imshow('...', img2)

    cv2.waitKey(1)
    #cv2.destroyAllWindows()
    #print(len(des))
    return des


#-----------------------------------------------------------------------------------------------------------------------
#---Metodo que imprime la matriz de confusion---------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#-----------------------------------------------------------------------------------------------------------------------
#---Codigo perteneciente al main que lee las imagenes y llama a las funciones de arriba---------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#creating desc for each file with label
for (dirpath,dirnames,filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        labels.append(dirname)
        for(direcpath,direcnames,files) in os.walk(path+"\\"+dirname):
            for file in files:
                actual_path=path+"\\\\"+dirname+"\\\\"+file
                #print(actual_path)
                des=func(actual_path)
                img_descs.append(des)
                y.append(label)
        label=label+1

y=np.array(y)

training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(len(img_descs), 0.4, 0.0)

X, cluster_model = cluster_features(img_descs, training_idxs, MiniBatchKMeans(n_clusters=150))

X_train, X_test, X_val, y_train, y_test, y_val = perform_data_split(X, y, training_idxs, test_idxs, val_idxs)

predict_svm(X_train, X_test, y_train, y_test)
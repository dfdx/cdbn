
from __future__ import print_function
import numpy as np
import matplotlib.pylab as plt
import cv2

from sklearn import linear_model, svm, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

from helpers import smartshow, list_images



# threshold = 90

# def samples(im):
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     im = cv2.resize(im, (100, 100))
#     eq = cv2.equalizeHist(im)
#     bld = cv2.GaussianBlur(eq, (3, 3), 0)
#     edg = cv2.Canny(bld, 80, 240, apertureSize=3)
#     bnr = eq.copy()
#     bnr[bnr <= threshold] = 0
#     bnr[threshold < bnr] = 1
#     return [im, eq, bld, edg, bnr]

# def run_binarize():
#     files = list_images('cropped')[:5]
#     all_samples = []
#     for fname in files:
#         im = cv2.imread(fname)
#         all_samples += samples(im)
#     smartshow(all_samples)


# def run_edges():
#     imfile = list_images('cropped')[0]
#     im = cv2.imread(imfile)
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     im = cv2.resize(im, (100, 100))
#     eq = cv2.equalizeHist(im)
#     bld = cv2.GaussianBlur(eq, (3, 3), 0)
#     samples = []
#     for thr in range(10, 90, 10):
#         edg = cv2.Canny(bld, thr, thr * 3, apertureSize=3)
#         samples.append(edg)
#     smartshow(samples)
    


# def preprocess(im):
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     im = cv2.resize(im, (100, 100))
#     return cv2.equalizeHist(im)
#     eq = cv2.equalizeHist(im)
#     bld = cv2.GaussianBlur(eq, (3, 3), 0)
#     edg = cv2.Canny(bld, 80, 240, apertureSize=3)
#     bnr = eq.copy()
#     bnr[bnr <= threshold] = 0
#     bnr[threshold < bnr] = 1
#     return im
    

# # def mkdataset(path, label):
# #     images = (cv2.resize(cv2.imread(fname), (100, 100))
# #               for fname in list_images(path))
# #     images = (preprocess(im) for im in images)
# #     X = np.vstack([im.flatten() for im in images])
# #     Y = np.repeat(label, X.shape[0])
# #     return X, Y


# def train_gender(X, Y):
#     logistic = linear_model.LogisticRegression()
#     rbm = BernoulliRBM(random_state=0, verbose=True)
#     rbm.learning_rate = 0.06
#     rbm.n_iter = 20
#     rbm.n_components = 100
#     logistic.C = 6000.0
#     # model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
#     model = Pipeline(steps=[('logistic', logistic)])
#     model.fit(X, Y)
#     return model



# def run_gender_classifier():
#     Xm, Ym = mkdataset('gender/male', 1)
#     Xf, Yf = mkdataset('gender/female', 0)
#     X = np.vstack([Xm, Xf]) # .astype(np.float32) / 256 
#     Y = np.hstack([Ym, Yf])
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                         test_size=0.1,
#                                                         random_state=100)
#     # model = linear_model.LogisticRegression()
#     model = svm.SVC(kernel='rbf')
#     model.fit(X_train, Y_train)
#     print("Results:\n%s\n" % (
#         metrics.classification_report(
#             Y_test, model.predict(X_test))))
    
    
# def run_genders():
#     Xm, Ym = mkdataset('gender/male', 1)
#     Xf, Yf = mkdataset('gender/female', 0)
#     X = np.vstack([Xm, Xf])
#     X = X.astype(np.float32) / 256 
#     Y = np.hstack([Ym, Yf])
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                         test_size=0.1,
#                                                         random_state=100)
#     model = train_gender(X_train, Y_train)
#     print()
#     print("Logistic regression using RBM features:\n%s\n" % (
#         metrics.classification_report(
#             Y_test, model.predict(X_test))))


def normalize_image(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (100, 100))
    im = cv2.equalizeHist(im)
    im = cv2.GaussianBlur(im, (3, 3), 0)
    return im

def load_data(path):
    images = (normalize_image(cv2.imread(fname)) for fname in list_images(path))
    X = np.vstack([im.flatten() for im in images])
    return X

def mkdataset(path, label):
    X = load_data(path)
    Y = np.repeat(label, X.shape[0])
    return X, Y
    
def run_auto():
    X = load_data('gender/male')
    X = X.astype(np.float32) / 256
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 2000
    rbm.fit(X)
    cimgs = [comp.reshape(100, 100) for comp in rbm.components_]
    smartshow(cimgs[:12])
    return rbm


def run_deep():
    X = load_data('gender/male')
    X = X.astype(np.float32) / 256
    rbm1 = BernoulliRBM(n_components=10000, n_iter=20, learning_rate=0.05,
                        random_state=0, verbose=True)
    rbm2 = BernoulliRBM(n_components=200, n_iter=20, learning_rate=0.05,
                        random_state=0, verbose=True)
    model = Pipeline(steps=[('rbm1', rbm1), ('rbm2', rbm)])
    model.fit(X)
    cimgs = [comp.reshape(100, 100) for comp in model.steps[1][1].components_]
    smartshow(cimgs[:12])
    return model

def estimate_n_components():
    X = load_data('gender/male')
    X = X.astype(np.float32) / 256
    n_comp_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    scores = []
    for n_comps in n_comp_list:
        rbm = BernoulliRBM(random_state=0, verbose=True)
        rbm.learning_rate = 0.06
        rbm.n_iter = 50
        rbm.n_components = 100
        rbm.fit(X)
        score = rbm.score_samples(X).mean()
        scores.append(score)
    plt.figure()
    plt.plot(n_comp_list, scores)
    plt.show()
    return n_comp_list, scores
    
# Load Data
# digits = datasets.load_digits()
# X = np.asarray(digits.data, 'float32')
# X, Y = nudge_dataset(X, digits.target)
# X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                     test_size=0.2,
#                                                     random_state=0)

# # Models we will use
# logistic = linear_model.LogisticRegression()
# rbm = BernoulliRBM(random_state=0, verbose=True)
# rbm2 = BernoulliRBM(random_state=0, verbose=True)

# classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

# ###############################################################################
# # Training

# # Hyper-parameters. These were set by cross-validation,
# # using a GridSearchCV. Here we are not performing cross-validation to
# # save time.
# rbm.learning_rate = 0.06
# rbm.n_iter = 20
# # More components tend to give better prediction performance, but larger
# # fitting time
# rbm.n_components = 100
# logistic.C = 6000.0

# # Training RBM-Logistic Pipeline
# classifier.fit(X_train, Y_train)

# # Training Logistic regression
# logistic_classifier = linear_model.LogisticRegression(C=100.0)
# logistic_classifier.fit(X_train, Y_train)

# ###############################################################################
# # Evaluation

# print()
# print("Logistic regression using RBM features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,
#         classifier.predict(X_test))))

# print("Logistic regression using raw pixel features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,
#         logistic_classifier.predict(X_test))))

# ###############################################################################
# # Plotting

# plt.figure(figsize=(4.2, 4))
# for i, comp in enumerate(rbm2.components_):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
#                interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
# plt.suptitle('100 components extracted by RBM', fontsize=16)
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# plt.show()

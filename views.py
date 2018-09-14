from django.shortcuts import render, redirect, reverse, render_to_response, HttpResponse
from mlaas import forms
from mlaas import models
import numpy as np
import pandas as pd
# os.path.abspath(os.path.dirname(__file__))
# module_dir = os.path.dirname(__file__)  # get current directory
# file_path = os.path.join(module_dir, 'cf.csv')
#
# f = open('knn2.pkl', 'r')
# myfile=File(f)
# a=models.DataSet(DataSetName='knn2')
# a.DataSetFile=myfile
# a.save()
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# b=models.DataSet.objects.get(DataSetName='knn')
# content = b.DataSetFile.read()
# knn = pickle.loads(content)

# fid=knn.score(x_test, y_test)
######################  Sklearn Libraries   ################
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVR, SVC
from django.core.files import File


def t(request):
    if request.method == "POST":
        # f = open('iris.csv', 'r')
        # myfile=File(f)
        # a=models.DataSet(DataSetName='irsssssis')
        # a.DataSetFile=myfile
        # a.save()
        # a=models.DataSet.objects.get(DataSetName='iris')
        # print(a.Target_Type)

        f1 = forms.f1(request.POST, prefix="a")
        f2 = forms.f1(request.POST, prefix="b")
        if f1.is_valid():
            return HttpResponse(f1.cleaned_data['s'])
    else:
        f1 = forms.f1()
        f2 = forms.f1()
    return render(request, 'mlaas/test.html', {'f1': f1, 'f2': f2})


def temp(request):
    if request.method == "POST":
        Data_Model_Form = forms.DataSet_Model_Selection_Form(request.POST, prefix="a")
        kNNCatForm = forms.kNNCat_Form(request.POST, prefix="b")
        kNNRegForm = forms.kNNReg_Form(request.POST, prefix="c")
        GaussianNBForm = forms.GaussianNB_form(request.POST, prefix="d")
        BernoulliNBForm = forms.BernoulliNB_form(request.POST, prefix="e")
        MultinomialNBForm = forms.MultinomialNB_form(request.POST, prefix="f")
        SVCForm = forms.SVC_form(request.POST, prefix="g")
        SVRForm = forms.SVR_form(request.POST, prefix="h")

        if Data_Model_Form.is_valid():

            model_type = Data_Model_Form.cleaned_data['Classification_Model_Type']
            data = Data_Model_Form.cleaned_data['UserDataSet']

            if model_type == 'kNN_regression':
                if kNNRegForm.is_valid():
                    accuracy = kNN_Model_regression(kNNRegForm, data)
                    return HttpResponse(accuracy)

            if model_type == 'kNN_categorical':
                if kNNCatForm.is_valid():
                    accuracy = kNN_Model_categorical(kNNCatForm, data)
                    return HttpResponse(accuracy)

            if model_type == 'GaussianNB':
                if GaussianNBForm.is_valid():
                    accuracy = GaussianNB_model(GaussianNBForm, data)
                    return HttpResponse(accuracy)

            if model_type == 'BernoulliNB':
                if BernoulliNBForm.is_valid():
                    accuracy = BernoulliNB_model(BernoulliNBForm, data)
                    return HttpResponse(accuracy)

            if model_type == 'MultinomialNB':
                if MultinomialNBForm.is_valid():
                    accuracy = MultinomialNB_model(MultinomialNBForm, data)
                    return HttpResponse(accuracy)

            if model_type == 'SVC':
                if SVCForm.is_valid():
                    accuracy = SVC_model(MultinomialNBForm, data)
                    return HttpResponse(accuracy)

            if model_type == 'SVR':
                if SVRForm.is_valid():
                    accuracy = SVR_model(SVRForm, data)
                    return HttpResponse(accuracy)

        else:
            HttpResponse("Wrong Value")
    else:
        Data_Model_Form = forms.DataSet_Model_Selection_Form(prefix="a")
        kNNCatForm = forms.kNNCat_Form(prefix="b")
        kNNRegForm = forms.kNNReg_Form(prefix="c")
        GaussianNBForm = forms.GaussianNB_form(prefix="d")
        BernoulliNBForm = forms.BernoulliNB_form(prefix="e")
        MultinomialNBForm = forms.MultinomialNB_form(prefix="f")
        SVCForm = forms.SVC_form(prefix="g")
        SVRForm = forms.SVR_form(prefix="h")
    return render(request, 'mlaas/Model_Page.html', {'Data_Model_Form': Data_Model_Form,
                                                     'kNNCatForm': kNNCatForm,
                                                     'kNNRegForm': kNNRegForm,
                                                     'GaussianNBForm': GaussianNBForm,
                                                     'BernoulliNBForm': BernoulliNBForm,
                                                     'MultinomialNBForm': MultinomialNBForm,
                                                     'SVCForm': SVCForm,
                                                     'SVRForm': SVRForm})


def kNN_Model_regression(form, data):
    ###########################Form Varibles###############
    n_neighbors = int(form.cleaned_data['n_neighbors'])
    tstSplit = float(form.cleaned_data['Test_Train_Split'])
    weights = form.cleaned_data['weights']
    algorithm = form.cleaned_data['algorithm']
    leaf_size = int(form.cleaned_data['leaf_size'])
    p = int(form.cleaned_data['p'])
    metric = form.cleaned_data['metric']
    n_jobs = int(form.cleaned_data['n_jobs'])
    ################################################
    #################### Data Set ##################
    temp_data = pd.read_csv(data.DataSetFile)
    target = temp_data['target']
    data = temp_data.drop(['target'], axis=1)
    ################################################
    ################# Model########################
    accuracy = 0
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=tstSplit, random_state=42, stratify=target)

    knn = KNeighborsRegressor(n_neighbors=n_neighbors,
                              weights=weights,
                              algorithm=algorithm,
                              leaf_size=leaf_size,
                              p=p,
                              metric=metric,
                              n_jobs=n_jobs)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    accuracy = knn.score(x_test, y_test)

    return accuracy


def kNN_Model_categorical(form, data):
    ###########################Form Varibles###############
    n_neighbors = int(form.cleaned_data['n_neighbors'])
    tstSplit = float(form.cleaned_data['Test_Train_Split'])
    weights = form.cleaned_data['weights']
    algorithm = form.cleaned_data['algorithm']
    leaf_size = int(form.cleaned_data['leaf_size'])
    p = int(form.cleaned_data['p'])
    metric = form.cleaned_data['metric']
    n_jobs = int(form.cleaned_data['n_jobs'])
    ################################################
    #################### Data Set ##################
    temp_data = pd.read_csv(data.DataSetFile)
    target = temp_data['target']
    data = temp_data.drop(['target'], axis=1)
    ################################################
    ################# Model########################
    accuracy = 0
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=tstSplit, random_state=42, stratify=target)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                               weights=weights,
                               algorithm=algorithm,
                               leaf_size=leaf_size,
                               p=p,
                              metric=metric,
                               n_jobs=n_jobs)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    accuracy = knn.score(x_test, y_test)

    return accuracy


##################################################################################################################################
# Naive Bayes
# Gaussian NB When dealing with continuous data
def GaussianNB_model(form, data):
    ###########################Form Varibles###############
    tstSplit = float(form.cleaned_data['Test_Train_Split'])
    #################### Data Set ##################
    temp_data = pd.read_csv(data.DataSetFile)
    target = temp_data['target']
    data = temp_data.drop(['target'], axis=1)
    ################################################

    ################################################

    ################# Model########################
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=tstSplit, random_state=42, stratify=target)
    GNB = GaussianNB()
    GNB.fit(x_train, y_train)
    y_pred = GNB.predict(x_test)
    accuracy = GNB.score(x_test, y_test)
    return accuracy
    # render(request, 'mlaas/Model_Accuracy.html', {'accuracy': accuracy})


# Bernoulli Naive Bayes: this class requires samples to be represented as binary-valued feature vectors
def BernoulliNB_model(form, data):
    ###########################Form Varibles###############
    tstSplit = float(form.cleaned_data['Test_Train_Split'])
    binarize = float(form.cleaned_data['binarize'])
    alpha = float(form.cleaned_data['alpha'])
    fit_prior = bool(form.cleaned_data['fit_prior'])
    #################### Data Set ##################
    temp_data = pd.read_csv(data.DataSetFile)
    target = temp_data['target']
    data = temp_data.drop(['target'], axis=1)
    ################################################

    ################################################

    ################# Model########################
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=tstSplit, random_state=42, stratify=target)
    GNB = BernoulliNB(binarize=binarize,
                      alpha=alpha,
                      fit_prior=fit_prior)
    GNB.fit(x_train, y_train)
    y_pred = GNB.predict(x_test)
    accuracy = GNB.score(x_test, y_test)
    return accuracy


# categorical target
def MultinomialNB_model(form, data):
    ###########################Form Varibles###############
    tstSplit = float(form.cleaned_data['Test_Train_Split'])
    alpha = float(form.cleaned_data['alpha'])
    fit_prior = bool(form.cleaned_data['fit_prior'])
    #################### Data Set ##################
    temp_data = pd.read_csv(data.DataSetFile)
    target = temp_data['target']
    data = temp_data.drop(['target'], axis=1)
    ################################################

    ################################################

    ################# Model########################
    GNB = MultinomialNB(alpha=alpha,
                        fit_prior=fit_prior)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=tstSplit, random_state=42, stratify=target)
    GNB.fit(x_train, y_train)
    y_pred = GNB.predict(x_test)
    accuracy = GNB.score(x_test, y_test)
    return accuracy


##################################################################################################################################
# Support Vecor Machine
# Multi-class classification
def SVC_model(form, data):
    ##########Form Varibles###############
    tstSplit = float(form.cleaned_data['Test_Train_Split'])
    decision_function_shape = form.cleaned_data['decision_function_shape']
    probability = bool(form.cleaned_data['probability'])
    kernel = form.cleaned_data['kernel']
    degree = int(form.cleaned_data['degree'])
    coef0 = float(form.cleaned_data['coef0'])
    tol = float(form.cleaned_data['tol'])
    C = float(form.cleaned_data['C'])
    shrinking = bool(form.cleaned_data['shrinking'])
    cache_size = float(form.cleaned_data['cache_size'])
    verbose = bool(form.cleaned_data['verbose'])
    max_iter = int(form.cleaned_data['max_iter'])
    ################################################
    #################### Data Set ##################
    temp_data = pd.read_csv(data.DataSetFile)
    target = temp_data['target']
    data = temp_data.drop(['target'], axis=1)
    ################################################

    ################# Model########################
    svr_model = SVC(C=C,
                    kernel=kernel,
                    degree=degree,
                    coef0=coef0,
                    shrinking=shrinking,
                    probability=probability,
                    tol=tol,
                    cache_size=cache_size,
                    verbose=verbose,
                    max_iter=max_iter,
                    decision_function_shape=decision_function_shape)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=tstSplit, random_state=42, stratify=target)
    svr_model.fit(x_train, y_train)
    y_pred = svr_model.predict(x_test)
    accuracy = svr_model.score(x_test, y_test)

    return accuracy


# Regression
def SVR_model(form, data):
    ##########Form Varibles###############
    tstSplit = float(form.cleaned_data['Test_Train_Split'])
    kernel = form.cleaned_data['kernel']
    degree = int(form.cleaned_data['degree'])
    coef0 = float(form.cleaned_data['coef0'])
    tol = float(form.cleaned_data['tol'])
    C = float(form.cleaned_data['C'])
    epsilon = float(form.cleaned_data['epsilon'])
    shrinking = bool(form.cleaned_data['shrinking'])
    cache_size = float(form.cleaned_data['cache_size'])
    verbose = bool(form.cleaned_data['verbose'])
    max_iter = int(form.cleaned_data['max_iter'])
    ################################################
    #################### Data Set ##################
    temp_data = pd.read_csv(data.DataSetFile)
    target = temp_data['target']
    data = temp_data.drop(['target'], axis=1)
    ################################################

    ################# Model########################
    svr_model = SVR(C=C,
                    kernel=kernel,
                    degree=degree,
                    coef0=coef0,
                    shrinking=shrinking,
                    tol=tol,
                    epsilon=epsilon,
                    cache_size=cache_size,
                    verbose=verbose,
                    max_iter=max_iter)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=tstSplit, random_state=42, stratify=target)
    svr_model.fit(x_train, y_train)
    y_pred = svr_model.predict(x_test)
    accuracy = svr_model.score(x_test, y_test)

    return accuracy
###########################     Data View       #############################


# def DataSet_Statistics(request):
#     data=datasets.load_iris()

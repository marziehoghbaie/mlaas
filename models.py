from django.db import models

# Create your models here.

TTS_Select = (
    ('0.1', '0.1'),
    ('0.2', '0.2'),
    ('0.3', '0.3'),
    ('0.4', '0.4'),
    ('0.5', '0.5'),
)

TargetType= (
    ('Regression','Regression'),
    ('Binary','Binary'),
    ('Categorical','Categorical'),
)

class DataSet(models.Model):
    DataSetName=models.CharField(max_length=100)
    Target_Type=models.CharField(max_length=20, choices=TargetType, default='Categorical')
    DataSetFile=models.FileField(upload_to='media')
    def __str__(self):
        return self.DataSetName

class MultinomialNB_form(models.Model):
    alpha = models.FloatField(default=1.0)
    fit_prior=models.BooleanField(default=True)
    Test_Train_Split = models.FloatField(choices=TTS_Select)


class GaussianNB_form(models.Model):
    Test_Train_Split = models.FloatField(choices=TTS_Select)


class Model_Selection_Model(models.Model):
        # kNN, SVM, NB
        # ('value retrived from the form or stored in DB', 'human readable value represented to user')
    ClassificationTypes = (
        ('kNN', (
            ('kNN_regression', 'kNN regression'),
            ('kNN_categorical', 'kNN categorical')
        )
         ),
        ('Naive bayes', (
            ('GaussianNB', 'GaussianNB'),
            ('BernoulliNB', 'BernoulliNB'),
            ('MultinomialNB', 'MultinomialNB')
        )
         ),
        ('Support Vector Machine', (
            ('SVC', 'SVC'),
            ('SVR', 'SVR'),
        )),
        )
    Classification_Model_Type=models.CharField(choices=ClassificationTypes,default='kNN_categorical',max_length=200)
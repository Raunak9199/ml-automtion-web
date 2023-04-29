import django
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
from django.core.files.storage import FileSystemStorage
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io
import matplotlib.pyplot as plt; plt.rcdefaults()
from plotly.offline import plot
from plotly.graph_objs import Scatter,bar,Pie,Histogram,Heatmap
import plotly.express as px


global data
global features
global target
global column

def ml_home(request):
    return render(request, "ml_home.html")

def upload(request):
    global data
    global column
    column = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
        column['url'] = fs.url(uploaded_file.name)
        data = pd.read_csv(r'/home/raunak/Desktop/sem8/machine learning' + fs.url(uploaded_file.name))
        print(data.head())
        column['a'] = data.columns
        features = request.POST.getlist('Cars[]')
        column['features'] = features
        for i in data.columns:
            if type(data[i][0]) == np.float64 :
                data[i].fillna(data[i].mean(), inplace=True)
            elif type(data[i][0]) == np.int64 :
                data[i].fillna(data[i].median(), inplace=True)
            elif type(data[i][0]) == type(""):
                imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                s = imp.fit_transform(data[i].values.reshape(-1, 1))
                data[i] = s



    return render(request, ['upload.html', 'home.html'],column)

def home(request):
    return render(request, "home.html")


def result(request):
    if 'knn' in request.POST:
        return render(request, "knn.html")
    if 'DTC' in request.POST:
        return render(request, "Decision.html")
    if 'LOG' in request.POST:
        return render(request, "logisticreg.html")
    if 'LIN' in request.POST:
        return render(request, "linearreg.html")
    if 'MLP' in request.POST:
        return render(request, "MLPClassifier.html")
    if 'SVC' in request.POST:
        return render(request, "SVC_Classifier.html")
    if 'MULNB' in request.POST:
        return render(request, "naivebayes.html")
    if 'DTR' in request.POST:
        return render(request, "DecisionTreeREG.html")
    if 'km' in request.POST:
        return render(request,"kmeans.html")


def result1(request):
    global data
    global features
    global target
    if 'submit' in request.GET:
        global data
        lis = []
        lis.append(request.GET['SL'])

        feature=[]
        feature.append(request.GET['fc'])


        target=[]
        target.append(request.GET['tc'])

        s = lis[0].split(",")
        s=[float (i) for i in s]
        feature= feature[0].split(",")
        feature=[int(i) for i in feature]
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['neighbour'])
        hyp.append(request.GET['weights'])
        hyp.append(request.GET['algorithm'])
        hyp.append(request.GET['xaxis'])
        hyp.append(request.GET['yaxis'])
        hyp.append(request.GET['graph'])

        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        x = data.iloc[:,feature].values
        y = data.iloc[:,int(target[0])].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = KNeighborsClassifier(n_neighbors=int(hyp[1]), weights=hyp[2], algorithm=hyp[3])
        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100
        u=hyp[6]


        x_data = data.iloc[:,int(hyp[4])]
        y_data = data.iloc[:,int(hyp[5])]
        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            df = data.corr()
            fig = px.imshow(df)
            plot_div=fig.show()

        return render(request, "knn.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div})
    if 'submit1' in request.GET:
        lis = []
        lis.append(request.GET['SL'])
        feature=[]

        target=[]
        feature.append(request.GET['fc'])
        print(feature)
        target.append(request.GET['tc'])

        s = lis[0].split(",")
        s=[float (i) for i in s]
        feature= feature[0].split(",")
        feature=[int(i) for i in feature]
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['criterion'])
        hyp.append(request.GET['splitter'])
        hyp.append(request.GET['max'])
        hyp.append(request.GET['xaxis'])
        hyp.append(request.GET['yaxis'])
        hyp.append(request.GET['graph'])

        import numpy as np
        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        x = data.iloc[:,feature].values
        y = data.iloc[:,int(target[0])].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = DecisionTreeClassifier(criterion=hyp[1], splitter=hyp[2], max_features=hyp[3])

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100

        x_data = data.iloc[:,int(hyp[4])]
        y_data = data.iloc[:,int(hyp[5])]
        u=hyp[6]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            df = data.corr()
            fig = px.imshow(df)
            plot_div=fig.show()
        return render(request, "Decision.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1,'plot_div': plot_div})


    if 'submit2' in request.GET:
        lis = []
        lis.append(request.GET['SL'])
        feature=[]
        feature.append(request.GET['fc'])

        target=[]
        target.append(request.GET['tc'])

        s = lis[0].split(",")
        s=[float (i) for i in s]
        feature= feature[0].split(",")
        feature=[int(i) for i in feature]
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['alpha'])
        hyp.append(request.GET['fit'])
        hyp.append(request.GET['xaxis'])
        hyp.append(request.GET['yaxis'])
        hyp.append(request.GET['graph'])

        import numpy as np
        import pandas as pd
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        x = data.iloc[:,feature].values
        y = data.iloc[:,int(target[0])].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = MultinomialNB(alpha=int(hyp[1]), fit_prior=bool(hyp[2]))

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test)*100
        acc1 = cls.score(x_train,y_train)*100

        x_data = data.iloc[:,int(hyp[3])]
        y_data = data.iloc[:,int(hyp[4])]
        u=hyp[5]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            df = data.corr()
            fig = px.imshow(df)
            plot_div=fig.show()
        return render(request, "naivebayes.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1,'plot_div': plot_div})


    if 'submit3' in request.GET:
        lis = []
        lis.append(request.GET['SL'])
        feature=[]
        feature.append(request.GET['fc'])

        target=[]
        target.append(request.GET['tc'])

        s = lis[0].split(",")
        s = [float(i) for i in s]
        feature= feature[0].split(",")
        feature=[int(i) for i in feature]
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['solver'])
        hyp.append(request.GET['penalty'])
        hyp.append(request.GET['xaxis'])
        hyp.append(request.GET['yaxis'])
        hyp.append(request.GET['graph'])

        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        x = data.iloc[:,feature].values
        y = data.iloc[:,int(target[0])].values

        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = float(hyp[0]))


        cls = LogisticRegression(solver=hyp[1],penalty=hyp[2])


        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test)*100
        acc1 = cls.score(x_train,y_train)*100

        x_data = data.iloc[:,int(hyp[3])]
        y_data = data.iloc[:,int(hyp[4])]
        u=hyp[5]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            df = data.corr()
            fig = px.imshow(df)
            plot_div=fig.show()
        return render(request, "logisticreg.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1,'plot_div': plot_div})

    if 'submit4' in request.GET:
        
        lis = []
        lis.append(request.GET['SL'])
        feature=[]
        feature.append(request.GET['fc'])

        target=[]
        target.append(request.GET['tc'])

        s = lis[0].split(",")
        s = [float(i) for i in s]
        feature= feature[0].split(",")
        feature=[int(i) for i in feature]
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['fit_intercept'])
        # hyp.append(request.GET['normalize'])
        hyp.append(request.GET['xaxis'])
        hyp.append(request.GET['yaxis'])
        hyp.append(request.GET['graph'])

        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        x = data.iloc[:,feature].values
        y = data.iloc[:,int(target[0])].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))
        # normalize=bool(hyp[2])
        sc=StandardScaler()
        x_train= sc.fit_transform(x_train)
        x_test= sc.transform(x_test)
        cls = LinearRegression(fit_intercept=bool(hyp[1]))

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test)*100
        acc1 = cls.score(x_train,y_train)*100

        x_data = data.iloc[:,int(hyp[3])]
        y_data = data.iloc[:,int(hyp[4])]
        u=hyp[5]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            df = data.corr()
            fig = px.imshow(df)
            plot_div=fig.show()
        return render(request, "linearreg.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1,'plot_div': plot_div})

    # MLP Classifier
    # if 'submit5' in request.GET:
    #     lis = []
    #     lis.append(request.GET['SL'])
    #     feature=[]
    #     feature.append(request.GET['fc'])

    #     target=[]
    #     target.append(request.GET['tc'])

    #     s = lis[0].split(",")
    #     s = [float(i) for i in s]
    #     feature= feature[0].split(",")
    #     feature=[int(i) for i in feature]
    #     hyp = []
    #     hyp.append(request.GET['Split'])
    #     hyp.append(request.GET['activation'])
    #     hyp.append(request.GET['solver'])
    #     hyp.append(request.GET['learning_rate'])
    #     hyp.append(request.GET['xaxis'])
    #     hyp.append(request.GET['yaxis'])
    #     hyp.append(request.GET['graph'])

    #     import numpy as np
    #     import pandas as pd
    #     from sklearn.neural_network import MLPClassifier
    #     from sklearn.preprocessing import LabelEncoder
    #     from sklearn.model_selection import train_test_split
    #     from sklearn.metrics import accuracy_score
    #     x = data.iloc[:,feature].values
    #     y = data.iloc[:,int(target[0])].values

    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

    #     cls = MLPClassifier(activation=hyp[1],solver=hyp[2],learning_rate=hyp[3])

    #     cls.fit(x_train, np.ravel(y_train))
    #     y_pred = cls.predict([s])

    #     print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
    #     print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
    #     acc = cls.score(x_test, y_test)*100
    #     acc1 = cls.score(x_train,y_train)*100

    #     x_data = data.iloc[:,int(hyp[4])]
    #     y_data = data.iloc[:,int(hyp[5])]
    #     u=hyp[6]

    #     if u == 'scatter':
    #         plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
    #     if u == 'line':
    #         plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
    #     if u == 'pie':
    #         plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
    #     if u == 'hist':
    #         plot_div=plot([Histogram(x=x_data)],output_type='div')
    #     if u == 'corr':
    #         df = data.corr()
    #         fig = px.imshow(df)
    #         plot_div=fig.show()
    #     return render(request, "MLPClassifier.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1,'plot_div': plot_div})


    if 'submit6' in request.GET:
        lis = []
        lis.append(request.GET['SL'])
        feature=[]
        feature.append(request.GET['fc'])

        target=[]
        target.append(request.GET['tc'])

        s = lis[0].split(",")
        s = [float(i) for i in s]
        feature= feature[0].split(",")
        feature=[int(i) for i in feature]
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['C'])
        hyp.append(request.GET['kernel'])
        hyp.append(request.GET['gamma'])
        hyp.append(request.GET['xaxis'])
        hyp.append(request.GET['yaxis'])
        hyp.append(request.GET['graph'])

        import numpy as np
        import pandas as pd
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        x = data.iloc[:,feature].values
        y = data.iloc[:,int(target[0])].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = SVC(C=float(hyp[1]),kernel=hyp[2],gamma=hyp[3])

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test)*100
        acc1 = cls.score(x_train,y_train)*100
        print(data.head())
        x_data = data.iloc[:,int(hyp[4])]
        y_data = data.iloc[:,int(hyp[5])]
        u=hyp[6]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            df = data.corr()
            fig = px.imshow(df)
            plot_div=fig.show()
        return render(request, "SVC_Classifier.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1,'plot_div': plot_div})

    # Decision Tree Regressor
    if 'submit7' in request.GET:
        lis = []
        lis.append(request.GET['SL'])
        feature=[]
        feature.append(request.GET['fc'])

        target=[]
        target.append(request.GET['tc'])

        s = lis[0].split(",")
        s = [float(i) for i in s]
        feature= feature[0].split(",")
        feature=[int(i) for i in feature]
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['criterion'])
        hyp.append(request.GET['splitter'])
        hyp.append(request.GET['max'])
        hyp.append(request.GET['xaxis'])
        hyp.append(request.GET['yaxis'])
        hyp.append(request.GET['graph'])

        import numpy as np
        import pandas as pd
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        x = data.iloc[:,feature].values
        y = data.iloc[:,int(target[0])].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = DecisionTreeRegressor(criterion=hyp[1],splitter=hyp[2],max_features=hyp[3])

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test)*100
        acc1 = cls.score(x_train,y_train)*100

        x_data = data.iloc[:,int(hyp[4])]
        y_data = data.iloc[:,int(hyp[5])]
        u=hyp[6]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            df = data.corr()
            fig = px.imshow(df)
            plot_div=fig.show()
        return render(request, "DecisionTreeREG.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1,'plot_div': plot_div})

    if 'submit8' in request.GET:
        lis = []
        lis.append(request.GET['SL'])
        feature=[]
        feature.append(request.GET['fc'])

        target=[]
        target.append(request.GET['tc'])

        s = lis[0].split(",")
        s = [float(i) for i in s]
        feature= feature[0].split(",")
        feature=[int(i) for i in feature]
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['n_clusters'])
        hyp.append(request.GET['algorithm'])
        hyp.append(request.GET['xaxis'])
        hyp.append(request.GET['yaxis'])
        hyp.append(request.GET['graph'])

        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        x = data.iloc[:,feature].values
        y = data.iloc[:,int(target[0])].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = KMeans(n_clusters=int(hyp[1]),algorithm=hyp[2])

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test)*100
        acc1 = cls.score(x_train,y_train)*100

        x_data = data.iloc[:,int(hyp[3])]
        y_data = data.iloc[:,int(hyp[4])]
        u=hyp[5]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            df = data.corr()
            fig = px.imshow(df)
            plot_div=fig.show()
        return render(request, "kmeans.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1,'plot_div': plot_div})


def Decision(request):
    return render(request, "Decision.html")
def naivebayes(request):
    return render(request, "naivebayes.html")
def logisticreg(request):
    return render(request, "logisticreg.html")
def linearreg(request):
    return render(request, "linearreg.html")
def MLPClassifier(request):
    return render(request, "MLPClassifier.html")
def SVC_Classifier(request):
    return render(request, "SVC_Classifier.html")
def DecisionTreeREG(request):
    return render(request, "DecisionTreeREG.html")
def kmeans(request):
    return render(request, "kmeans.html")

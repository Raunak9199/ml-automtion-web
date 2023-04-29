
from django.contrib import admin
from django.urls import path
from . import view
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('admin/', admin.site.urls),
    path('',view.ml_home,name="ml_home"),
    path('upload/',view.upload,name="upload"),
    path('model/',view.result,name="knn"),
    path('result/',view.result1,name="result"),
    path('Decision/',view.Decision,name="Decision"),
    path('naivebayes/',view.naivebayes,name="naivebayes"),
    path('logisticreg/',view.logisticreg,name="logisticreg"),
    path('linearreg/',view.linearreg,name="linearreg"),
    path('MLPClassifier/',view.MLPClassifier,name="MLPClassifier"),
    path('SVC_Classifier/',view.SVC_Classifier,name="SVC_Classifier"),
    path('DecisionTreeREG/',view.DecisionTreeREG,name="DecisionTreeREG"),
    path('kmeans/',view.kmeans,name="kmeans"),
    path('home/',view.home,name='home'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

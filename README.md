Dieses Projekt analysiert Daten von Fahrradfahrten, die aus fit-Dateien gelesen werden. 

1\. Stelle sicher, dass die fit-Dateien im Unterordner Fitnessdaten im Ordner Source\_Data im Projektverzeichnis sind. Und dass eine csv-Datei mit Temperaturdaten im Ordner Source\_Data liegt.

2\. Führe **load\_data.py** aus. Dies speichert die ausgelesenen Daten in zwei csv-Dateien im Ordner Dataframes. session\_records.csv speichert einen Datenpunkt für jede Fahrt und enthält z.B. die Dauer der Fahrt und die Durchschnitts- und Maximalleistung. workouts\_per\_second.csv speichert einen Datenpunkt für jede Sekunde aller Fahrten und enthält z.B Leistung und Herzfrequenz zu dem Zeitpunkt.

3\. Führe **clean\_and\_join\_data.ipynb** aus. Dies entfernt unnötige Spalten und Datenpunkte, die nicht zu gebrauchen sind. Außerdem werden die Fitnessdaten mit Temperaturdaten gejoint. Ausgegeben wird merged\_data.csv im Ordner Dataframes.

4\. Führe **clean\_heart\_rate\_data.ipynb** aus. Dies kümmert sich um fehlende Herzfrequenzdaten in den Daten und gibt processed\_data.csv im Ordner Dataframes aus.

5\. Führe **analysis.ipynb** aus.

Außerdem kann eine Herzfrequenzvorhersage gemacht werden. Dafür wird ein Model auf 10-Minuten-Fenstern trainiert, um die Herzfrequenz der nächsten Minute vorherzusagen. 
Um das Model zu trainieren, führe zuerst **load\_data.py**. In **config.py** können Parameter geändert werden. In **run\_experiments.py** kann die Architektur der neuronalen Netze geändert werden. Führe dann **run\_experiments.py** aus.
**feature\_importance.py** zeigt an, welche Features den meisten Einfluss haben. Das wird daraan gemessen, wie stark der mean absolute error steigt, wenn ein Feature durch Permutation zufällige Werte enthält.
**large\_power\_changes.py** zeigt an, wei gut das Model die Herzfrequenz vorhersagt, wenn die Power stark erhöht oder reduziert wurde.
**visualize\_forecast.py** zeigt an Beispielfahrten, welche Herzfrequenzen das Model vorhersagt und wie weit es daneben liegt.

Dieses Projekt analysiert Daten von Fahrradfahrten, die aus fit-Dateien gelesen werden. 

1\. Stelle sicher, dass die fit-Dateien im Unterordner Fitnessdaten im Ordner Source\_Data im Projektverzeichnis sind. Und dass eine csv-Datei mit Temperaturdaten im Ordner Source\_Data liegt.

2\. Führe **import\_data.ipynb** aus. Dies speichert die ausgelesenen Daten in zwei csv-Dateien im Ordner Dataframes. session\_records.csv speichert einen Datenpunkt für jede Fahrt und enthält z.B. die Dauer der Fahrt und die Durchschnitts- und Maximalleistung. workouts\_per\_second.csv speichert einen Datenpunkt für jede Sekunde aller Fahrten und enthält z.B Leistung und Herzfrequenz zu dem Zeitpunkt.

3\. Führe **clean\_and\_join\_data.ipynb** aus. Dies entfernt unnötige Spalten und Datenpunkte, die nicht zu gebrauchen sind. Außerdem werden die Fitnessdaten mit Temperaturdaten gejoint. Ausgegeben wird merged\_data.csv im Ordner Dataframes.

4\. Führe **clean\_heart\_rate\_data.ipynb** aus. Dies kümmert sich um fehlende Herzfrequenzdaten in den Daten und gibt processed\_data.csv im Ordner Dataframes aus.

5\. Führe **analysis.ipynb** aus.


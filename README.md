# Time Series Anomaly Detection su dati elettrici reali

Anomaly detection su time series elettriche reali. Il repository unisce quattro approcci complementari:

1. **Feature-based anomaly detection**: sliding windows, statistiche tempo/frequenza/dinamica, scaling, Isolation Forest.
2. **Matrix Profile**: ricerca di subsequence discordanti e stima dell'inizio grezzo dell'anomalia.
3. **ARIMA/SARIMAX**: anomaly detection forecast-based sui residui.
4. **Prophet**: anomaly detection forecast-based con intervalli previsionali.

I dati reali inclusi sono quelli del progetto originale:

- `data/raw/ASILO_20240426.csv`
- `data/raw/ED14_20240426.csv`
- `data/raw/ED18_20240426.csv`

Le notebook originali sono conservate in `notebooks/`; il codice legacy utile è in `legacy/`. Il file di fetch da MongoDB è stato trasformato in un template senza credenziali hardcoded.

## Results Report

The report analyzes the real-world energy time series processed by the pipeline and discusses:

Open the report locally:

```bash
open report_analisi_critica_timeseries.html
```
## Installazione

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

Installazione minima senza Prophet/Matrix Profile:

```bash
pip install -e ".[dev]"
```

Extra disponibili:

```bash
pip install -e ".[matrix]"    # STUMPY / Matrix Profile
pip install -e ".[forecast]"  # statsmodels + prophet
pip install -e ".[fetch]"     # pymongo per export dati
```

## Struttura

```text
data/raw/                  CSV reali del progetto
src/tsad_feature/           package principale
  real_data.py              loader e preprocessing dati reali
  features.py               feature tempo/frequenza/dinamica
  detectors.py              Isolation Forest + scaler
  matrix_profile.py         STUMPY Matrix Profile
  forecast.py               ARIMA/SARIMAX e Prophet
  ensemble.py               fusione voti e score
  cli.py                    interfaccia da terminale
notebooks/                  notebook originali del progetto caricato
legacy/                     funzioni originali riordinate/sanificate
outputs/                    output esperimenti, non versionare se pesanti
```

## 1. Esplora le serie reali disponibili

```bash
tsad-feature real-summary --data-dir data/raw --out outputs/series_summary.csv
```

Esempi di meter presenti:

- `ASILO.GEN`
- `ARCH_FM`
- `ARCH_CDZ`
- `ARCH_GEN_corC`
- `ED18_FM`
- `ED18_CDZ`
- `ED18_CAPFM`

## 2. Esegui la pipeline completa sui dati reali

Esempio su potenza attiva totale `TotW` del sensore `ARCH_FM`:

```bash
tsad-feature real-run \
  --csv data/raw/ED14_20240426.csv \
  --meter-id ARCH_FM \
  --value-col TotW \
  --resample 15min \
  --agg mean \
  --window-size 96 \
  --overlap 0.5 \
  --contamination 0.03 \
  --arima-order 2,1,2 \
  --seasonal-order 1,0,1,96 \
  --arima-threshold-z 4.5 \
  --prophet-threshold-z 3.5 \
  --mp-zscore-threshold 3.0 \
  --mp-top-k 30 \
  --min-votes 2 \
  --outdir outputs/ED14_ARCH_FM_TotW
```

Con `resample=15min`, `window-size=96` corrisponde a una finestra giornaliera. Per finestre più locali puoi usare `--window-size 24` cioè 6 ore.

## 3. Output principali

La pipeline scrive:

```text
clean_series.csv                       serie filtrata/resamplata
feature_isolation_forest_scores.csv     score su finestre
matrix_profile.csv                      profilo STUMPY, se disponibile
matrix_profile_discords.csv             subsequence discordanti
matrix_profile_onsets.csv               onset grezzi da Matrix Profile
arima_scores.csv                        residui e anomalie ARIMA
prophet_scores.csv                      residui, forecast interval e anomalie Prophet
ensemble_sample_scores.csv              score combinato per timestamp
ensemble_anomaly_events.csv             eventi anomalici raggruppati
run_metadata.json                       configurazione e diagnostica run
report.html                             report HTML automatico con metriche, eventi e grafici
*.png                                   grafici diagnostici
```

L'ensemble usa una logica a voti: feature detector, Matrix Profile, ARIMA e Prophet producono flag separati; un campione è anomalico se almeno `--min-votes` detector concordano. Gli eventi finali vengono anche classificati automaticamente in categorie interpretabili come `HIGH_LOAD_FORECAST_RESIDUAL`, `LOW_LOAD_SHAPE_PROFILE_SHIFT`, `SHAPE_BASED_DISCORD`, `FORECAST_RESIDUAL` e `FEATURE_WINDOW_OUTLIER`.



| Metodo | Input | Cosa trova meglio | Note |
|---|---|---|---|
| Isolation Forest su feature | finestre + descrittori | pattern statistici anomali | non richiede label |
| Matrix Profile | segnale grezzo | shape discord locali | ottimo per onset e subsequence rare |
| ARIMA/SARIMAX | segnale ordinato temporalmente | residui previsivi anomali | buono per serie stazionarie/differenziabili |
| Prophet | timestamp + valore | trend/stagionalità + outlier | utile su dati business/calendariali |


## 4.1 Miglioramenti implementati nella versione avanzata

Questa versione include le correzioni emerse dall'analisi dei risultati reali:

- **ARIMA/SARIMAX stagionale giornaliero**: default `--seasonal-order 1,0,1,96` per dati a 15 minuti, dove 96 campioni corrispondono a 24 ore.
- **Soglie separate per detector forecast-based**: `--arima-threshold-z` e `--prophet-threshold-z`, così ARIMA può essere reso più conservativo senza modificare Prophet.
- **Matrix Profile parametrico**: `--mp-zscore-threshold` e `--mp-top-k` controllano quanto severa deve essere la selezione dei discords.
- **Prophet con timestamp reali**: nei run reali il file `prophet_scores.csv` usa le date originali, non più date sintetiche.
- **Classificazione eventi**: `ensemble_anomaly_events.csv` contiene `event_class`, contributi per detector e statistiche di valore per evento.
- **Report HTML automatico**: `report.html` riassume configurazione, detector, voti, top eventi, statistiche mensili/orarie e grafici.
- **Log Prophet/CmdStan più puliti**: i log informativi vengono ridotti a warning.

Per disabilitare il report:

```bash
tsad-feature real-run ... --no-report
```

## 5. Feature estratte

Dominio del tempo:

- mean, std, var, RMS, min, max, peak-to-peak
- median, MAD, skewness, kurtosis
- zero crossing rate, crest factor

Dominio frequenza:

- dominant frequency
- main peak power
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral entropy
- band energies
- peak ratio
- THD, se viene fornita una frequenza fondamentale

Dinamiche:

- autocorrelation lag 1
- trend slope

## 6. Esempi rapidi

Solo feature + Matrix Profile su CSV generico:

```bash
tsad-feature run \
  --csv path/to/signal.csv \
  --value-col value \
  --sampling-rate 1000 \
  --window-size 1024 \
  --overlap 0.5
```

Solo ARIMA:

```bash
tsad-feature forecast \
  --csv outputs/ED14_ARCH_FM_TotW/clean_series.csv \
  --value-col value \
  --model arima \
  --arima-order 2,1,2
```

Solo Prophet:

```bash
tsad-feature forecast \
  --csv outputs/ED14_ARCH_FM_TotW/clean_series.csv \
  --value-col value \
  --model prophet \
  --prophet-freq 15min
```

## 7. Note sui dati reali

I CSV contengono misure elettriche multi-meter: tensioni, correnti, potenza, energia, frequenza, power factor e THD. La pipeline filtra una singola serie attraverso `--meter-id` e `--value-col`, poi ordina i timestamp, elimina duplicati, resampla e interpola i gap.

Per `TotW` è sensato usare `--agg mean`. Per colonne energetiche cumulative, valuta differenziazione o aggregazione coerente prima dell'anomaly detection.

## 8. Sicurezza credenziali

Il progetto originale conteneva uno script di fetch con credenziali MongoDB in chiaro. In questa versione non vengono incluse credenziali: `legacy/fetch_template.py` usa variabili d'ambiente.


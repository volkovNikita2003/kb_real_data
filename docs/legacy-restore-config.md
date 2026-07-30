# Контракт legacy-восстановления

Документ фиксирует происхождение параметров `ExperimentConfig`, используемых
эталонным [ref/restore.py](../ref/restore.py), и определяет границу между новой
инфраструктурой и legacy-вычислениями. При адаптации изменены только импорты,
источник одной конфигурации и установка глобальных параметров решателя.
Функции `save_reference_restoration`, `run_cfg` и `run_cfg_lin_cut` сохранены
структурно эквивалентными эталону, что проверяется сравнением их AST.

Одна конфигурация legacy-кода соответствует одной паре:

```text
restore profile × measurement
```

Публичной точкой входа является:

```bash
python src/restore.py experiments/<эксперимент> \
  [--profile <профиль>] [--measurement <измерение>] \
  [--force] [--warnings-as-errors | --no-warnings]
```

Без селекторов запускается декартово произведение всех профилей и измерений.
Оба селектора можно повторять. Перед вычислениями загружаются общие результаты
калибровки и DARL и валидируются все выбранные пары. Наличие ошибки или
предупреждения при `--warnings-as-errors` останавливает весь выбранный запуск
до создания новых результатов.

## Входы этапа

Восстановление должно читать только:

- `input_parameters/general.yaml`;
- выбранный `input_parameters/restore_profiles/<profile>.yaml`;
- данные выбранного `data/<measurement>/`;
- `output/calibration/result.yaml`;
- `output/darl/result.yaml`.

`calibration.yaml`, `darl.yaml` и `output/darl/used-parameters.yaml` не являются
входами восстановления. Файл `darl/result.yaml` содержит машинный контракт:
детекторы, семантику сигнала, матрицу, классы частиц, бины и доступные
моделируемые сигналы.

Все пути к артефактам предыдущих этапов разрешаются относительно
`output/calibration/` или `output/darl/`. Пути из `result.yaml` не должны
выходить за пределы соответствующего каталога результата.

## Основные пути `ExperimentConfig`

Для автоматизированного запуска `dir_case` равен корню эксперимента. Благодаря
этому все остальные legacy-пути можно передавать относительными и сохранять в
штатном `params.txt` без привязки к текущей рабочей директории процесса.

| Поле legacy | Новый источник | Правило |
|---|---|---|
| `dir_case` | `Experiment.path` | абсолютный корень эксперимента |
| `dir_save_rel` | `Experiment.restore_output_dir(profile, measurement)` | временный транзакционный каталог относительно корня эксперимента |
| `dir_save_preprocessing_rel` | не используется восстановлением | сохраняется значение по умолчанию |
| `dir_signal_rel` | `Measurement.camera_dir` | `data/<measurement>/cam` относительно эксперимента |
| `dir_back_rel` | `Measurement.camera_background_dir` | путь задаётся только при `camera.use_background: true` |
| `dir_signal_lin_rel` | `Measurement.line_dir` | `data/<measurement>/lin` относительно эксперимента |
| `dir_back_lin_rel` | `Measurement.line_background_dir` | путь задаётся только при `line_sensor.use_background: true` |
| `dir_save` | вычисляемое свойство | `dir_case / dir_save_rel` |

## Оптическая схема

| Поле legacy | Новый источник | Преобразование |
|---|---|---|
| `labm_um` | `general.instrument.wavelength_um` | без преобразования |
| `F_lens_um` | `general.instrument.focal_length_um` | без преобразования |
| `detector_configuration_type` | `general.instrument.detector_position` | `old` → 0, `new` → 1 |

Эти значения должны входить в фактически использованные параметры, даже если
конкретная legacy-функция не обращается к части из них напрямую.

## Результат калибровки камеры

| Поле legacy | Новый источник |
|---|---|
| `W` | `general.detectors.camera.width_px` |
| `H` | `general.detectors.camera.height_px` |
| `cam_pixel_width_m` | `calibration/result.yaml: camera.pixel_width_m` |
| `x_shift_m` | `calibration/result.yaml: camera.x_shift_m` |
| `y_shift_m` | `calibration/result.yaml: camera.y_shift_m` |

Свойства `x_shift_pix` и `y_shift_pix` вычисляются самим `ExperimentConfig` как
отношение смещения к скорректированной ширине пикселя.

Поля `calib_d_pinhole_um`, `calib_cam_gaussian_sigma` и
`calib_cam_corrected` нужны этапу калибровки, но не вычислениям восстановления.
При переносе они сохраняют legacy-значения по умолчанию и не становятся
пользовательскими параметрами restore profile.

## HDR камеры

Параметры HDR относятся к конкретному способу восстановления, поэтому должны
храниться в restore profile, а не в `calibration.yaml`.

| Поле legacy | Поле restore profile | Legacy-значение эталона |
|---|---|---|
| `cam_hdr_files_mode` | внутреннее фиксированное значение | `bmp` |
| `cam_hdr_mode` | `detectors.camera.hdr.mode` | `l2h` |
| `cam_hdr_diff_mode` | `detectors.camera.hdr.difference_mode` | `after_hdr` или `per_exposure` |
| `cam_hdr_back_level` | `detectors.camera.hdr.background_level` | 12 |
| `cam_hdr_low_thr` | `detectors.camera.hdr.low_threshold` | 10 |
| `cam_hdr_top_thr` | `detectors.camera.hdr.top_threshold` | 240 |
| `cam_hdr_filtered` | `detectors.camera.hdr.filtered` | `false` |
| `cam_hdr_gauss_sigma` | `detectors.camera.hdr.gaussian_sigma` | 5 |
| `cam_hdr_exposure_coefs` | пока не поддерживается | `None` |
| `exposure_time_arr` | имена файлов `data/<measurement>/cam/*.bmp` | положительные целые числа, сортируются по возрастанию |

Если включено вычитание фона, валидатор до запуска требует фон для каждой
экспозиции сигнала. Лишние экспозиции фона являются предупреждением и не
добавляются в `exposure_time_arr`.

## Линейка

| Поле legacy | Новый источник | Правило |
|---|---|---|
| `filename_lin_template` | строгая структура измерения | `{}.txt` |
| `filename_lin_back_template` | строгая структура измерения | `{}.txt` |
| `exposure_time_us_lin_arr` | имя единственного файла `lin/*.txt` | один положительный `int` |
| `num_pix_lin` | `general.detectors.line_sensor.pixel_count` | без преобразования |
| `width_pix_x_m` | `calibration/result.yaml: line_sensor.pixel_width_m` | без преобразования |
| `width_pix_y_m` | `calibration/result.yaml: line_sensor.pixel_height_m` | без преобразования |
| `coef_lin_to_cam` | `calibration/result.yaml: line_sensor.to_camera_coefficient` | без преобразования |
| `shift_lin_m` | `calibration/result.yaml: line_sensor.shift_m` | без преобразования |
| `pix_max_ampl` | `calibration/result.yaml: line_sensor.peak_pixel` | без преобразования |
| `lin_time_add` | `detectors.line_sensor.time_offset_us` restore profile | эталонное значение 2 мкс |
| `lin_signal_mode` | `detectors.line_sensor.signal_mode` restore profile | эталонное значение 2 |

Поля `calib_lin_position_pinhole_m`, `calib_lin_position_signal_m` и
`calib_lin_gaussian_sigma` восстановлением не используются и сохраняют
legacy-значения по умолчанию.

## Контракт DARL

| Поле legacy | Новый источник |
|---|---|
| `matrix_name` | `darl/result.yaml: matrix_file` |
| `classes_name` | `darl/result.yaml: particle_classes_file` |
| `bins_name` | файл камеры из `darl/result.yaml: detector_bin_files` |
| `bins_lin_name` | файл линейки из `darl/result.yaml: detector_bin_files` |
| `signal_type` | `darl/result.yaml: signal.value_type` |
| `path_signal_darl_rel` | `distributions[measurement].modeled_signal`, если такое распределение существует |

Набор `darl/result.yaml: detectors` должен содержать каждый детектор,
включённый в restore profile. Профиль может использовать подмножество
детекторов аппаратной матрицы: для режима только с камерой legacy-код обрезает
строки матрицы по количеству камерных бинов.

Отсутствие распределения с именем измерения допустимо. В этом случае
`path_signal_darl_rel = None`, сравнение реального и ожидаемого сигналов не
строится, а восстановление продолжается.

## Параметры обратной задачи

Часть настроек хранится в `ExperimentConfig`, а порядок регуляризации и alpha
в эталонном `func.py` являются глобальными величинами. Адаптер обязан задать
обе группы перед запуском одной конфигурации.

Реализованный адаптер возвращает `LegacyRestoreConfigArtifact`, содержащий
готовый `ExperimentConfig`, отдельные `LegacyRestoreSolverSettings` для
глобальных величин решателя и выбранный режим `camera`/`camera_line`. Сам
адаптер не изменяет глобальное состояние и не запускает вычисления.

Функция `run_legacy_restore()` временно устанавливает эти глобальные значения,
вызывает ровно одну legacy-функцию и восстанавливает прежнее состояние даже
при исключении. Генераторы жёстко заданных экспериментов `make_cfg_arr()` и
`make_cfg_arr_from_params()` из автоматизированной копии удалены.

| Legacy-параметр | Новый источник | Значение по умолчанию |
|---|---|---|
| `REGULARIZATION_TYPE` | `solver.regularization_order` | 1 |
| `REGULARIZATION_ALPHA` | `solver.regularization_alpha` | `best` |
| `use_chahine` | `solver.use_chahine` | `true` |
| `use_conc_corr` | `solver.use_concentration_correction` | `true` |
| `use_w_critical` | `solver.use_w_critical` | `false` |
| `W_CRITICAL` | внутренний параметр совместимости | `1e-3` |
| `cut_classes` | `class_slice.drop_first` | 10 |
| `cut_classes_top` | `class_slice.drop_last` | `null` |

Тип решателя `solver.type` на этапе первичного переноса допускает только
`tikhonov_nnls`, как уже установлено текущей YAML-схемой.

## Выбор вычислительной функции

Legacy-файл содержит два рабочих пути:

- `run_cfg` — совместное восстановление по камере и линейке;
- `run_cfg_lin_cut` — восстановление только по камере с обрезкой строк матрицы
  перед линейкой.

Первичный перенос поддерживает следующие наборы:

| Детекторы профиля | Legacy-функция |
|---|---|
| `camera`, `line_sensor` | `run_cfg` |
| только `camera` | `run_cfg_lin_cut` |

Профиль только с линейкой пока не имеет эквивалентной функции в эталонном
`restore.py` и должен отклоняться строгой проверкой до начала вычислений.

## Сохранение параметров и результатов

Для каждой пары создаётся отдельный результат:

```text
output/restore/<profile>/<measurement>/
├── used-parameters.yaml
├── params.txt
├── result.yaml
└── <legacy-артефакты>
```

`params.txt` создаётся исключительно штатным
`ExperimentConfig.save_params()`. Отдельный `legacy-parameters.yaml` не
создаётся. `used-parameters.yaml` содержит полный нормализованный снимок новой
системы, а `result.yaml` — строгий манифест выходных файлов.

Манифест разделяет сигналы камеры, линейки и их объединение, полный результат
решателя и опциональный результат после среза классов. Сравнение с ожидаемым
сигналом допускается только при наличии соответствующего распределения DARL.
Все пути должны быть безопасными и относительными; обязательные численные
файлы должны входить в общий список `artifacts`, а диагностические PNG — также
в список `figures`.

Каждая пара публикуется отдельной транзакцией. Без `--force` существующий
каталог результата не перезаписывается. С `--force` он переносится в
`output/archive/restore/<profile>/<measurement>/<timestamp>/`, после чего
временный каталог нового результата атомарно занимает рабочий путь.

## Эталонные проверки

Численная эквивалентность проверяется по текстовым файлам в каталогах
`ref/test_17_07_26_kmk_15/kmk_15/restore_*`. Обязательны точные сравнения:

- сигналов камеры, линейки и объединённого сигнала;
- восстановленных распределений;
- GCV-кривых и выбранных alpha;
- результатов после `cut_classes`;
- сравнения реального и моделируемого сигналов, когда оно доступно.

PNG-файлы проверяются на наличие, но не используются как основной численный
критерий.

Для профиля `default` и измерения `kmk_15` эталонного эксперимента выполнен
реальный запуск нового CLI. Побайтово совпали девять основных файлов: сигналы
камеры, линейки и объединённый сигнал, полное и срезанное восстановления,
соответствующие GCV-кривые и значения alpha.

Полный расчёт остаётся локальной интеграционной проверкой, поскольку исходные
BMP/TXT эксперимента исключены из Git. Обязательные быстрые тесты репозитория
проверяют AST-эквивалентность перенесённых численных функций, адаптер
конфигурации, временную установку параметров решателя, строгий манифест,
валидацию и пакетную логику CLI. Поэтому изменение вычислительного кода
обнаруживается без запуска тяжёлого эксперимента, а окончательная численная
проверка выполняется на локально доступном эталонном наборе.
